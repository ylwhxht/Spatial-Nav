import json
import logging
import os
import time
import torch
import numpy as np
import cv2

from PIL import Image
from utils import append_mime_tag, encode_image_b64, resize_image_if_needed


class VLM:
    """
    Base class for a Vision-Language Model (VLM) agent.
    This class should be extended to implement specific VLMs.
    """

    def __init__(self, **kwargs):
        """
        Initializes the VLM agent with optional parameters.
        """
        self.name = "not implemented"

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform inference with the VLM agent, passing images and a text prompt.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError

    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the VLM, incorporating past context.

        Parameters
        ----------
        history : int
            The number of context steps to keep for inference.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the context state of the VLM agent.
        """
        pass

    def rewind(self):
        """
        Rewind the VLM agent one step by removing the last inference context.
        """
        pass

    def get_spend(self):
        """
        Retrieve the total cost or spend associated with the agent.
        """
        return 0


class GeminiVLM(VLM):
    """
    A specific implementation of a VLM using the Gemini API for image and text inference.
    """

    def __init__(self, model="gemini-1.5-flash", system_instruction=None):
        """
        Initialize the Gemini model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        import google.generativeai as genai  # Optional legacy client; not needed by local policies.

        self.name = model
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        # Configure generation parameters
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 500,
            "response_mime_type": "text/plain",
        }

        self.spend = 0
        self.cost_per_input_token = 0.075 / 1_000_000 if 'flash' in self.name else 1.25 / 1_000_000
        self.cost_per_output_token = 0.3 / 1_000_000 if 'flash' in self.name else 5 / 1_000_000

        # Initialize Gemini model and chat session
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=self.generation_config,
            system_instruction=system_instruction
        )
        self.session = self.model.start_chat(history=[])

    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the Gemini model.

        Parameters
        ----------
        history : int
            The number of environment steps to keep in context.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        images = [Image.fromarray(image[:, :, :3], mode='RGB') for image in images]
        try:
            response = self.session.send_message([text_prompt] + images)
            self.spend += (response.usage_metadata.prompt_token_count * self.cost_per_input_token +
                           response.usage_metadata.candidates_token_count * self.cost_per_output_token)

            # Manage history length based on the number of past steps to keep
            if history == 0:
                self.session = self.model.start_chat(history=[])
            elif len(self.session.history) > 2 * history:
                self.session.history = self.session.history[-2 * history:]

        except Exception as e:
            logging.error(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"

        return response.text

    def rewind(self):
        """
        Rewind the chat history by one step.
        """
        if len(self.session.history) > 1:
            self.model.rewind()

    def reset(self):
        """
        Reset the chat history.
        """
        self.session = self.model.start_chat(history=[])

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform contextless inference with the Gemini model.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        images = [Image.fromarray(image[:, :, :3], mode='RGB') for image in images]
        try:
            response = self.model.generate_content([text_prompt] + images)
            self.spend += (response.usage_metadata.prompt_token_count * self.cost_per_input_token +
                           response.usage_metadata.candidates_token_count * self.cost_per_output_token)

        except Exception as e:
            logging.error(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"

        return response.text

    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        """
        return self.spend


class OpenAIVLM(VLM):
    """
    An implementation using models served via OpenAI API.
    """

    def __init__(
        self,
        model="gpt-4o-latest",
        system_instruction=None,
        max_image_res=None,
        image_encode_format="png",
        jpeg_quality=85,
        max_images_per_prompt=None,
        use_system_instruction=False,
        send_sampling_args=False,
        sampling_args_env_name=None,
        temperature=None,
        max_tokens=None,
        base_url=None,
        api_key=None,
        timeout_s=None,
        send_max_tokens=False,
        send_max_tokens_env_name=None,
    ):
        """
        Initialize the OpenAI model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        from openai import OpenAI
        self.name = model
        self.model = model
        self.system_instruction = (system_instruction or "").strip()
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        # The original VLMNav OpenAI adapter did not impose a short per-request
        # timeout. Keep this long by default so slow vLLM generations do not
        # spuriously retry in the middle of an episode.
        timeout_s = float(timeout_s if timeout_s is not None else os.environ.get("OPENAI_TIMEOUT", "600"))
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_s,
            max_retries=1,
        )
        self.history = []
        self.max_image_res = max_image_res
        self.image_encode_format = (image_encode_format or "png").lower()
        self.jpeg_quality = jpeg_quality
        self.max_images_per_prompt = None if max_images_per_prompt is None else int(max_images_per_prompt)
        self.use_system_instruction = self._coerce_bool_env(
            "VLMNAV_OPENAI_USE_SYSTEM_INSTRUCTION",
            use_system_instruction,
        )
        sampling_args_env_name = str(
            sampling_args_env_name or "VLMNAV_OPENAI_SEND_SAMPLING_ARGS"
        )
        self.send_sampling_args = self._coerce_bool_env(
            sampling_args_env_name,
            send_sampling_args,
            fallback_env_name="VLMNAV_OPENAI_SEND_SAMPLING_ARGS",
        )
        send_max_tokens_env_name = str(
            send_max_tokens_env_name or "VLMNAV_OPENAI_SEND_MAX_TOKENS"
        )
        self.send_max_tokens = self._coerce_bool_env(
            send_max_tokens_env_name,
            send_max_tokens,
            fallback_env_name="VLMNAV_OPENAI_SEND_MAX_TOKENS",
        )
        # Temperature remains coupled to the legacy sampling switch.  The new
        # length-only switch is separately opt-in, so historical requests keep
        # omitting both fields while strict callers may send max_tokens without
        # changing the backend's sampling defaults.
        self._chat_temperature = float(
            temperature if temperature is not None else os.environ.get("OPENAI_TEMPERATURE", "0.01")
        )
        self._chat_max_tokens = int(
            max_tokens if max_tokens is not None else os.environ.get("OPENAI_MAX_TOKENS", "512")
        )
        self.last_completion_tokens = None

    @staticmethod
    def _coerce_bool_env(env_name: str, default, fallback_env_name=None) -> bool:
        value = os.environ.get(env_name)
        if value is None and fallback_env_name and fallback_env_name != env_name:
            value = os.environ.get(fallback_env_name)
        if value is None:
            value = default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

    @staticmethod
    def _is_internvl_model(model_name: str) -> bool:
        return "internvl" in (model_name or "").lower()

    @classmethod
    def _merge_system_into_user_for_openai_vl(cls, model_name: str) -> bool:
        """
        Some OpenAI-compatible VL backends (vLLM + InternVL, and some Qwen-VL builds)
        error on role=system together with multimodal user content; merge system into user text.
        """
        m = (model_name or "").lower()
        if "internvl" in m:
            return True
        if "qwen" in m and "vl" in m:
            return True
        return False

    def _user_text_for_prompt(self, text_prompt: str) -> str:
        if not self.use_system_instruction or not self.system_instruction:
            return text_prompt
        if self._merge_system_into_user_for_openai_vl(self.model):
            return f"{self.system_instruction}\n\n{text_prompt}"
        return text_prompt

    def _system_prefix_messages(self):
        if (
            self.use_system_instruction
            and self.system_instruction
            and not self._merge_system_into_user_for_openai_vl(self.model)
        ):
            return [{"role": "system", "content": self.system_instruction}]
        return []

    def _completion_content(self, response, messages):
        content = response.choices[0].message.content
        if (os.environ.get("VLMNAV_ACTION_CANONICAL_MOVE") == "1"
                and self.model == os.environ.get("VLMNAV_ACTION_MODEL")):
            from embodied_nav_cot.move_sc_rle_protocol import public_move_inputs_from_messages
            from embodied_nav_cot.sc_esr_move_protocol import isolate_valid_move_json
            _, legal = public_move_inputs_from_messages(messages)
            return isolate_valid_move_json(str(content or ""), legal=legal)
        return content

    def _completion_kwargs(self, messages):
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if (os.environ.get("VLMNAV_ACTION_CANONICAL_MOVE") == "1"
                and self.model == os.environ.get("VLMNAV_ACTION_MODEL")):
            from embodied_nav_cot.sc_esr_move_protocol import canonicalize_sc_esr_openai_messages
            kwargs["messages"] = canonicalize_sc_esr_openai_messages(messages)
            kwargs.update(top_p=1.0, n=1, stream=False)
        template_kwargs = os.environ.get("VLMNAV_OPENAI_CHAT_TEMPLATE_KWARGS")
        if template_kwargs:
            parsed_template_kwargs = json.loads(template_kwargs)
            if not isinstance(parsed_template_kwargs, dict):
                raise ValueError("VLMNAV_OPENAI_CHAT_TEMPLATE_KWARGS must be a JSON object")
            kwargs["extra_body"] = {"chat_template_kwargs": parsed_template_kwargs}
        if self.send_sampling_args:
            kwargs["temperature"] = self._chat_temperature
        if self.send_sampling_args or self.send_max_tokens:
            kwargs["max_tokens"] = self._chat_max_tokens
        return kwargs

    def _record_completion_tokens(self, response) -> None:
        usage = getattr(response, "usage", None)
        value = getattr(usage, "completion_tokens", None)
        if value is None and isinstance(usage, dict):
            value = usage.get("completion_tokens")
        try:
            self.last_completion_tokens = int(value) if value is not None else None
        except (TypeError, ValueError):
            self.last_completion_tokens = None

    def call_guided_choice(
        self,
        images: list[np.array],
        text_prompt: str,
        choices: list[str],
        history: int = 0,
    ):
        """
        Ask an OpenAI-compatible backend to choose exactly one string.

        vLLM supports guided_choice through extra_body on recent builds.  This
        helper is deliberately side-effect free: it does not update chat
        history, so it can be used as a shadow action-head approximation.
        """
        choices = [str(choice) for choice in choices if str(choice).strip()]
        if not choices:
            return ""

        text_contents = [{
            "type": "text",
            "text": self._user_text_for_prompt(text_prompt),
        }]
        image_contents = self._image_contents_from_images(images)
        messages = [{"role": "user", "content": text_contents + image_contents}]
        prefix = self._system_prefix_messages()
        hist = [] if history == 0 else self.history[-2 * int(history):]

        kwargs = self._completion_kwargs(prefix + hist + messages)
        kwargs["temperature"] = float(os.environ.get("VLMNAV_GUIDED_CHOICE_TEMPERATURE", "0.0"))
        kwargs["max_tokens"] = int(os.environ.get("VLMNAV_GUIDED_CHOICE_MAX_TOKENS", "16"))
        use_extra_body = self._coerce_bool_env("VLMNAV_GUIDED_CHOICE_USE_EXTRA_BODY", True)
        if use_extra_body:
            kwargs.setdefault("extra_body", {})["guided_choice"] = choices

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            if not use_extra_body:
                logging.error(f"OPENAI GUIDED CHOICE ERROR: {e}")
                return "OPENAI API ERROR"
            logging.warning("guided_choice extra_body failed; retrying as plain constrained prompt: %s", e)

        fallback_prompt = (
            text_prompt.rstrip()
            + "\n\nConstrained action-choice retry: return exactly one of these labels and no other text:\n"
            + ", ".join(choices)
        )
        text_contents = [{
            "type": "text",
            "text": self._user_text_for_prompt(fallback_prompt),
        }]
        messages = [{"role": "user", "content": text_contents + image_contents}]
        kwargs = self._completion_kwargs(prefix + hist + messages)
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = int(os.environ.get("VLMNAV_GUIDED_CHOICE_MAX_TOKENS", "16"))
        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OPENAI GUIDED CHOICE FALLBACK ERROR: {e}")
            return "OPENAI API ERROR"

    @staticmethod
    def _logprob_entry_to_dict(entry):
        token = getattr(entry, "token", None)
        if token is None and isinstance(entry, dict):
            token = entry.get("token")
        logprob = getattr(entry, "logprob", None)
        if logprob is None and isinstance(entry, dict):
            logprob = entry.get("logprob")
        try:
            logprob = float(logprob)
        except Exception:
            logprob = None
        return {"token": "" if token is None else str(token), "logprob": logprob}

    @classmethod
    def _action_label_suffix_from_token(cls, token: str):
        text = str(token or "").strip().lower()
        text = text.replace("−", "-").replace("–", "-").replace("—", "-")
        text = text.strip("\"'`.,;:()[]{}\n\r\t ")
        if text.startswith("candidate_"):
            text = text.split("_", 1)[1]
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    @staticmethod
    def _binary_label_from_token(token: str):
        text = str(token or "").strip().lower()
        text = text.strip("\"'`.,;:()[]{}\n\r\t ")
        if text in ("0", "1"):
            return text
        if text.startswith("0"):
            return "0"
        if text.startswith("1"):
            return "1"
        return None

    @classmethod
    def _stop_binary_logprobs_from_response(cls, response):
        """Extract the final JSON binary value and its 0/1 token logprobs."""
        choice = response.choices[0]
        content = choice.message.content or ""
        logprobs_obj = getattr(choice, "logprobs", None)
        lp_content = getattr(logprobs_obj, "content", None)
        if lp_content is None and isinstance(logprobs_obj, dict):
            lp_content = logprobs_obj.get("content")

        selected_position = None
        for index in range(len(lp_content or []) - 1, -1, -1):
            entry = lp_content[index]
            token = getattr(entry, "token", None)
            if token is None and isinstance(entry, dict):
                token = entry.get("token")
            if cls._binary_label_from_token(token) in ("0", "1"):
                selected_position = index
                break

        payload = {
            "status": "no_binary_token_logprobs",
            "generated": content,
            "generated_label": None,
            "token_position": selected_position,
            "top_logprobs": [],
            "scores": {"0": None, "1": None},
            "margin_stop_minus_continue": None,
            "valid": False,
        }
        if selected_position is None:
            return payload

        entry = lp_content[selected_position]
        generated_token = getattr(entry, "token", None)
        if generated_token is None and isinstance(entry, dict):
            generated_token = entry.get("token")
        payload["generated_label"] = cls._binary_label_from_token(generated_token)
        top_entries = getattr(entry, "top_logprobs", None)
        if top_entries is None and isinstance(entry, dict):
            top_entries = entry.get("top_logprobs")
        if not top_entries:
            top_entries = [entry]
        top_logprobs = [cls._logprob_entry_to_dict(item) for item in top_entries]
        payload["top_logprobs"] = top_logprobs

        scores = {"0": None, "1": None}
        for item in top_logprobs:
            label = cls._binary_label_from_token(item.get("token"))
            score = item.get("logprob")
            if label not in scores or score is None:
                continue
            if scores[label] is None or score > scores[label]:
                scores[label] = score
        payload["scores"] = scores
        if scores["0"] is None or scores["1"] is None:
            payload["status"] = "missing_binary_label_logprob"
            return payload

        payload["margin_stop_minus_continue"] = scores["1"] - scores["0"]
        payload["status"] = "ok"
        payload["valid"] = True
        return payload

    def call_stop_with_logprobs(self, images: list[np.array], text_prompt: str):
        """Run the unchanged Stop prompt and return its binary decision logits."""
        text_contents = [{
            "type": "text",
            "text": self._user_text_for_prompt(text_prompt),
        }]
        image_contents = self._image_contents_from_images(images)
        messages = [{"role": "user", "content": text_contents + image_contents}]
        kwargs = self._completion_kwargs(self._system_prefix_messages() + messages)
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = int(os.environ.get(
            "VLMNAV_STOP_LOGIT_TOP_LOGPROBS", "20"
        ))

        max_retries = 5
        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._record_completion_tokens(response)
                content = response.choices[0].message.content or ""
                binary = self._stop_binary_logprobs_from_response(response)
            except Exception as e:
                logging.error(
                    "OPENAI STOP LOGPROB ERROR (attempt %s/%s): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
            else:
                from embodied_nav_cot.stop_request_audit import record_stop_exchange
                record_stop_exchange(kwargs, content, binary)
                return content, binary
        return "OPENAI API ERROR", {
            "status": "error",
            "error": "stop_logprob_request_failed",
            "valid": False,
        }

    def call_chat_stop_with_logprobs(
        self,
        history: int,
        images: list[np.array],
        text_prompt: str,
        max_tokens: int,
    ):
        """Run the final staged Stop decision over the retained chat prefix."""
        max_tokens = int(max_tokens)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if history == 0:
            self.history = []
        text_contents = [{
            "type": "text",
            "text": self._user_text_for_prompt(text_prompt),
        }]
        image_contents = self._image_contents_from_images(images)
        messages = [{"role": "user", "content": text_contents + image_contents}]
        kwargs = self._completion_kwargs(
            self._system_prefix_messages() + self.history + messages
        )
        kwargs["max_tokens"] = max_tokens
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = int(os.environ.get(
            "VLMNAV_STOP_LOGIT_TOP_LOGPROBS", "20"
        ))

        max_retries = 5
        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._record_completion_tokens(response)
                content = response.choices[0].message.content or ""
                return content, self._stop_binary_logprobs_from_response(response)
            except Exception as e:
                logging.error(
                    "OPENAI STAGED STOP LOGPROB ERROR (attempt %s/%s): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
        self.last_completion_tokens = None
        return "OPENAI API ERROR", {
            "status": "error",
            "error": "staged_stop_logprob_request_failed",
            "valid": False,
        }

    @staticmethod
    def _trim_action_json_contract(text_prompt: str) -> str:
        text = str(text_prompt or "")
        cut = len(text)
        for marker in (
            "\nReturn exactly one compact JSON object",
            "\nStrict constraints:",
            "\nConstrained action-choice pass:",
            "\nAction-label scoring pass:",
            "\nAction-stop binary scoring pass:",
        ):
            idx = text.find(marker)
            if idx >= 0:
                cut = min(cut, idx)
        return text[:cut].rstrip()

    def score_action_labels(
        self,
        images: list[np.array],
        text_prompt: str,
        choices: list[str],
        history: int = 0,
    ):
        """
        Score valid action labels from first-token logprobs.

        This is a shadow-friendly approximation of an action head.  The prompt
        asks for a single integer action id and the backend returns top
        logprobs for that first token, so candidate ids can be compared without
        generating a full JSON response.  It is not a true classifier head;
        missing top-logprob labels are logged explicitly and should fail the
        shadow gate if common.
        """
        choices = [str(choice).strip() for choice in choices if str(choice).strip()]
        valid_numbers = {}
        for choice in choices:
            number = self._action_label_suffix_from_token(choice)
            if number is not None:
                valid_numbers[number] = f"candidate_{number}"
        if not valid_numbers:
            return {
                "status": "error",
                "error": "no_valid_choices",
                "choices": choices,
                "valid": False,
            }

        base_prompt = self._trim_action_json_contract(text_prompt)
        valid_ids = ", ".join(str(num) for num in sorted(valid_numbers))
        scorer_prompt = (
            base_prompt.rstrip()
            + "\n\nAction-label scoring pass:\n"
            "- Choose the single best next action id from the valid list.\n"
            "- Candidate numbers are identifiers, not preferences.\n"
            "- Return only one integer id.\n"
            "- Do not write candidate_, JSON, braces, quotes, markdown, or prose.\n"
            f"Valid labels: {', '.join(choices)}\n"
            f"Valid action ids: {valid_ids}\n"
            "Best action id:"
        )
        text_contents = [{
            "type": "text",
            "text": self._user_text_for_prompt(scorer_prompt),
        }]
        image_contents = self._image_contents_from_images(images)
        messages = [{"role": "user", "content": text_contents + image_contents}]
        prefix = self._system_prefix_messages()
        hist = [] if history == 0 else self.history[-2 * int(history):]

        kwargs = self._completion_kwargs(prefix + hist + messages)
        kwargs["temperature"] = float(os.environ.get("VLMNAV_ACTION_LABEL_SCORER_TEMPERATURE", "0.0"))
        kwargs["max_tokens"] = int(os.environ.get("VLMNAV_ACTION_LABEL_SCORER_MAX_TOKENS", "1"))
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = int(os.environ.get("VLMNAV_ACTION_LABEL_SCORER_TOP_LOGPROBS", "20"))

        try:
            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            content = choice.message.content or ""
            logprobs_obj = getattr(choice, "logprobs", None)
            lp_content = getattr(logprobs_obj, "content", None)
            if lp_content is None and isinstance(logprobs_obj, dict):
                lp_content = logprobs_obj.get("content")
            first_lp = lp_content[0] if lp_content else None
            top_entries = []
            if first_lp is not None:
                top_entries = getattr(first_lp, "top_logprobs", None)
                if top_entries is None and isinstance(first_lp, dict):
                    top_entries = first_lp.get("top_logprobs")
                if not top_entries:
                    top_entries = [first_lp]
            top_logprobs = [self._logprob_entry_to_dict(item) for item in (top_entries or [])]

            scores = {label: None for label in valid_numbers.values()}
            for item in top_logprobs:
                number = self._action_label_suffix_from_token(item.get("token"))
                if number not in valid_numbers:
                    continue
                label = valid_numbers[number]
                score = item.get("logprob")
                if score is None:
                    continue
                if scores[label] is None or score > scores[label]:
                    scores[label] = score

            finite_scores = {label: score for label, score in scores.items() if score is not None}
            selected_label = None
            selected_score = None
            if finite_scores:
                selected_label, selected_score = max(finite_scores.items(), key=lambda kv: kv[1])
            else:
                generated_number = self._action_label_suffix_from_token(content)
                if generated_number in valid_numbers:
                    selected_label = valid_numbers[generated_number]

            selected_number = self._action_label_suffix_from_token(selected_label) if selected_label else None
            generated_number = self._action_label_suffix_from_token(content)
            sorted_scores = sorted(finite_scores.items(), key=lambda kv: kv[1], reverse=True)
            margin = None
            if len(sorted_scores) >= 2:
                margin = sorted_scores[0][1] - sorted_scores[1][1]
            try:
                min_label_logprob = float(os.environ.get("VLMNAV_ACTION_LABEL_SCORER_MIN_LOGPROB", "-8.0"))
            except ValueError:
                min_label_logprob = -8.0
            confident_label = selected_score is not None and selected_score >= min_label_logprob
            generated_is_label = generated_number in valid_numbers
            status = "ok"
            if finite_scores and not confident_label and not generated_is_label:
                status = "label_logprob_below_threshold"
                selected_label = None
                selected_number = None
            elif not finite_scores and generated_is_label:
                status = "generated_label_no_scores"
            elif not finite_scores:
                status = "no_label_logprobs"
            return {
                "status": status,
                "method": "next_token_action_id",
                "choices": choices,
                "generated": content,
                "top_logprobs": top_logprobs,
                "scores": scores,
                "missing_labels": [label for label, score in scores.items() if score is None],
                "action": selected_label,
                "action_number": selected_number,
                "score": selected_score,
                "score_margin": margin,
                "min_label_logprob": min_label_logprob,
                "valid": selected_number in valid_numbers,
            }
        except Exception as e:
            logging.error(f"OPENAI ACTION LABEL SCORER ERROR: {e}")
            return {
                "status": "error",
                "error": repr(e),
                "choices": choices,
                "valid": False,
            }

    def score_action_stop_binary(
        self,
        images: list[np.array],
        text_prompt: str,
        history: int = 0,
    ):
        """Score a 0/1 action-stop auxiliary label from first-token logprobs."""
        base_prompt = self._trim_action_json_contract(text_prompt)
        scorer_prompt = (
            base_prompt.rstrip()
            + "\n\nAction-stop binary scoring pass:\n"
            "- Decide whether candidate_-1 is the single best executable action now.\n"
            "- Return 1 only if the goal object is visually confirmed and close enough to finish.\n"
            "- Return 0 if the goal is absent, uncertain, occluded, or still needs any approach.\n"
            "- Candidate numbers are identifiers, not preferences.\n"
            "- Return exactly one digit and no other text.\n"
            "Valid labels: 0, 1\n"
            "Stop label:"
        )
        text_contents = [{
            "type": "text",
            "text": self._user_text_for_prompt(scorer_prompt),
        }]
        image_contents = self._image_contents_from_images(images)
        messages = [{"role": "user", "content": text_contents + image_contents}]
        prefix = self._system_prefix_messages()
        hist = [] if history == 0 else self.history[-2 * int(history):]

        kwargs = self._completion_kwargs(prefix + hist + messages)
        kwargs["temperature"] = float(os.environ.get("VLMNAV_ACTION_STOP_BINARY_SCORER_TEMPERATURE", "0.0"))
        kwargs["max_tokens"] = int(os.environ.get("VLMNAV_ACTION_STOP_BINARY_SCORER_MAX_TOKENS", "1"))
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = int(os.environ.get("VLMNAV_ACTION_STOP_BINARY_SCORER_TOP_LOGPROBS", "20"))

        try:
            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            content = choice.message.content or ""
            logprobs_obj = getattr(choice, "logprobs", None)
            lp_content = getattr(logprobs_obj, "content", None)
            if lp_content is None and isinstance(logprobs_obj, dict):
                lp_content = logprobs_obj.get("content")
            first_lp = lp_content[0] if lp_content else None
            top_entries = []
            if first_lp is not None:
                top_entries = getattr(first_lp, "top_logprobs", None)
                if top_entries is None and isinstance(first_lp, dict):
                    top_entries = first_lp.get("top_logprobs")
                if not top_entries:
                    top_entries = [first_lp]
            top_logprobs = [self._logprob_entry_to_dict(item) for item in (top_entries or [])]

            scores = {"0": None, "1": None}
            for item in top_logprobs:
                label = self._binary_label_from_token(item.get("token"))
                if label not in scores:
                    continue
                score = item.get("logprob")
                if score is None:
                    continue
                if scores[label] is None or score > scores[label]:
                    scores[label] = score

            finite_scores = {label: score for label, score in scores.items() if score is not None}
            selected_label = None
            selected_score = None
            if finite_scores:
                selected_label, selected_score = max(finite_scores.items(), key=lambda kv: kv[1])
            else:
                generated_label = self._binary_label_from_token(content)
                if generated_label in scores:
                    selected_label = generated_label

            generated_label = self._binary_label_from_token(content)
            sorted_scores = sorted(finite_scores.items(), key=lambda kv: kv[1], reverse=True)
            margin = None
            if len(sorted_scores) >= 2:
                margin = sorted_scores[0][1] - sorted_scores[1][1]
            try:
                min_label_logprob = float(os.environ.get("VLMNAV_ACTION_STOP_BINARY_SCORER_MIN_LOGPROB", "-8.0"))
            except ValueError:
                min_label_logprob = -8.0
            confident_label = selected_score is not None and selected_score >= min_label_logprob
            generated_is_label = generated_label in scores
            status = "ok"
            if finite_scores and not confident_label and not generated_is_label:
                status = "label_logprob_below_threshold"
                selected_label = None
            elif not finite_scores and generated_is_label:
                status = "generated_label_no_scores"
            elif not finite_scores:
                status = "no_label_logprobs"
            return {
                "status": status,
                "method": "next_token_action_stop_binary",
                "generated": content,
                "top_logprobs": top_logprobs,
                "scores": scores,
                "missing_labels": [label for label, score in scores.items() if score is None],
                "stop_label": None if selected_label is None else int(selected_label),
                "stop_yes": selected_label == "1",
                "score": selected_score,
                "score_margin": margin,
                "min_label_logprob": min_label_logprob,
                "valid": selected_label in scores,
            }
        except Exception as e:
            logging.error(f"OPENAI ACTION STOP BINARY SCORER ERROR: {e}")
            return {
                "status": "error",
                "error": repr(e),
                "valid": False,
            }

    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the OpenAI model.

        Parameters
        ----------
        history : int
            The number of environment steps to keep in context.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        # If caller asked for no history, clear it so prompt length doesn't grow unbounded.
        if history == 0:
            self.history = []

        text_contents = [{
            "type": "text",
            "text": self._user_text_for_prompt(text_prompt),
        }]
        image_contents = self._image_contents_from_images(images)
        messages = [{"role": "user", "content": text_contents + image_contents}]
        prefix = self._system_prefix_messages()

        # Retry logic to handle transient API errors (e.g., 429 rate limits)
        max_retries = 5
        delay = 1.0

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    **self._completion_kwargs(prefix + self.history + messages)
                )
                self._record_completion_tokens(response)
                # Only update history on successful response
                content = self._completion_content(response, messages)
                self.history.append(messages[0])  # append user message
                self.history.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": content}],
                })  # append response

                # Manage history length based on the number of past steps to keep
                if history == 0:
                    self.history = []
                elif len(self.history) > 2 * history:
                    self.history = self.history[-2 * history:]

                return content

            except Exception as e:
                logging.error(f"OPENAI API ERROR (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    time.sleep(delay)
                    delay *= 2
                else:
                    return "OPENAI API ERROR"

    def call_chat_with_max_tokens(
        self,
        history: int,
        images: list[np.array],
        text_prompt: str,
        max_tokens: int,
    ):
        """Run one chat stage with an explicit completion-token ceiling."""
        max_tokens = int(max_tokens)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if history == 0:
            self.history = []

        text_contents = [{
            "type": "text",
            "text": self._user_text_for_prompt(text_prompt),
        }]
        image_contents = self._image_contents_from_images(images)
        messages = [{"role": "user", "content": text_contents + image_contents}]
        prefix = self._system_prefix_messages()
        max_retries = 5
        delay = 1.0

        for attempt in range(max_retries):
            try:
                kwargs = self._completion_kwargs(prefix + self.history + messages)
                kwargs["max_tokens"] = max_tokens
                response = self.client.chat.completions.create(**kwargs)
                self._record_completion_tokens(response)
                content = self._completion_content(response, messages) or ""
                self.history.append(messages[0])
                self.history.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": content}],
                })
                if history == 0:
                    self.history = []
                elif len(self.history) > 2 * history:
                    self.history = self.history[-2 * history:]
                return content
            except Exception as e:
                logging.error(
                    "OPENAI STAGED CHAT ERROR (attempt %s/%s): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    self.last_completion_tokens = None
                    return "OPENAI API ERROR"


    def rewind(self):
        """
        Rewind the chat history by one step.
        """
        if len(self.history) > 1:
            self.history = self.history[:-2]

    def reset(self):
        """
        Reset the chat history.
        """
        self.history = []

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform contextless inference with the Gemini model.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        text_contents = [{
            "type": "text",
            "text": self._user_text_for_prompt(text_prompt),
        }]
        image_contents = self._image_contents_from_images(images)
        messages = [{"role": "user", "content": text_contents + image_contents}]
        prefix = self._system_prefix_messages()

        max_retries = 5
        delay = 1.0

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    **self._completion_kwargs(prefix + messages)
                )
                self._record_completion_tokens(response)
                return self._completion_content(response, messages)

            except Exception as e:
                logging.error(f"OPENAI API ERROR (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    return "OPENAI API ERROR"


    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        NOTE: not implemented
        """
        return 0


    def _image_contents_from_images(self, images: list[np.ndarray]):
        # Optional cap for newer large-history runs; None matches original VLMNav.
        if self.max_images_per_prompt is not None:
            images = images[: self.max_images_per_prompt]

        def _encode(img: Image.Image) -> str:
            if self.image_encode_format in ("jpeg", "jpg"):
                return encode_image_b64(img, format="JPEG", quality=self.jpeg_quality)
            return encode_image_b64(img, format=self.image_encode_format.upper())

        image_contents = [
            {
                "type": "image_url",
                "image_url":
                    {
                        "url": append_mime_tag(
                            _encode(Image.fromarray(image[:, :, :3], mode="RGB"))
                            if self.max_image_res is None
                            else _encode(
                                resize_image_if_needed(
                                    Image.fromarray(image[:, :, :3], mode="RGB"), self.max_image_res
                                )
                            ),
                            format=("jpeg" if self.image_encode_format in ("jpeg", "jpg") else self.image_encode_format),
                        )
                    }
            }
            for image in images
        ]
        return image_contents


class DepthEstimator:
    """
    A class for depth estimation from images using a pre-trained model.
    """

    def __init__(self):
        """
        Initialize the depth estimation pipeline with the appropriate model.
        """
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = "Intel/zoedepth-kitti"
        self.pipe = pipeline("depth-estimation", model=checkpoint, device=device)

    def call(self, image: np.array):
        """
        Perform depth estimation on an image.

        Parameters
        ----------
        image : np.array
            An RGB image for depth estimation.
        """
        original_shape = image.shape
        image_rgb = Image.fromarray(image[:, :, :3])
        depth_predictions = self.pipe(image_rgb)['predicted_depth']

        # Resize the depth map back to the original image dimensions
        depth_predictions = depth_predictions.squeeze().cpu().numpy()
        depth_predictions = cv2.resize(depth_predictions, (original_shape[1], original_shape[0]))

        return depth_predictions


class Segmentor:
    """
    A class for semantic segmentation using a pre-trained model.
    """

    def __init__(self):
        """
        Initialize the segmentation model and processor.
        """
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic").to(self.device)

        # Get class ids for navigable regions
        id2label = self.model.config.id2label
        self.navigability_class_ids = [id for id, label in id2label.items() if 'floor' in label.lower() or 'rug' in label.lower()]

    def get_navigability_mask(self, im: np.array):
        """
        Generate a navigability mask from an input image.

        Parameters
        ----------
        im : np.array
            An RGB image for generating the navigability mask.
        """
        image = Image.fromarray(im[:, :, :3])
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()

        navigability_mask = np.isin(predicted_semantic_map, self.navigability_class_ids)
        return navigability_mask


# --- SNSB in-process adapter bridge ---------------------------------------
# SNSB drives navigation from shared-slot readouts rather than text generation.
# The adapter lives under this repository's codes/src tree; this thin subclass
# makes it importable as a VLMnav model_cls via
# `from vlm import *`. Import is lazy so non-SNSB runs never load torch/SNSB.
import os as _os
import sys as _sys

_EMBODIED_NAV_COT_SRC = _os.environ.get(
    "EMBODIED_NAV_COT_SRC",
    str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"),
)


def _load_snsb_vlm_cls():
    if _EMBODIED_NAV_COT_SRC and _EMBODIED_NAV_COT_SRC not in _sys.path:
        _sys.path.insert(0, _EMBODIED_NAV_COT_SRC)
    from embodied_nav_cot.snsb.vlmnav_adapter import SNSBVLM as _Impl  # type: ignore
    return _Impl


class SNSBVLM(VLM):
    """VLMnav model_cls bridge to the bundled SNSB adapter."""

    def __init__(self, **kwargs):
        impl_cls = _load_snsb_vlm_cls()
        self._impl = impl_cls(**kwargs)
        self.name = getattr(self._impl, "name", "snsb_nav")

    def call(self, images, text_prompt):
        return self._impl.call(images, text_prompt)

    def call_chat(self, history, images, text_prompt):
        return self._impl.call_chat(history, images, text_prompt)

    def score_action_labels(self, images, text_prompt, choices, history=0):
        return self._impl.score_action_labels(images, text_prompt, choices, history=history)

    def score_action_stop_binary(self, images, text_prompt, history=0):
        return self._impl.score_action_stop_binary(images, text_prompt, history=history)

    def call_stop_with_logprobs(self, images, text_prompt):
        return self._impl.call_stop_with_logprobs(images, text_prompt)

    def reset(self):
        return self._impl.reset()

    def rewind(self):
        return self._impl.rewind()

    def get_spend(self):
        return self._impl.get_spend()
