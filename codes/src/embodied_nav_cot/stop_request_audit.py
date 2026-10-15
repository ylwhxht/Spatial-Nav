"""Opt-in capture of Stop SDK arguments, independent of visualizations."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def record_stop_exchange(request: dict, content: str, binary: dict) -> None:
    """Store model inputs and parsed outputs; never store client headers/API keys.

    Disabled by default. Captures include inline image bytes and can be large.
    They represent SDK arguments, not a packet capture or server tokens.
    An enabled audit write failure must surface instead of silently losing data.
    """
    directory = os.environ.get("VLMNAV_STOP_REQUEST_AUDIT_DIR", "")
    if not directory:
        return
    body = json.dumps(request, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    root = Path(directory)
    requests = root / "requests"
    responses = root / "responses"
    requests.mkdir(parents=True, exist_ok=True)
    responses.mkdir(parents=True, exist_ok=True)
    try:
        with (requests / f"{digest}.json").open("xb") as handle:
            handle.write(body)
    except FileExistsError:
        pass  # Identical request bodies share their image/prompt artifact.
    record = {"request_sha256": digest, "pid": os.getpid(), "time_ns": time.time_ns(),
              "content": content, "binary": binary}
    with (responses / f"{uuid.uuid4().hex}.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
