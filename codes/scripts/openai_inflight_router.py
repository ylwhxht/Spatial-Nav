#!/usr/bin/env python3
"""Local least-in-flight routing used by the historical evaluation launcher.

Forward non-streaming requests without rewriting their JSON or model responses.
The policy orders replicas by (in-flight requests, error count, backend URL),
as in EmbodiedNavCoT/scripts/openai_inflight_router.py. This is a CPU service,
not an additional model replica.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, urlunsplit

import requests


class RouterState:
    def __init__(self, backends: list[str], timeout: float) -> None:
        self.backends = [backend.rstrip("/") for backend in backends]
        self.timeout = timeout
        self.inflight = {backend: 0 for backend in self.backends}
        self.ok = {backend: 0 for backend in self.backends}
        self.err = {backend: 0 for backend in self.backends}
        self.lock = threading.Lock()

    def snapshot(self) -> dict:
        with self.lock:
            return {"backends": self.backends, "inflight": dict(self.inflight),
                    "ok": dict(self.ok), "err": dict(self.err)}

    def ordered_backends(self) -> list[str]:
        with self.lock:
            return sorted(self.backends, key=lambda b: (self.inflight[b], self.err[b], b))

    def acquire(self, backend: str) -> None:
        with self.lock:
            self.inflight[backend] += 1

    def release(self, backend: str, ok: bool) -> None:
        with self.lock:
            self.inflight[backend] = max(0, self.inflight[backend] - 1)
            (self.ok if ok else self.err)[backend] += 1


def make_handler(state: RouterState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OpenAIInflightRouter/0.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def _send_json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/health", "/v1/health"):
                self._send_json(200, {"status": "ok", **state.snapshot()})
            elif self.path in ("/stats", "/v1/stats"):
                self._send_json(200, state.snapshot())
            else:
                self._proxy_request("GET", b"")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            self._proxy_request("POST", self.rfile.read(length) if length else b"")

        def _proxy_request(self, method: str, body: bytes) -> None:
            last_error = None
            headers = {"Content-Type": self.headers.get("Content-Type", "application/json"),
                       "Authorization": self.headers.get("Authorization", "Bearer EMPTY")}
            for backend in state.ordered_backends():
                parsed = urlsplit(backend)
                prefix = parsed.path.rstrip("/")
                path = self.path
                if prefix and path == prefix:
                    path = "/"
                elif prefix and path.startswith(prefix + "/"):
                    path = path[len(prefix):]
                url = urlunsplit((parsed.scheme, parsed.netloc, prefix + path,
                                  parsed.query, parsed.fragment))
                state.acquire(backend)
                try:
                    start = time.time()
                    response = requests.request(method, url, data=body if method == "POST" else None,
                                                headers=headers, timeout=state.timeout)
                except requests.RequestException as exc:
                    state.release(backend, ok=False)
                    last_error = repr(exc)
                    continue
                state.release(backend, ok=200 <= response.status_code < 500)
                self.send_response(response.status_code)
                for key, value in response.headers.items():
                    if key.lower() not in {"content-encoding", "transfer-encoding", "connection", "content-length"}:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(response.content)))
                self.send_header("X-Router-Backend", backend)
                self.send_header("X-Router-Latency-S", f"{time.time() - start:.3f}")
                self.end_headers()
                self.wfile.write(response.content)
                return
            self._send_json(503, {"error": {"message": f"all backends failed: {last_error}",
                                           "type": "router_unavailable"}, **state.snapshot()})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backends", required=True, help="Comma-separated local base URLs ending in /v1")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    backends = [backend.strip() for backend in args.backends.split(",") if backend.strip()]
    if not backends or len(backends) != len(set(backends)):
        parser.error("Provide at least one unique backend")
    for backend in backends:
        url = urlsplit(backend)
        if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"} or url.path.rstrip("/") != "/v1":
            parser.error("This router only accepts local HTTP /v1 backends")
    state = RouterState(backends, args.timeout)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    print(f"router listening http://127.0.0.1:{args.port}/v1", flush=True)
    print("backends=" + ",".join(state.backends), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
