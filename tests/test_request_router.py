"""CPU-only HTTP checks for replica dispatch and unchanged request bodies."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("request_router", ROOT / "codes/scripts/openai_inflight_router.py")
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)


class RequestRouterTest(unittest.TestCase):
    def start_server(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop():
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        self.addCleanup(stop)
        return f"http://127.0.0.1:{server.server_port}/v1"

    def backend(self, captured, entered=None, release=None):
        class Backend(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                captured.append((self.path, body))
                if entered is not None:
                    entered.set()
                if release is not None:
                    release.wait(timeout=5)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        return Backend

    def request(self, url, body=b'{"temperature":0,"messages":[]}'):
        request = Request(url + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=8) as response:
            return response.read(), response.headers["X-Router-Backend"]

    def test_ordering_matches_historical_policy(self):
        state = router.RouterState(["http://127.0.0.1:9002/v1", "http://127.0.0.1:9001/v1"], 1)
        first, second = sorted(state.backends)
        self.assertEqual(state.ordered_backends(), [first, second])
        state.acquire(first)
        self.assertEqual(state.ordered_backends(), [second, first])
        state.release(first, ok=False)
        self.assertEqual(state.ordered_backends(), [second, first])
        self.assertEqual(state.snapshot()["err"][first], 1)

    def test_body_and_api_prefix_are_unchanged(self):
        captured = []
        backend = self.start_server(self.backend(captured))
        state = router.RouterState([backend], 2)
        url = self.start_server(router.make_handler(state))
        body = b'{ "messages": [{"type":"image_url","data":"opaque-bytes"}], "temperature": 0 }'
        self.assertEqual(self.request(url, body), (body, backend))
        self.assertEqual(captured, [("/v1/chat/completions", body)])
        with urlopen(url + "/stats", timeout=2) as response:
            stats = json.load(response)
        self.assertEqual(stats["ok"], {backend: 1})
        self.assertEqual(stats["inflight"], {backend: 0})

    def test_concurrent_request_avoids_busy_backend(self):
        captured, entered, release = [], threading.Event(), threading.Event()
        backends = [self.start_server(self.backend(captured, entered, release)),
                    self.start_server(self.backend(captured))]
        state = router.RouterState(backends, 5)
        # Error history is the historical tie-breaker after current load.
        state.err[backends[1]] = 1
        url = self.start_server(router.make_handler(state))
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.request, url)
            try:
                self.assertTrue(entered.wait(timeout=3))
                self.assertEqual(self.request(url)[1], backends[1])
            finally:
                release.set()
            self.assertEqual(first.result(timeout=5)[1], backends[0])
        self.assertEqual(state.snapshot()["ok"], {backend: 1 for backend in backends})
        self.assertEqual(state.snapshot()["inflight"], {backend: 0 for backend in backends})


if __name__ == "__main__":
    unittest.main()
