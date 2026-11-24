import json
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

from lean_translator.refiner import distill

ITEMS = [
    {"id": "d0", "model": "m", "name": "a", "kind": "theorem", "file": "A.lean", "lean": "...", "gloss": "...",
     "masked": "Theorem a: Then [T0] equals [T1].", "slots": ["n * 2", "n + n"], "name_gloss": "a"},
    {"id": "d1", "model": "m", "name": "b", "kind": "theorem", "file": "A.lean", "lean": "...", "gloss": "...",
     "masked": "Theorem b: Then [T0] equals [T1].", "slots": ["x", "y + 1"], "name_gloss": "b"},
    {"id": "d2", "model": "m", "name": "c", "kind": "theorem", "file": "B.lean", "lean": "...", "gloss": "...",
     "masked": "Theorem c: Then [T0] is positive.", "slots": ["n + 1"], "name_gloss": "c"},
]
REPLIES = {
    "Theorem a": "<english>For all n, [T0] = [T1].</english>",
    "Theorem b": "<english>[T0] equals something.</english>",  # dropped [T1] -> rejected
    "Theorem c": "<think>maybe <english>wrong</english>?</think><english>[T0] is positive.</english>",
}


class FakeOllama(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["stream"] is False and body["messages"][0]["role"] == "system"
        user = body["messages"][1]["content"]
        reply = next(v for k, v in REPLIES.items() if k in user)
        FakeOllama.calls += 1
        data = json.dumps({"message": {"role": "assistant", "content": reply}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_ollama_generates_validates_and_resumes(tmp_path):
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in ITEMS))
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    args = Namespace(dir=str(tmp_path), ollama_model="fake:1b",
                     host=f"http://127.0.0.1:{server.server_port}", workers=2)
    try:
        distill.cmd_local(args)
        pairs = [json.loads(line) for line in (tmp_path / "distilled.jsonl").read_text().splitlines()]
        assert {p["name"]: p["tgt"] for p in pairs} == {
            "a": "For all n, [T0] = [T1].",
            "c": "[T0] is positive.",  # the <english> inside <think> was ignored
        }
        assert FakeOllama.calls == 3
        distill.cmd_local(args)  # second run resumes: nothing left to ask
        assert FakeOllama.calls == 3
    finally:
        server.shutdown()
