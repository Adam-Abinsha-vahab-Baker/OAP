"""
A minimal fake agent server for integration testing.
Run with: python tests/fake_agent_server.py
It listens on localhost:9999 and responds to POST /invoke
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class FakeAgentHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/invoke":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))

        goal = body.get("goal", "")
        response = {
            "result": f"Fake agent processed: {goal}",
            "memory": {
                "processed_by": "fake-agent",
                "goal_length": len(goal),
            },
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass  # silence request logs during tests


def run(port: int = 9999):
    server = HTTPServer(("localhost", port), FakeAgentHandler)
    print(f"Fake agent running on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()