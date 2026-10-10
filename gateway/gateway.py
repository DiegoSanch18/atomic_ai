"""
LlamaLaunch Inference Gateway (:8081)
=====================================
Lightweight, non-blocking OpenAI-compatible HTTP Gateway that decouples
cognitive client proxies (like Atomic AI) from underlying physical compute backends
(local llama-server on RTX 5060 Ti, Vulkan on Laptop, or TrueNAS over LAN).

Features:
- Standard endpoints: POST /v1/chat/completions, GET /v1/models, GET /health
- Dynamic backend resolution from ProcessManager or LAN node registry
- Unbuffered SSE streaming pass-through
- RFC 7230 case-insensitive header forwarding
- Pre-flight health checking (503 Service Unavailable when backend is down)
- Zero extra heavy dependencies (ThreadingHTTPServer + requests)
"""

import sys
import os
import json
import logging
import threading
from typing import Optional, Dict, Any
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import requests

try:
    from .manager import ProcessManager
except ImportError:
    try:
        from app.backend.manager import ProcessManager
    except ImportError:
        ProcessManager = None

logger = logging.getLogger("LlamaLaunch.Gateway")

DEFAULT_GATEWAY_HOST = os.getenv("GATEWAY_HOST", "127.0.0.1")
DEFAULT_GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8081"))
DEFAULT_LOCAL_BACKEND = os.getenv("DEFAULT_LOCAL_BACKEND", "http://127.0.0.1:8080")

LAN_NODE_ROUTES = {
    "desktop": os.getenv("DESKTOP_BACKEND_URL", "http://127.0.0.1:8080"),
    "laptop": os.getenv("LAPTOP_BACKEND_URL", "http://127.0.0.1:8080"),
    "truenas": os.getenv("TRUENAS_BACKEND_URL", "http://truenas.local:8080"),
}


def resolve_backend_url(headers: Optional[Dict[str, str]] = None) -> str:
    """
    Dynamically determines the upstream backend URL.
    Checks headers for 'x-target-backend', then ProcessManager active state,
    and falls back to DEFAULT_LOCAL_BACKEND.
    """
    if headers:
        for k, v in headers.items():
            if k.lower() == "x-target-backend":
                target = v.strip().lower()
                if target in LAN_NODE_ROUTES:
                    return LAN_NODE_ROUTES[target]
                if v.startswith("http://") or v.startswith("https://"):
                    return v.rstrip("/")

    env_target = os.getenv("UPSTREAM_TARGET_BACKEND")
    if env_target:
        return env_target.rstrip("/")

    if ProcessManager:
        try:
            pm = ProcessManager()
            port = getattr(pm, "active_port", 8080) or 8080
            return f"http://127.0.0.1:{port}"
        except Exception:
            pass

    return DEFAULT_LOCAL_BACKEND


def is_backend_healthy(backend_url: str, timeout: float = 0.3) -> bool:
    """
    Quick non-blocking probe to verify that the upstream server is up.
    If ProcessManager is available and reports RUNNING for local port, bypasses probe.
    """
    if ProcessManager and "127.0.0.1" in backend_url:
        try:
            pm = ProcessManager()
            if getattr(pm, "status", "") == "RUNNING":
                return True
        except Exception:
            pass

    try:
        resp = requests.get(f"{backend_url.rstrip('/')}/health", timeout=timeout)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    try:
        resp = requests.get(f"{backend_url.rstrip('/')}/v1/models", timeout=timeout)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    return False


class GatewayRequestHandler(BaseHTTPRequestHandler):
    """Handles incoming OpenAI-compatible HTTP requests and proxies them."""
    
    server_version = "LlamaLaunchGateway/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s - - [%s] %s\n", self.address_string(), self.log_date_time_string(), format % args)

    def _get_headers_dict(self) -> Dict[str, str]:
        headers = {}
        for k, v in self.headers.items():
            headers[k] = v
        return headers

    def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        headers = self._get_headers_dict()
        backend_url = resolve_backend_url(headers)

        if path in ("/health", ""):
            healthy = is_backend_healthy(backend_url, timeout=0.3)
            status_code = 200 if healthy else 503
            self._send_json(status_code, {
                "status": "ok" if healthy else "degraded",
                "gateway": "online",
                "backend_url": backend_url,
                "backend_healthy": healthy,
                "active_model": getattr(ProcessManager(), "active_model", "") if ProcessManager else ""
            })
            return

        if path == "/v1/models":
            if is_backend_healthy(backend_url, timeout=0.3):
                try:
                    resp = requests.get(f"{backend_url}/v1/models", headers={"Accept": "application/json"}, timeout=1.0)
                    if resp.status_code == 200:
                        self._send_json(200, resp.json())
                        return
                except Exception:
                    pass

            # Fallback model list if backend is loading or registered
            active_model = getattr(ProcessManager(), "active_model", "") if ProcessManager else "gemma-4-e2b"
            self._send_json(200, {
                "object": "list",
                "data": [
                    {
                        "id": active_model or "gemma-4-e2b",
                        "object": "model",
                        "created": 1700000000,
                        "owned_by": "llamalaunch"
                    }
                ]
            })
            return

        self._send_json(404, {"error": {"message": f"Endpoint {self.path} not found on Gateway", "code": 404}})

    def do_POST(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        headers = self._get_headers_dict()
        backend_url = resolve_backend_url(headers)

        if path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": f"Endpoint {self.path} not found", "code": 404}})
            return

        # Read and validate body first
        content_length_header = None
        for k, v in headers.items():
            if k.lower() == "content-length":
                content_length_header = v
                break

        if not content_length_header:
            self._send_json(400, {"error": {"message": "Missing Content-Length header", "code": 400}})
            return

        try:
            content_length = int(content_length_header)
            if content_length <= 0:
                self._send_json(400, {"error": {"message": "Request body cannot be empty or negative", "code": 400}})
                return
            body_bytes = self.rfile.read(content_length)
        except Exception as e:
            self._send_json(400, {"error": {"message": f"Invalid request body or length: {e}", "code": 400}})
            return

        if not body_bytes:
            self._send_json(400, {"error": {"message": "Request body cannot be empty", "code": 400}})
            return

        # Parse request body to check stream flag and validate JSON
        is_stream = False
        try:
            body_json = json.loads(body_bytes.decode("utf-8"))
            is_stream = bool(body_json.get("stream", False))
        except Exception as e:
            self._send_json(400, {"error": {"message": f"Invalid JSON in request body: {e}", "code": 400}})
            return

        # Pre-flight health check
        if not is_backend_healthy(backend_url, timeout=0.3):
            self._send_json(503, {
                "error": {
                    "message": f"Local inference backend is not ready at {backend_url}. Start llama-server first.",
                    "type": "service_unavailable",
                    "code": 503
                }
            })
            return

        upstream_headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if is_stream else "application/json",
        }
        for k, v in headers.items():
            if k.lower() in ("authorization", "user-agent"):
                upstream_headers[k] = v

        try:
            resp = requests.post(
                f"{backend_url}/v1/chat/completions",
                data=body_bytes,
                headers=upstream_headers,
                stream=is_stream,
                timeout=(5.0, 300.0)
            )
        except requests.exceptions.RequestException as e:
            self._send_json(502, {"error": {"message": f"Failed to connect to backend {backend_url}: {e}", "code": 502}})
            return

        if not is_stream:
            self.send_response(resp.status_code)
            for k, v in resp.headers.items():
                if k.lower() in ("content-type", "content-length"):
                    self.send_header(k, v)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(resp.content)
            return

        # Streaming pass-through
        self.send_response(resp.status_code)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class InferenceGateway:
    """Manages the background ThreadingHTTPServer instance."""

    def __init__(self, host: str = DEFAULT_GATEWAY_HOST, port: int = DEFAULT_GATEWAY_PORT):
        self.host = host
        self.port = port
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self._is_running = False

    def start(self) -> None:
        if self._is_running:
            return

        self.server = ThreadingHTTPServer((self.host, self.port), GatewayRequestHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="LlamaLaunch-Gateway")
        self.thread.start()
        self._is_running = True
        logger.info(f"LlamaLaunch Gateway listening on http://{self.host}:{self.port}")

    def stop(self) -> None:
        if not self._is_running or not self.server:
            return

        self.server.shutdown()
        self.server.server_close()
        self._is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("LlamaLaunch Gateway stopped.")

    @property
    def is_running(self) -> bool:
        return self._is_running


_global_gateway: Optional[InferenceGateway] = None

def get_gateway() -> InferenceGateway:
    global _global_gateway
    if _global_gateway is None:
        host = os.getenv("GATEWAY_HOST", DEFAULT_GATEWAY_HOST)
        port = int(os.getenv("GATEWAY_PORT", DEFAULT_GATEWAY_PORT))
        _global_gateway = InferenceGateway(host=host, port=port)
    return _global_gateway


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    host = os.getenv("GATEWAY_HOST", DEFAULT_GATEWAY_HOST)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv("GATEWAY_PORT", DEFAULT_GATEWAY_PORT))
    gw = InferenceGateway(host=host, port=port)
    gw.start()
    print(f"Gateway running at http://{host}:{port}. Press Ctrl+C to stop.")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        gw.stop()
        print("Gateway stopped.")
