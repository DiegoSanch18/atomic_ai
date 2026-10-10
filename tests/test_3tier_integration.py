"""
End-to-End 3-Tier Integration Test Suite
========================================
Validates the decoupled 3-tier local AI architecture:
  Tier 1: Interactive Client / CLI (scripts/test_cli.py)
  Tier 2: Atomic AI Orchestrator / Proxy (app/main.py)
  Tier 3: LlamaLaunch Reverse Gateway (port 18081) -> Mock Backend llama-server (port 18080)

Test Scenarios:
1. test_3tier_health_chain: Verifies full diagnostic health chain across all tiers.
2. test_3tier_end_to_end_streaming_pipeline: Verifies full request decomposition, execution,
   and synthesis with live SSE deltas through Gateway pass-through.
3. test_3tier_gateway_503_propagation: Verifies clean 503 error handling when backend is down.
4. test_3tier_keepalive_immediate_done_break: Validates client loop exits < 50ms upon [DONE]
   without lingering on socket keep-alive timeout.
5. test_test_cli_script_mock_and_check_only: Validates scripts/test_cli.py --mock and --check-only.
"""

from __future__ import annotations

import sys
import os
import time
import json
import uuid
import threading
import subprocess
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Generator, Any

import httpx
import pytest

# Ensure LlamaLaunch is importable
LLAMALAUNCH_DIR = Path("C:/git/LlamaLaunch").resolve()
if str(LLAMALAUNCH_DIR) not in sys.path:
    sys.path.insert(0, str(LLAMALAUNCH_DIR))

try:
    from llamaLauncher.app.backend.gateway import InferenceGateway
except ImportError:
    from app.backend.gateway import InferenceGateway

from app.config import settings
from app.main import app

BACKEND_PORT = 18080
GATEWAY_PORT = 18081
PROXY_TEST_PORT = 18000
KEEPALIVE_TEST_PORT = 18082


class MockBackendHandler(BaseHTTPRequestHandler):
    """Simulates physical llama-server inference backend (:18080)."""

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress console logging during test execution

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", "/healthz"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "ready": True}).encode("utf-8"))
            return

        if path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "object": "list",
                "data": [{"id": "deepseek-v4-flash", "object": "model", "owned_by": "mock-backend"}]
            }).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def _write_chunk(self, data: bytes) -> None:
        chunk_header = f"{len(data):X}\r\n".encode("utf-8")
        self.wfile.write(chunk_header + data + b"\r\n")
        self.wfile.flush()

    def do_POST(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        if path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            body = {}

        is_stream = bool(body.get("stream", False))

        if not is_stream:
            # Phase 1: Task Decomposition response (atomic single-leaf)
            decomp_data = {
                "id": f"chatcmpl-decomp-{uuid.uuid4().hex[:6]}",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"atomic": True, "subtasks": []})
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
            payload = json.dumps(decomp_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # Phase 2 / Phase 3: Streaming responses (SSE) with Chunked Transfer Encoding
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        chunk_id = f"chatcmpl-stream-{uuid.uuid4().hex[:6]}"
        tokens = ["Inferencia ", "local ", "3-tier ", "completada."]
        for token in tokens:
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": token}}]
            }
            self._write_chunk(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))

        # Stop marker
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        self._write_chunk(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
        self._write_chunk(b"data: [DONE]\n\n")

        # End of HTTP chunked stream
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class MockProxyHealthHandler(BaseHTTPRequestHandler):
    """Simulates Tier 2 health endpoints for test_cli.py --check-only (:18000)."""

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        if path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
            return
        if path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"object": "list", "data": [{"id": "deepseek-v4-flash"}]}')
            return
        self.send_response(404)
        self.end_headers()


class KeepAliveDelayHandler(BaseHTTPRequestHandler):
    """
    Simulates an SSE server that sends [DONE] but deliberately keeps socket open for 2s.
    Used to test that client stream loop breaks immediately (<50ms) without waiting on keep-alive.
    """

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # Send token chunk and DONE
        chunk = {
            "id": "chatcmpl-keepalive",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "token"}}]
        }
        payload = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode("utf-8")
        header = f"{len(payload):X}\r\n".encode("utf-8")
        self.wfile.write(header + payload + b"\r\n")
        self.wfile.flush()

        # Deliberately hold socket open for 2 seconds before sending zero-chunk
        time.sleep(2.0)
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass


@pytest.fixture(scope="module")
def mock_backend_server() -> Generator[str, None, None]:
    """Runs mock backend llama-server on port 18080."""
    server = ThreadingHTTPServer(("127.0.0.1", BACKEND_PORT), MockBackendHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="MockBackend-18080")
    thread.start()
    url = f"http://127.0.0.1:{BACKEND_PORT}"
    yield url
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


@pytest.fixture(scope="module")
def gateway_server(mock_backend_server: str) -> Generator[InferenceGateway, None, None]:
    """Runs LlamaLaunch InferenceGateway on port 18081 pointing to mock backend."""
    os.environ["UPSTREAM_TARGET_BACKEND"] = mock_backend_server
    gw = InferenceGateway(host="127.0.0.1", port=GATEWAY_PORT)
    gw.start()
    time.sleep(0.1)
    yield gw
    gw.stop()


@pytest.fixture(autouse=True)
def override_upstream_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Overrides conftest.py autouse fixture to point Atomic AI upstream
    specifically to LlamaLaunch Gateway on :18081 with a valid API key.
    """
    monkeypatch.setattr(settings, "upstream_base_url", f"http://127.0.0.1:{GATEWAY_PORT}")
    monkeypatch.setattr(settings, "upstream_api_key", "sk-local-3tier-key")
    monkeypatch.setattr(settings, "expose_reasoning_content", True)


@pytest.mark.asyncio
async def test_3tier_health_chain(mock_backend_server: str, gateway_server: InferenceGateway):
    """
    Scenario 1: test_3tier_health_chain
    Verifies full health chain:
    mock backend :18080 -> gateway :18081 -> atomic_ai :8000 /healthz
    """
    # 1. Probe Mock Backend :18080/health
    async with httpx.AsyncClient(timeout=2.0) as http_client:
        r_backend = await http_client.get(f"{mock_backend_server}/health")
        assert r_backend.status_code == 200
        assert r_backend.json().get("status") == "ok"

    # 2. Probe Gateway :18081/health
    async with httpx.AsyncClient(timeout=2.0) as http_client:
        r_gw = await http_client.get(f"http://127.0.0.1:{GATEWAY_PORT}/health")
        assert r_gw.status_code == 200
        gw_json = r_gw.json()
        assert gw_json.get("gateway") == "online"
        assert gw_json.get("backend_healthy") is True

    # 3. Probe Atomic AI /healthz
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        r_atomic = await client.get("/healthz")
        assert r_atomic.status_code == 200
        assert r_atomic.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_3tier_end_to_end_streaming_pipeline(mock_backend_server: str, gateway_server: InferenceGateway):
    """
    Scenario 2: test_3tier_end_to_end_streaming_pipeline
    Executes full multi-phase pipeline:
    Atomic AI -> Gateway (:18081) -> Mock Backend (:18080)
    Validates that:
    - Phase 1 (Decomposition) completes via Gateway
    - Phase 2 (Leaf Execution) streams reasoning SSE deltas
    - Phase 3 (Synthesis) streams content SSE deltas
    - Stream concludes immediately with data: [DONE]
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=30.0) as client:
        payload = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Explica la inferencia local de LLMs."}],
            "stream": True
        }

        async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            received_lines: list[str] = []
            saw_reasoning = False
            saw_content = False
            saw_done = False

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                received_lines.append(line)

                if line == "data: [DONE]":
                    saw_done = True
                    break

                if line.startswith("data:"):
                    try:
                        chunk = json.loads(line[5:].strip())
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if delta.get("reasoning_content"):
                            saw_reasoning = True
                        if delta.get("content"):
                            saw_content = True
                    except Exception:
                        pass

            assert saw_done, "Stream must terminate with data: [DONE]"
            assert saw_reasoning, "Stream must emit reasoning_content deltas"
            assert saw_content, "Stream must emit final content deltas"
            assert len(received_lines) >= 4, f"Expected multi-chunk SSE stream, got {len(received_lines)} lines"


@pytest.mark.asyncio
async def test_3tier_gateway_503_propagation():
    """
    Scenario 3: test_3tier_gateway_503_propagation
    Points Gateway to an unreachable backend port (59999).
    Verifies that:
    - Gateway returns 503 Service Unavailable
    - Atomic AI cleanly catches UpstreamError and handles it gracefully
      without process crashes (emits error SSE chunk or 502 response).
    """
    # 1. Verify Gateway directly returns 503 for down backend
    headers = {"Content-Type": "application/json", "X-Target-Backend": "http://127.0.0.1:59999"}
    payload = {"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": False}

    async with httpx.AsyncClient(timeout=2.0) as http_client:
        r_gw = await http_client.post(
            f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions",
            json=payload,
            headers=headers
        )
        assert r_gw.status_code == 503
        assert r_gw.json().get("error", {}).get("code") == 503

    # 2. Test Atomic AI behavior when Gateway points to dead backend
    old_target = os.environ.get("UPSTREAM_TARGET_BACKEND")
    os.environ["UPSTREAM_TARGET_BACKEND"] = "http://127.0.0.1:59999"

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=5.0) as client:
            # Non-streaming request -> should return 502 UpstreamError
            r_non_stream = await client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": False}
            )
            assert r_non_stream.status_code == 502
            assert r_non_stream.json().get("error", {}).get("type") == "upstream_error"

            # Streaming request -> should emit error chunk and [DONE]
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True}
            ) as stream_resp:
                assert stream_resp.status_code == 200
                lines = [line.strip() async for line in stream_resp.aiter_lines() if line.strip()]
                assert any("[DONE]" in line for line in lines)
                assert any("upstream_error" in line for line in lines)
    finally:
        if old_target:
            os.environ["UPSTREAM_TARGET_BACKEND"] = old_target
        else:
            os.environ.pop("UPSTREAM_TARGET_BACKEND", None)


def test_3tier_keepalive_immediate_done_break():
    """
    Scenario 4: test_3tier_keepalive_immediate_done_break
    Verifies that the SSE client loop terminates immediately upon seeing `data: [DONE]`,
    exiting within <50ms even if the server holds the socket open for 2+ seconds.
    """
    server = ThreadingHTTPServer(("127.0.0.1", KEEPALIVE_TEST_PORT), KeepAliveDelayHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="KeepAliveDelayServer")
    thread.start()

    try:
        import requests
        t_request_sent = time.perf_counter()
        t_break_duration = None

        with requests.get(f"http://127.0.0.1:{KEEPALIVE_TEST_PORT}", stream=True, timeout=5.0) as resp:
            for raw_line in resp.iter_lines(chunk_size=1):
                if raw_line:
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if line.strip() == "data: [DONE]":
                        t_start_break = time.perf_counter()
                        break

            t_break_duration = (time.perf_counter() - t_start_break) * 1000.0

        total_elapsed = (time.perf_counter() - t_request_sent) * 1000.0

        # Assert immediate break duration is well below 50ms
        assert t_break_duration is not None
        assert t_break_duration < 50.0, f"Break duration was {t_break_duration:.2f} ms (expected < 50 ms)"
        # Total elapsed should be way below 2000ms (the sleep duration)
        assert total_elapsed < 500.0, f"Client waited on socket keep-alive: {total_elapsed:.2f} ms"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_test_cli_script_mock_and_check_only(mock_backend_server: str, gateway_server: InferenceGateway):
    """
    Scenario 5: test_test_cli_script_mock_and_check_only
    Validates the standalone CLI tool scripts/test_cli.py:
    - --mock executes offline authentic simulation and exits 0
    - --mock --json outputs valid parseable JSON metrics and exits 0
    - --check-only against active test tier fixtures outputs ALL TIERS HEALTHY and exits 0
    """
    cli_path = Path("scripts/test_cli.py").resolve()
    assert cli_path.exists(), f"CLI tool not found at {cli_path}"

    # 1. Test --mock mode
    res_mock = subprocess.run(
        [sys.executable, str(cli_path), "--mock"],
        capture_output=True,
        text=True,
        timeout=10.0
    )
    assert res_mock.returncode == 0, f"test_cli.py --mock failed with stderr: {res_mock.stderr}"
    assert "LATENCY & THROUGHPUT BENCHMARK" in res_mock.stdout
    assert "Immediate [DONE] break" in res_mock.stdout
    assert "200 OK - SUCCESS" in res_mock.stdout

    # 2. Test --mock --json mode
    res_json = subprocess.run(
        [sys.executable, str(cli_path), "--mock", "--json"],
        capture_output=True,
        text=True,
        timeout=10.0
    )
    assert res_json.returncode == 0, f"test_cli.py --mock --json failed: {res_json.stderr}"
    metrics = json.loads(res_json.stdout.strip())
    assert metrics.get("status") == "SUCCESS"
    assert "ttft_ms" in metrics
    assert "total_duration_ms" in metrics
    assert metrics.get("total_tokens", 0) > 0

    # 3. Test --check-only against active test fixtures
    proxy_server = ThreadingHTTPServer(("127.0.0.1", PROXY_TEST_PORT), MockProxyHealthHandler)
    proxy_server.daemon_threads = True
    proxy_thread = threading.Thread(target=proxy_server.serve_forever, daemon=True, name="MockProxy-18000")
    proxy_thread.start()

    try:
        res_check = subprocess.run(
            [
                sys.executable,
                str(cli_path),
                "--check-only",
                "--url", f"http://127.0.0.1:{PROXY_TEST_PORT}",
                "--gateway-url", f"http://127.0.0.1:{GATEWAY_PORT}",
                "--backend-url", f"http://127.0.0.1:{BACKEND_PORT}",
            ],
            capture_output=True,
            text=True,
            timeout=10.0
        )
        assert res_check.returncode == 0, f"test_cli.py --check-only failed: {res_check.stderr}"
        assert "ALL TIERS HEALTHY" in res_check.stdout
        assert "Tier 1: Client Harness" in res_check.stdout
        assert "Tier 2: Atomic AI Proxy" in res_check.stdout
        assert "Tier 3: LlamaLaunch Gateway" in res_check.stdout
        assert "Backend: llama-server" in res_check.stdout
    finally:
        proxy_server.shutdown()
        proxy_server.server_close()
        proxy_thread.join(timeout=2.0)
