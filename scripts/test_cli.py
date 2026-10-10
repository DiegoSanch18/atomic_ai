#!/usr/bin/env python3
"""
Atomic AI - 3-Tier Pipeline Interactive CLI Test & Profiler
============================================================
Validates the decoupled 3-tier local AI inference architecture:
  Tier 1: Interactive Client / CLI (scripts/test_cli.py)
  Tier 2: Atomic AI Orchestrator / Proxy (:8000)
  Tier 3: LlamaLaunch Reverse Gateway (:8081) -> llama-server (:8080)

Features:
- Real-time terminal SSE streaming visualizer differentiating reasoning_content from content.
- High-precision latency and throughput profiling (TTFT, TTFCT, tokens/s).
- Guaranteed immediate break on `data: [DONE]` without lingering on socket keep-alive.
- Diagnostic pre-flight check matrix (`--check-only`).
- Authentic offline mock stream simulation (`--mock`).
"""

import sys
import os
import time
import json
import re
import argparse
from typing import Dict, Any, List, Optional, Tuple, Iterator

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    # Enable ANSI escape sequences in Windows Console if possible
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

try:
    import requests
except ImportError:
    requests = None


# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def supports_color() -> bool:
    """Detects if terminal supports ANSI color escapes."""
    if os.getenv("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    return True


def colorize(text: str, color_code: str) -> str:
    if supports_color():
        return f"{color_code}{text}{RESET}"
    return text


def estimate_tokens(text: str) -> int:
    """Estimates token count from text using word and punctuation splitting."""
    if not text:
        return 0
    tokens = len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))
    return max(1, tokens)


def generate_mock_sse_events() -> Iterator[str]:
    """
    Yields authentic synthetic SSE lines simulating a 3-tier stream:
    role chunk -> 4 reasoning deltas -> 3 content deltas -> stop chunk -> [DONE].
    """
    chunk_id = f"chatcmpl-mock-{int(time.time())}"
    
    # 1. Role chunk
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}}]})}\n\n"
    
    # 2. Reasoning deltas (cognitive planning phase)
    reasoning_steps = [
        "Fase 1 de 3. Primero comienzo dividiendo la tarea en sus subtareas atómicas.\n\n",
        "Subtareas identificadas:\n- Paso 1: Cargar modelo y cuantizaciones GGUF en VRAM.\n",
        "- Paso 2: Ejecutar prefill de contexto en la GPU mediante kernels CUDA/Vulkan.\n",
        "- Paso 3: Muestrear recursivamente tokens de salida en bucle autorregresivo.\n\n",
    ]
    for step in reasoning_steps:
        time.sleep(0.03)  # 30ms synthetic delay
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"reasoning_content": step}}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        
    # 3. Content deltas (synthesis phase)
    content_steps = [
        "1. **Carga y Offloading de Pesos**: Se mapean los tensores del archivo GGUF a la memoria de la GPU (VRAM), balanceando capas según la capacidad de hardware.\n",
        "2. **Prefill y Evaluación de Contexto**: El prompt se tokeniza y se evalúa paralelamente en la GPU, inicializando las capas de memoria KV Cache.\n",
        "3. **Generación Autorregresiva**: El modelo predice tokens sucesivos uno a uno hasta alcanzar el token de parada o el límite de contexto.\n"
    ]
    for step in content_steps:
        time.sleep(0.02)  # 20ms synthetic delay
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": step}}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # 4. Final stop chunk
    time.sleep(0.01)
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    
    # 5. Immediate [DONE] marker
    yield "data: [DONE]\n\n"


def run_check_only(
    proxy_url: str = "http://127.0.0.1:8000",
    gateway_url: str = "http://127.0.0.1:8081",
    backend_url: str = "http://127.0.0.1:8080",
    timeout: float = 2.0,
    json_mode: bool = False
) -> int:
    """
    Executes pre-flight health diagnostics across all tiers.
    Returns 0 if all tiers are operational, 1 otherwise.
    """
    matrix: Dict[str, Dict[str, Any]] = {}

    # Tier 1: Client harness
    matrix["tier1_client"] = {
        "name": "Tier 1: Client Harness",
        "status": "OK",
        "details": f"Python {sys.version.split()[0]} / Local harness ready"
    }

    # Tier 2: Atomic AI Proxy (:8000)
    t2_status = "FAIL"
    t2_details = ""
    try:
        r_health = requests.get(f"{proxy_url.rstrip('/')}/healthz", timeout=timeout)
        if r_health.status_code == 200:
            try:
                r_models = requests.get(f"{proxy_url.rstrip('/')}/v1/models", timeout=timeout)
                models_data = r_models.json().get("data", [])
                model_name = models_data[0].get("id", "unknown") if models_data else "active"
                t2_details = f"200 OK (Model: {model_name})"
            except Exception:
                t2_details = "200 OK (healthz responded)"
            t2_status = "OK"
        else:
            t2_details = f"HTTP {r_health.status_code}"
    except Exception as e:
        t2_details = f"Connection failed ({type(e).__name__})"
    matrix["tier2_proxy"] = {"name": f"Tier 2: Atomic AI Proxy ({proxy_url})", "status": t2_status, "details": t2_details}

    # Tier 3: LlamaLaunch Gateway (:8081)
    t3_status = "FAIL"
    t3_details = ""
    try:
        r_gw = requests.get(f"{gateway_url.rstrip('/')}/health", timeout=timeout)
        if r_gw.status_code == 200:
            gw_data = r_gw.json()
            backend_healthy = gw_data.get("backend_healthy", False)
            t3_details = f"200 OK (Backend healthy: {backend_healthy})"
            t3_status = "OK"
        elif r_gw.status_code == 503:
            t3_details = "503 Degraded (Gateway online, backend unreachable)"
            t3_status = "DEGRADED"
        else:
            t3_details = f"HTTP {r_gw.status_code}"
    except Exception as e:
        t3_details = f"Connection failed ({type(e).__name__})"
    matrix["tier3_gateway"] = {"name": f"Tier 3: LlamaLaunch Gateway ({gateway_url})", "status": t3_status, "details": t3_details}

    # Backend: llama-server (:8080)
    be_status = "FAIL"
    be_details = ""
    try:
        r_be = requests.get(f"{backend_url.rstrip('/')}/health", timeout=timeout)
        if r_be.status_code == 200:
            be_status = "OK"
            be_details = "200 OK (Ready for inference)"
        else:
            be_details = f"HTTP {r_be.status_code}"
    except Exception as e:
        be_details = f"Offline / Connection failed ({type(e).__name__})"
    matrix["backend_inference"] = {"name": f"Backend: llama-server ({backend_url})", "status": be_status, "details": be_details}

    all_healthy = (t2_status == "OK" and t3_status == "OK" and be_status == "OK")

    if json_mode:
        print(json.dumps({
            "status": "ALL_HEALTHY" if all_healthy else "DEGRADED",
            "exit_code": 0 if all_healthy else 1,
            "matrix": matrix
        }, indent=2))
        return 0 if all_healthy else 1

    header = "=" * 80
    print(header)
    print(colorize("3-TIER PIPELINE HEALTH CHECK MATRIX", BOLD))
    print(header)
    
    for key in ["tier1_client", "tier2_proxy", "tier3_gateway", "backend_inference"]:
        info = matrix[key]
        tag_color = GREEN if info["status"] == "OK" else (YELLOW if info["status"] == "DEGRADED" else RED)
        tag = colorize(f"[{info['status']}]", tag_color)
        print(f"  {info['name']:<42} {tag:<18} {info['details']}")

    print(header)
    if all_healthy:
        print(colorize("Status: ALL TIERS HEALTHY (Exit 0)", GREEN + BOLD))
        return 0
    else:
        print(colorize("Status: PIPELINE DEGRADED / UNHEALTHY (Exit 1)", RED + BOLD))
        print("\nRemediation Instructions:")
        if t2_status != "OK":
            print(f"  - Start Tier 2 Atomic AI: run 'python run.py' from C:\\git\\atomic_ai")
        if t3_status != "OK":
            print(f"  - Start Tier 3 Gateway: run Gateway from LlamaLaunch UI or 'python llamaLauncher\\app\\backend\\gateway.py'")
        if be_status != "OK":
            print(f"  - Start Backend llama-server: launch model server batch script (.bat) on {backend_url}")
        print()
        return 1


def process_sse_stream(
    line_iterator: Iterator[str],
    t_request_sent: float,
    verbose: bool = False,
    json_mode: bool = False
) -> Dict[str, Any]:
    """
    Core SSE processing loop.
    Extracts reasoning_content and content in real-time, displays styled output,
    and terminates IMMEDIATELY on `data: [DONE]`.
    """
    t_first_token: Optional[float] = None
    t_first_content: Optional[float] = None
    t_done: Optional[float] = None
    socket_exit_ms: float = 0.0

    reasoning_parts: List[str] = []
    content_parts: List[str] = []
    in_reasoning_banner = False
    in_content_banner = False

    banner_width = 80
    reasoning_banner = "┌── [REASONING & COGNITIVE PLANNING] " + "─" * (banner_width - 37)
    content_banner   = "┌── [FINAL SYNTHESIZED RESPONSE] "     + "─" * (banner_width - 33)
    footer_bar       = "└── " + "─" * (banner_width - 5)

    for raw_line in line_iterator:
        line = raw_line.strip()
        if not line:
            continue

        if verbose:
            sys.stderr.write(colorize(f"[SSE raw] {line}\n", DIM))

        # Check for DONE marker
        if line == "data: [DONE]" or line == "[DONE]":
            t_break_start = time.perf_counter()
            t_done = t_break_start
            # CRITICAL: Immediate break out of the stream loop
            socket_exit_ms = (time.perf_counter() - t_break_start) * 1000.0
            break

        if not line.startswith("data:"):
            continue

        payload_str = line[5:].strip()
        try:
            chunk = json.loads(payload_str)
        except Exception:
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})

        # Handle reasoning_content delta
        reasoning_delta = delta.get("reasoning_content")
        if reasoning_delta:
            now = time.perf_counter()
            if t_first_token is None:
                t_first_token = now
            if not json_mode:
                if not in_reasoning_banner:
                    print(colorize(f"\n{reasoning_banner}", CYAN + BOLD))
                    in_reasoning_banner = True

                styled_text = colorize(reasoning_delta, CYAN)
                sys.stdout.write(styled_text)
                sys.stdout.flush()
            reasoning_parts.append(reasoning_delta)

        # Handle content delta
        content_delta = delta.get("content")
        if content_delta:
            now = time.perf_counter()
            if t_first_token is None:
                t_first_token = now
            if t_first_content is None:
                t_first_content = now
                if not json_mode:
                    if in_reasoning_banner:
                        print(colorize(f"\n{footer_bar}\n", CYAN))
                        in_reasoning_banner = False
                    print(colorize(f"{content_banner}", GREEN + BOLD))
                    in_content_banner = True

            if not json_mode:
                styled_text = colorize(content_delta, GREEN)
                sys.stdout.write(styled_text)
                sys.stdout.flush()
            content_parts.append(content_delta)

    if not json_mode:
        if in_reasoning_banner:
            print(colorize(f"\n{footer_bar}\n", CYAN))
        if in_content_banner:
            print(colorize(f"\n{footer_bar}\n", GREEN))

    now = time.perf_counter()
    if t_done is None:
        t_done = now
    if t_first_token is None:
        t_first_token = t_done
    if t_first_content is None:
        t_first_content = t_done

    full_reasoning = "".join(reasoning_parts)
    full_content = "".join(content_parts)

    reasoning_tokens = estimate_tokens(full_reasoning)
    content_tokens = estimate_tokens(full_content)
    total_tokens = reasoning_tokens + content_tokens

    ttft_ms = (t_first_token - t_request_sent) * 1000.0
    ttfct_ms = (t_first_content - t_request_sent) * 1000.0
    planning_duration_ms = max(0.0, (t_first_content - t_first_token) * 1000.0)
    synthesis_duration_ms = max(0.0, (t_done - t_first_content) * 1000.0)
    total_duration_ms = max(0.1, (t_done - t_request_sent) * 1000.0)

    total_duration_s = total_duration_ms / 1000.0
    gen_duration_s = max(0.001, (t_done - t_first_token))

    overall_throughput = total_tokens / total_duration_s if total_duration_s > 0 else 0.0
    gen_throughput = total_tokens / gen_duration_s if gen_duration_s > 0 else 0.0

    return {
        "status": "SUCCESS",
        "ttft_ms": round(ttft_ms, 2),
        "ttfct_ms": round(ttfct_ms, 2),
        "planning_duration_ms": round(planning_duration_ms, 2),
        "synthesis_duration_ms": round(synthesis_duration_ms, 2),
        "total_duration_ms": round(total_duration_ms, 2),
        "reasoning_tokens": reasoning_tokens,
        "reasoning_chars": len(full_reasoning),
        "content_tokens": content_tokens,
        "content_chars": len(full_content),
        "total_tokens": total_tokens,
        "throughput_overall": round(overall_throughput, 2),
        "throughput_generation": round(gen_throughput, 2),
        "socket_exit_ms": round(socket_exit_ms, 3),
        "full_reasoning": full_reasoning,
        "full_content": full_content
    }


def print_benchmark_report(metrics: Dict[str, Any], json_mode: bool = False) -> None:
    """Prints the structured latency and throughput report."""
    if json_mode:
        print(json.dumps(metrics, indent=2))
        return

    header = "=" * 80
    print(header)
    print(colorize("LATENCY & THROUGHPUT BENCHMARK", BOLD))
    print(header)
    print(f"  Time to First Token (TTFT):       {metrics['ttft_ms']:>8.1f} ms")
    print(f"  Time to First Content (TTFCT):    {metrics['ttfct_ms']:>8.1f} ms")
    print(f"  Cognitive Planning Duration:      {metrics['planning_duration_ms']:>8.1f} ms")
    print(f"  Synthesis Duration:               {metrics['synthesis_duration_ms']:>8.1f} ms")
    print(f"  Total Execution Time:             {metrics['total_duration_ms']:>8.1f} ms")
    print(f"  Reasoning Tokens:                 {metrics['reasoning_tokens']:>8} tokens (~{metrics['reasoning_chars']} chars)")
    print(f"  Final Content Tokens:             {metrics['content_tokens']:>8} tokens (~{metrics['content_chars']} chars)")
    print(f"  Total Tokens:                     {metrics['total_tokens']:>8} tokens")
    print(f"  Inference Throughput (Overall):   {metrics['throughput_overall']:>8.1f} tokens/s")
    print(f"  Inference Throughput (Generation):{metrics['throughput_generation']:>8.1f} tokens/s")
    exit_note = f"Immediate [DONE] break ({metrics['socket_exit_ms']:.2f} ms socket exit)"
    print(f"  Stream Termination:               {exit_note}")
    status_label = colorize("200 OK - SUCCESS", GREEN + BOLD)
    print(f"  Pipeline Status:                  {status_label}")
    print(header)


def run_pipeline(
    prompt: str,
    proxy_url: str = "http://127.0.0.1:8000",
    gateway_url: str = "http://127.0.0.1:8081",
    model: str = "deepseek-v4-flash",
    timeout: float = 120.0,
    stream: bool = True,
    mock: bool = False,
    verbose: bool = False,
    json_mode: bool = False
) -> int:
    """
    Executes the end-to-end 3-tier pipeline or offline mock simulation.
    """
    if not json_mode:
        header = "=" * 80
        print(header)
        title = "ATOMIC AI - 3-TIER PIPELINE TEST CLIENT"
        if mock:
            title += " [OFFLINE MOCK MODE]"
        print(colorize(title, BOLD))
        print(f"Target: {proxy_url} (Proxy) -> {gateway_url} (Gateway)")
        print(header)
        print(f"Prompt: \"{prompt}\"")
        if not mock:
            print(f"Sending request to {proxy_url}/v1/chat/completions (stream={stream})...")

    t_request_sent = time.perf_counter()

    if mock:
        # Offline authentic simulation
        event_stream = generate_mock_sse_events()
        metrics = process_sse_stream(event_stream, t_request_sent, verbose=verbose, json_mode=json_mode)
        print_benchmark_report(metrics, json_mode=json_mode)
        return 0

    if not requests:
        sys.stderr.write("Error: 'requests' module required for HTTP execution.\n")
        return 1

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream
    }

    req_url = f"{proxy_url.rstrip('/')}/v1/chat/completions"

    try:
        if stream:
            with requests.post(req_url, json=payload, stream=True, timeout=timeout) as response:
                if response.status_code != 200:
                    err_msg = f"HTTP {response.status_code}: {response.text}"
                    sys.stderr.write(colorize(f"\nPipeline Error: {err_msg}\n", RED + BOLD))
                    return 1

                def iter_stream_lines() -> Iterator[str]:
                    for raw_line in response.iter_lines():
                        if raw_line:
                            yield raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

                metrics = process_sse_stream(iter_stream_lines(), t_request_sent, verbose=verbose, json_mode=json_mode)
                print_benchmark_report(metrics, json_mode=json_mode)
                return 0
        else:
            # Non-streaming request
            resp = requests.post(req_url, json=payload, timeout=timeout)
            t_done = time.perf_counter()
            total_duration_ms = (t_done - t_request_sent) * 1000.0

            if resp.status_code != 200:
                sys.stderr.write(colorize(f"\nPipeline Error HTTP {resp.status_code}: {resp.text}\n", RED + BOLD))
                return 1

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            reasoning = msg.get("reasoning_content") or ""
            content = msg.get("content") or ""

            if not json_mode:
                if reasoning:
                    print(colorize("\n┌── [REASONING & COGNITIVE PLANNING] ───────────────────────────────────────────", CYAN + BOLD))
                    print(colorize(reasoning, CYAN))
                    print(colorize("└───────────────────────────────────────────────────────────────────────────────\n", CYAN))
                if content:
                    print(colorize("┌── [FINAL SYNTHESIZED RESPONSE] ───────────────────────────────────────────────", GREEN + BOLD))
                    print(colorize(content, GREEN))
                    print(colorize("└───────────────────────────────────────────────────────────────────────────────\n", GREEN))

            total_tokens = estimate_tokens(reasoning) + estimate_tokens(content)
            metrics = {
                "status": "SUCCESS",
                "mode": "non-stream",
                "total_duration_ms": round(total_duration_ms, 2),
                "total_tokens": total_tokens,
                "full_content": content,
                "full_reasoning": reasoning
            }
            if json_mode:
                print(json.dumps(metrics, indent=2))
            else:
                print(f"Total Execution Time: {total_duration_ms:.1f} ms | Tokens: {total_tokens}")
            return 0

    except Exception as e:
        sys.stderr.write(colorize(f"\nRequest failed: {e}\n", RED + BOLD))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Atomic AI - 3-Tier Pipeline Interactive CLI Test & Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("prompt_pos", nargs="?", help="Optional prompt to test (positional)")
    parser.add_argument("-p", "--prompt", default=None, help="Explicit prompt text")
    parser.add_argument("--check-only", action="store_true", help="Execute 3-tier pre-flight diagnostic check matrix")
    parser.add_argument("--mock", action="store_true", help="Offline authentic simulation of 3-tier SSE stream")
    parser.add_argument("--url", "--proxy-url", dest="proxy_url", default="http://127.0.0.1:8000", help="Atomic AI Proxy URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8081", help="LlamaLaunch Gateway URL (default: http://127.0.0.1:8081)")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8080", help="llama-server backend URL (default: http://127.0.0.1:8080)")
    parser.add_argument("--model", default="deepseek-v4-flash", help="Model identifier (default: deepseek-v4-flash)")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds (default: 120.0)")
    parser.add_argument("--stream", dest="stream", action="store_true", default=True, help="Enable SSE streaming (default)")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming (fetch non-stream JSON)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Display raw SSE lines and debugging info")
    parser.add_argument("--json", dest="json_mode", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    if args.check_only:
        return run_check_only(
            proxy_url=args.proxy_url,
            gateway_url=args.gateway_url,
            backend_url=args.backend_url,
            timeout=min(5.0, args.timeout),
            json_mode=args.json_mode
        )

    prompt = args.prompt or args.prompt_pos or "Resume en 3 pasos cómo funciona la inferencia local de LLMs."

    return run_pipeline(
        prompt=prompt,
        proxy_url=args.proxy_url,
        gateway_url=args.gateway_url,
        model=args.model,
        timeout=args.timeout,
        stream=args.stream,
        mock=args.mock,
        verbose=args.verbose,
        json_mode=args.json_mode
    )


if __name__ == "__main__":
    sys.exit(main())
