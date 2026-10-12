"""Measure local CineMatch startup, API work, memory, and disk footprints."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import resource
import statistics
import sys
import time
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "reports" / "phase16_resource_measurements.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def tree_size(path: Path) -> int | None:
    """Count regular-file bytes once, without following cache symlinks."""
    if not path.exists():
        return None
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size

    total = 0
    seen_inodes: set[tuple[int, int]] = set()
    for item in path.rglob("*"):
        try:
            item_stat = item.lstat()
        except OSError:
            continue
        if item.is_symlink() or not item.is_file():
            continue
        inode = (item_stat.st_dev, item_stat.st_ino)
        if inode not in seen_inodes:
            total += item_stat.st_size
            seen_inodes.add(inode)
    return total


async def request(method: str, path: str, body: dict | None = None) -> tuple[int, object]:
    """Call FastAPI through ASGI, avoiding a network socket or HTTP client dependency."""
    from app.main import app

    request_body = json.dumps(body).encode("utf-8") if body is not None else b""
    messages: list[dict] = []
    received = False

    async def receive() -> dict:
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": request_body, "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": method, "scheme": "http",
        "path": path.split("?", 1)[0], "raw_path": path.split("?", 1)[0].encode(),
        "query_string": path.split("?", 1)[1].encode() if "?" in path else b"",
        "root_path": "", "headers": [(b"host", b"benchmark"),
        (b"content-type", b"application/json")],
        "server": ("benchmark", 80), "client": ("benchmark", 50000),
    }
    await app(scope, receive, send)
    start = next(item for item in messages if item["type"] == "http.response.start")
    payload = b"".join(item.get("body", b"") for item in messages
                       if item["type"] == "http.response.body")
    return start["status"], json.loads(payload) if payload else None


def summarize_latency(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("At least one latency sample is required")
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "samples": len(samples),
        "median_ms": round(statistics.median(samples) * 1000, 2),
        "max_ms": round(max(samples) * 1000, 2),
        "p95_ms_nearest_rank": round(ordered[p95_index] * 1000, 2),
    }


async def measure() -> dict:
    started = time.perf_counter()
    from app import main
    imported = time.perf_counter()

    async with main.app.router.lifespan_context(main.app):
        ready = time.perf_counter()
        cases = [
            ("health", "GET", "/api/v1/health", None),
            ("search", "GET", f"/api/v1/movies?{urlencode({'q': 'Toy Story', 'limit': 10})}", None),
            ("movie_details", "GET", "/api/v1/movies/1", None),
            ("movie_recommendations", "GET", "/api/v1/movies/1/recommendations?top_k=5", None),
            ("user_recommendations", "GET", "/api/v1/users/1/recommendations?top_k=5", None),
            ("favorite_recommendations", "POST", "/api/v1/recommendations/from-favorites",
             {"favorite_movie_ids": [1, 50, 260], "top_k": 5}),
        ]
        latency: dict[str, dict] = {}
        for name, method, path, body in cases:
            samples = []
            status = None
            for _ in range(5):
                before = time.perf_counter()
                status, _ = await request(method, path, body)
                samples.append(time.perf_counter() - before)
            latency[name] = {"http_status": status, **summarize_latency(samples)}

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    paths = {
        "project_data": ROOT / "data",
        "deployment_runtime_data": [
            ROOT / "data" / "enriched" / "movies.csv",
            ROOT / "data" / "cleaned" / "movies.csv",
            ROOT / "data" / "cleaned" / "ratings.csv",
            ROOT / "data" / "models" / "all_minilm_l6_v2_plots.npz",
        ],
        "python_virtual_environment": ROOT / ".venv",
        "huggingface_model_cache": Path.home() / ".cache" / "huggingface",
        "frontend_production_bundle": ROOT / "frontend" / "dist",
    }
    footprint = {
        name: (
            sum(tree_size(path) or 0 for path in value)
            if isinstance(value, list)
            else tree_size(value)
        )
        for name, value in paths.items()
    }
    loaded_modules = {
        name: name in sys.modules
        for name in ("numpy", "pandas", "scipy", "sklearn", "torch", "sentence_transformers")
    }
    return {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "measurement_method": "Single-process FastAPI lifespan + 5 sequential ASGI requests per endpoint; no socket or external service.",
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "startup_seconds": {
            "python_imports": round(imported - started, 3),
            "model_and_catalog_lifespan": round(ready - imported, 3),
            "total_until_ready": round(ready - started, 3),
        },
        "peak_rss_bytes": peak_bytes,
        "peak_rss_mib": round(peak_bytes / (1024 * 1024), 1),
        "loaded_modules_at_end": loaded_modules,
        "api_latency": latency,
        "disk_footprint_bytes": footprint,
        "disk_footprint_mib": {
            name: round(size / (1024 * 1024), 2) if size is not None else None
            for name, size in footprint.items()
        },
        "notes": [
            "Peak RSS covers one process with the full recommender and collaborative item-item matrix loaded.",
            "The embedding cache is used; PyTorch and sentence-transformers are not imported for serving when that cache validates.",
            "Latency measures warm in-process ASGI request handling, not socket, network transit, or TLS.",
            "For five observations, nearest-rank p95 is the maximum observation; the sample is too small for stable tail-latency claims.",
            "This result is specific to the machine, package versions, model cache, and warm request sequence.",
        ],
    }


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    result = asyncio.run(measure())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "report_path": str(OUTPUT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
