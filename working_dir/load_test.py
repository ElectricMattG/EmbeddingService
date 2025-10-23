#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.request
import urllib.error


def post_json(url: str, payload: dict, timeout: float = 30.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            _ = resp.read()  # drain
    except urllib.error.HTTPError as e:
        status = e.code
    except Exception:
        status = 0
    latency = time.time() - start
    return status, latency


essay = (
    "This is a moderately long text used for load testing the embeddings endpoint. "
    * 20
)


def run_load(base_url: str, total: int, workers: int, segmentation: bool):
    url = base_url.rstrip("/") + "/embeddings"
    payload = {
        "texts": [
            essay,
            "Short text for variety.",
            "Another input to increase batch coverage.",
        ],
        "use_segmentation": segmentation,
        "include_segment_embeddings": False,
        "show_progress": False,
    }

    statuses = []
    latencies = []

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(post_json, url, payload) for _ in range(total)]
        for f in cf.as_completed(futs):
            s, l = f.result()
            statuses.append(s)
            latencies.append(l)

    ok = sum(1 for s in statuses if s == 200)
    fail = len(statuses) - ok
    # Build simple status distribution
    dist = {}
    for s in statuses:
        dist[s] = dist.get(s, 0) + 1
    latencies_sorted = sorted(latencies)
    mean = sum(latencies) / len(latencies) if latencies else 0
    def pct(p):
        if not latencies_sorted:
            return 0
        k = int(round((p/100.0) * (len(latencies_sorted)-1)))
        return latencies_sorted[k]

    print(f"Total: {len(statuses)} 200: {ok} non-200: {fail}")
    top = sorted(dist.items(), key=lambda x: (-x[1], x[0]))[:5]
    print("status-dist:", ", ".join(f"{k}:{v}" for k,v in top))
    print(f"mean: {mean:.4f}s p50: {pct(50):.4f}s p95: {pct(95):.4f}s p99: {pct(99):.4f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL (without trailing slash)")
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--segmentation", choices=["true", "false"], default="false")
    args = ap.parse_args()
    segmentation = args.segmentation.lower() == "true"
    print(f"Running load: total={args.requests} conc={args.concurrency} segmentation={segmentation}")
    run_load(args.url, args.requests, args.concurrency, segmentation)


if __name__ == "__main__":
    main()
