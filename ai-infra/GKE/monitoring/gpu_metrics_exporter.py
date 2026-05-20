"""
monitoring/gpu_metrics_exporter.py
Exports GPU metrics to Cloud Monitoring during training.

Metrics exported every 60 seconds:
  - GPU utilization %
  - VRAM used / total
  - GPU power draw (watts)
  - GPU temperature
  - Training tokens/sec (from log file)
  - Estimated cost so far

Run: python gpu_metrics_exporter.py --port 8080
"""

import argparse
import time
import json
import os
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

try:
    from google.cloud import monitoring_v3
    from google.api import metric_pb2
    from google.api import monitored_resource_pb2
    GCP_MONITORING = True
except ImportError:
    GCP_MONITORING = False
    print("google-cloud-monitoring not installed — metrics will be logged only")


PROJECT_ID  = os.environ.get("PROJECT_ID", "your-project-id")
JOB_NAME    = os.environ.get("JOB_NAME", "mistral-gpu-training")
GPU_PRICE   = float(os.environ.get("GPU_PRICE_PER_HR", "4.26"))

# Shared metrics dict updated by background thread
METRICS = {
    "gpu_util":        0.0,
    "vram_used_gb":    0.0,
    "vram_total_gb":   80.0,
    "power_watts":     0.0,
    "temp_celsius":    0.0,
    "tokens_per_sec":  0.0,
    "elapsed_hours":   0.0,
    "cost_usd":        0.0,
    "last_updated":    "",
}
START_TIME = time.time()


def query_nvidia_smi() -> dict:
    """Query nvidia-smi for current GPU stats."""
    try:
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,"
                         "power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            return {}

        # Parse first GPU (index 0)
        line   = result.stdout.strip().split("\n")[0]
        parts  = [p.strip() for p in line.split(",")]
        return {
            "gpu_util":     float(parts[0]),
            "vram_used_gb": float(parts[1]) / 1024,
            "vram_total_gb":float(parts[2]) / 1024,
            "power_watts":  float(parts[3]),
            "temp_celsius": float(parts[4]),
        }
    except Exception as e:
        print(f"nvidia-smi error: {e}")
        return {}


def parse_tokens_per_sec() -> float:
    """
    Parse tokens/sec from training log file.
    The training script writes a line like:
      "Step 10/308 | Loss 1.234 | 12,345 tok/s"
    """
    try:
        log_path = "/tmp/training.log"
        if not os.path.exists(log_path):
            return 0.0
        with open(log_path, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "tok/s" in line:
                # Extract number before "tok/s"
                parts = line.split("tok/s")[0].strip().split()
                val   = parts[-1].replace(",", "")
                return float(val)
    except Exception:
        pass
    return 0.0


def collect_metrics():
    """Background thread: collect metrics every 30 seconds."""
    global METRICS
    while True:
        gpu_stats = query_nvidia_smi()

        elapsed_sec   = time.time() - START_TIME
        elapsed_hours = elapsed_sec / 3600
        cost_usd      = elapsed_hours * GPU_PRICE  # single GPU cost

        METRICS.update({
            **gpu_stats,
            "tokens_per_sec": parse_tokens_per_sec(),
            "elapsed_hours":  round(elapsed_hours, 4),
            "cost_usd":       round(cost_usd, 4),
            "last_updated":   datetime.utcnow().isoformat(),
        })

        # Push to Cloud Monitoring if available
        if GCP_MONITORING:
            push_to_cloud_monitoring(METRICS)

        # Print summary to stdout (visible in kubectl logs)
        print(
            f"[GPU Monitor] "
            f"util={METRICS['gpu_util']:.0f}% | "
            f"vram={METRICS['vram_used_gb']:.1f}/{METRICS['vram_total_gb']:.0f}GB | "
            f"power={METRICS['power_watts']:.0f}W | "
            f"temp={METRICS['temp_celsius']:.0f}°C | "
            f"tok/s={METRICS['tokens_per_sec']:,.0f} | "
            f"elapsed={elapsed_hours:.2f}hr | "
            f"cost=${METRICS['cost_usd']:.2f}"
        )
        time.sleep(30)


def push_to_cloud_monitoring(metrics: dict):
    """Push metrics to Google Cloud Monitoring as custom metrics."""
    try:
        client   = monitoring_v3.MetricServiceClient()
        project  = f"projects/{PROJECT_ID}"
        now      = time.time()
        interval = monitoring_v3.TimeInterval(
            {"end_time": {"seconds": int(now)}}
        )

        metric_map = {
            "custom.googleapis.com/training/gpu_utilization":  metrics["gpu_util"],
            "custom.googleapis.com/training/vram_used_gb":     metrics["vram_used_gb"],
            "custom.googleapis.com/training/power_watts":      metrics["power_watts"],
            "custom.googleapis.com/training/temperature":      metrics["temp_celsius"],
            "custom.googleapis.com/training/tokens_per_sec":   metrics["tokens_per_sec"],
            "custom.googleapis.com/training/cost_usd":         metrics["cost_usd"],
        }

        for metric_type, value in metric_map.items():
            series = monitoring_v3.TimeSeries()
            series.metric.type = metric_type
            series.metric.labels["job_name"] = JOB_NAME
            series.metric.labels["hardware"] = "gpu"
            series.resource.type = "k8s_container"
            series.resource.labels["project_id"] = PROJECT_ID

            point = monitoring_v3.Point(
                {"interval": interval,
                 "value": {"double_value": float(value)}}
            )
            series.points = [point]

            client.create_time_series(
                name=project, time_series=[series]
            )
    except Exception as e:
        # Don't crash the exporter on monitoring failures
        print(f"Cloud Monitoring push error: {e}")


# ── Prometheus-compatible /metrics endpoint ────────────────────────────────────
class MetricsHandler(BaseHTTPRequestHandler):
    """
    Simple HTTP server exposing metrics in Prometheus text format.
    GKE can scrape this via the prometheus.io annotations on the pod.
    """
    def do_GET(self):
        if self.path == "/metrics":
            m = METRICS
            body = "\n".join([
                f'# HELP gpu_utilization_percent GPU compute utilization',
                f'# TYPE gpu_utilization_percent gauge',
                f'gpu_utilization_percent{{job="{JOB_NAME}"}} {m["gpu_util"]}',
                f'# HELP vram_used_gigabytes VRAM currently used',
                f'# TYPE vram_used_gigabytes gauge',
                f'vram_used_gigabytes{{job="{JOB_NAME}"}} {m["vram_used_gb"]}',
                f'# HELP gpu_power_watts GPU power draw',
                f'# TYPE gpu_power_watts gauge',
                f'gpu_power_watts{{job="{JOB_NAME}"}} {m["power_watts"]}',
                f'# HELP gpu_temperature_celsius GPU die temperature',
                f'# TYPE gpu_temperature_celsius gauge',
                f'gpu_temperature_celsius{{job="{JOB_NAME}"}} {m["temp_celsius"]}',
                f'# HELP training_tokens_per_second Training throughput',
                f'# TYPE training_tokens_per_second gauge',
                f'training_tokens_per_second{{job="{JOB_NAME}"}} {m["tokens_per_sec"]}',
                f'# HELP training_cost_usd Estimated cost in USD so far',
                f'# TYPE training_cost_usd counter',
                f'training_cost_usd{{job="{JOB_NAME}"}} {m["cost_usd"]}',
                "",  # trailing newline
            ]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # silence HTTP request logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    # Start background metrics collection
    t = threading.Thread(target=collect_metrics, daemon=True)
    t.start()

    # Start HTTP server for Prometheus scraping
    server = HTTPServer(("0.0.0.0", args.port), MetricsHandler)
    print(f"GPU metrics exporter running on port {args.port}")
    print(f"Metrics: http://localhost:{args.port}/metrics")
    server.serve_forever()