# =============================================================================
# monitoring/metrics_reference.md
# Complete reference for GPU/TPU metrics available in Cloud Monitoring
# and how to query them during/after training on GKE
# =============================================================================

# GKE + GPU/TPU — Cloud Monitoring Metrics Reference

## 1. Built-in GKE Metrics (zero config — always available)

These are automatically collected by GKE — no exporter needed.

| Metric | Path | What it shows |
|---|---|---|
| Container CPU | `kubernetes.io/container/cpu/core_usage_time` | CPU cores used by pod |
| Container memory | `kubernetes.io/container/memory/used_bytes` | RAM used by pod |
| Container restart count | `kubernetes.io/container/restart_count` | Pod restarts (OOM, crash) |
| Node CPU | `kubernetes.io/node/cpu/total_cores` | Total CPU on node |
| Node memory | `kubernetes.io/node/memory/total_bytes` | Total RAM on node |
| Pod start latency | `kubernetes.io/pod/volume/total_bytes` | Disk used by pod volumes |

### Query in Cloud Monitoring UI:
```
resource.type="k8s_container"
resource.labels.cluster_name="mistral-training-cluster"
resource.labels.namespace_name="default"
metric.type="kubernetes.io/container/memory/used_bytes"
```

---

## 2. GPU Metrics via DCGM Exporter (install separately)

DCGM (Data Center GPU Manager) exports detailed NVIDIA GPU metrics.

### Install DCGM Exporter on GKE:
```bash
helm repo add gpu-helm-charts \
  https://nvidia.github.io/dcgm-exporter/helm-charts
helm install dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --namespace monitoring \
  --create-namespace \
  --set serviceMonitor.enabled=true
```

### Key DCGM Metrics:

| Metric name | Unit | What it measures |
|---|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | % | GPU compute utilization — primary health indicator |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | % | Memory copy engine utilization |
| `DCGM_FI_DEV_FB_USED` | MiB | VRAM currently used (framebuffer) |
| `DCGM_FI_DEV_FB_FREE` | MiB | VRAM free |
| `DCGM_FI_DEV_POWER_USAGE` | W | GPU power draw |
| `DCGM_FI_DEV_GPU_TEMP` | °C | GPU die temperature |
| `DCGM_FI_DEV_SM_CLOCK` | MHz | Streaming Multiprocessor clock speed |
| `DCGM_FI_DEV_MEM_CLOCK` | MHz | Memory clock speed |
| `DCGM_FI_DEV_PCIE_TX_BYTES` | B | PCIe bytes sent (host→GPU) |
| `DCGM_FI_DEV_PCIE_RX_BYTES` | B | PCIe bytes received (GPU→host) |
| `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | B/s | NVLink bandwidth (multi-GPU) |

### Diagnose bottlenecks using DCGM:
```bash
# GPU utilization dropping → CPU bottleneck or data stall
# DCGM_FI_DEV_GPU_UTIL < 50% while training

# VRAM near limit → OOM risk
# DCGM_FI_DEV_FB_USED / (FB_USED + FB_FREE) > 90%

# Low power + high util → memory-bound kernel
# DCGM_FI_DEV_POWER_USAGE < 200W while DCGM_FI_DEV_GPU_UTIL > 90%
```

---

## 3. TPU Metrics (Cloud TPU built-in)

TPU metrics are available in Cloud Monitoring under the `tpu_node` resource type.

| Metric | Path | What it measures |
|---|---|---|
| TensorCore utilization | `tpu.googleapis.com/container/accelerator/duty_cycle` | % time TensorCore is active — equivalent to GPU util |
| Memory usage | `tpu.googleapis.com/container/accelerator/memory_used` | HBM used per chip |
| Memory bandwidth | `tpu.googleapis.com/container/accelerator/memory_bandwidth_utilization` | HBM bandwidth utilization |
| Network bandwidth | `tpu.googleapis.com/container/accelerator/network_bandwidth` | ICI inter-chip bandwidth |
| Idle time | `tpu.googleapis.com/container/accelerator/idle` | % time chip is idle (bad) |

### Query TPU metrics:
```
resource.type="k8s_node"
metric.type="tpu.googleapis.com/container/accelerator/duty_cycle"
resource.labels.cluster_name="mistral-training-cluster"
```

---

## 4. Custom Metrics (from gpu_metrics_exporter.py)

These are pushed by our custom exporter. Available in Cloud Monitoring
under Custom Metrics → training.

| Metric | Type | Labels |
|---|---|---|
| `custom.googleapis.com/training/gpu_utilization` | Gauge | job_name, hardware |
| `custom.googleapis.com/training/vram_used_gb` | Gauge | job_name, hardware |
| `custom.googleapis.com/training/power_watts` | Gauge | job_name, hardware |
| `custom.googleapis.com/training/temperature` | Gauge | job_name, hardware |
| `custom.googleapis.com/training/tokens_per_sec` | Gauge | job_name, hardware |
| `custom.googleapis.com/training/cost_usd` | Counter | job_name, hardware |

---

## 5. kubectl Commands for Live Monitoring

```bash
# Watch training job status
kubectl get jobs -w

# Stream logs from GPU training (real-time loss, throughput)
kubectl logs -f job/mistral-gpu-training -c trainer

# Stream logs from TPU training
kubectl logs -f job/mistral-tpu-training -c trainer

# Watch pod resource usage live (requires metrics-server)
kubectl top pods -l app=mistral-training --containers

# Watch node GPU usage
kubectl top nodes -l pool-type=gpu

# Describe pod for events (spot preemption, OOM killer, etc.)
kubectl describe pod -l job-name=mistral-gpu-training

# Check GPU allocation on nodes
kubectl get nodes -l pool-type=gpu -o json | \
  python3 -c "
import json, sys
data = json.load(sys.stdin)
for node in data['items']:
    name = node['metadata']['name']
    alloc = node['status'].get('allocatable', {})
    cap   = node['status'].get('capacity', {})
    print(f'{name}: GPU allocatable={alloc.get(\"nvidia.com/gpu\",0)} / capacity={cap.get(\"nvidia.com/gpu\",0)}')
"

# Check job completion and exit code
kubectl get job mistral-gpu-training -o jsonpath='{.status}'

# Get cost report from GCS after job finishes
gsutil cat gs://$(gcloud config get project)-mistral-training/reports/gpu_cost_*.json | \
  python3 -m json.tool
```

---

## 6. Cost Estimation Formulas

### GPU training cost:
```
cost = (wall_time_hours) × (price_per_gpu_hour) × (num_gpus)

Example: 1.13 hr × $4.26/hr × 2 GPUs = $9.63
```

### TPU training cost:
```
cost = (wall_time_hours) × (price_per_chip_hour) × (num_chips)

Example: 0.73 hr × $3.22/hr × 8 chips = $18.81
```

### Spot VM discount:
```
GPU spot (A100): ~70% discount → $4.26 × 0.30 = $1.28/hr per GPU
TPU spot (v4):   ~60% discount → $3.22 × 0.40 = $1.29/hr per chip
```

### Real-time cost in Python:
```python
import time

GPU_PRICE  = 4.26   # per A100 per hour (on-demand)
TPU_PRICE  = 3.22   # per v4 chip per hour
NUM_GPUS   = 2
NUM_CHIPS  = 8

start = time.time()

# ... training ...

elapsed_hr  = (time.time() - start) / 3600
gpu_cost    = elapsed_hr * GPU_PRICE * NUM_GPUS
tpu_cost    = elapsed_hr * TPU_PRICE * NUM_CHIPS
print(f"GPU cost: ${gpu_cost:.2f}")
print(f"TPU cost: ${tpu_cost:.2f}")
```

---

## 7. What Good vs Bad Metrics Look Like

### Healthy GPU training:
```
DCGM_FI_DEV_GPU_UTIL     = 80–95%    ✓ compute-bound
DCGM_FI_DEV_POWER_USAGE  = 300–400W  ✓ near TDP
DCGM_FI_DEV_FB_USED      = 50–85%    ✓ good utilisation
tokens_per_sec            = 12,000+   ✓ good throughput
```

### GPU bottleneck (CPU/data starving GPU):
```
DCGM_FI_DEV_GPU_UTIL     = 20–50%    ✗ too low (oscillating)
DCGM_FI_DEV_POWER_USAGE  = <150W     ✗ GPU idle between batches
tokens_per_sec            = <5,000    ✗ poor throughput
Fix: increase DataLoader num_workers, use pin_memory=True
```

### VRAM pressure (OOM incoming):
```
DCGM_FI_DEV_FB_USED / total > 90%    ✗ dangerous
Fix: reduce batch size, enable gradient checkpointing
```

### Healthy TPU training:
```
duty_cycle                = 70–90%    ✓ MXU busy
memory_bandwidth_util     = 60–80%    ✓ good HBM utilisation
idle                      = <10%      ✓ low idle time
tokens_per_sec            = 18,000+   ✓ better than GPU
```