#!/bin/bash
# =============================================================================
# monitoring/setup_dashboard.sh
# Creates Cloud Monitoring dashboards and alerts for GPU/TPU training jobs.
# Also includes the cost comparison query after both jobs complete.
# =============================================================================

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="us-central1"

echo "Setting up Cloud Monitoring for project: ${PROJECT_ID}"

# ── Step 1: Create custom metric descriptors ──────────────────────────────────
echo ""
echo "[1/4] Creating custom metric descriptors..."

create_metric() {
  local type=$1 display=$2 unit=$3 desc=$4
  gcloud monitoring metric-descriptors create \
    "projects/${PROJECT_ID}/metricDescriptors/${type}" \
    --description="${desc}" \
    --display-name="${display}" \
    --type="${type}" \
    --metric-kind=GAUGE \
    --value-type=DOUBLE \
    --unit="${unit}" \
    --labels="job_name:STRING:Job name,hardware:STRING:GPU or TPU" \
    2>/dev/null || echo "  Metric ${type} already exists"
}

create_metric \
  "custom.googleapis.com/training/gpu_utilization" \
  "GPU Utilization" "%" \
  "GPU compute utilization percentage during training"

create_metric \
  "custom.googleapis.com/training/vram_used_gb" \
  "VRAM Used" "GBy" \
  "GPU VRAM currently used in gigabytes"

create_metric \
  "custom.googleapis.com/training/power_watts" \
  "GPU Power Draw" "W" \
  "GPU power consumption in watts"

create_metric \
  "custom.googleapis.com/training/temperature" \
  "GPU Temperature" "Cel" \
  "GPU die temperature in Celsius"

create_metric \
  "custom.googleapis.com/training/tokens_per_sec" \
  "Training Throughput" "1/s" \
  "Training tokens processed per second"

create_metric \
  "custom.googleapis.com/training/cost_usd" \
  "Estimated Cost USD" "USD" \
  "Cumulative estimated training cost in USD"

echo "[1/4] ✓ Metric descriptors created"

# ── Step 2: Create alerting policies ─────────────────────────────────────────
echo ""
echo "[2/4] Creating alerting policies..."

# Alert: GPU utilization dropped below 30% for 10 min (possible stall)
gcloud monitoring policies create \
  --policy-from-file=- << 'POLICY'
displayName: "GPU Utilization Too Low"
conditions:
  - displayName: "GPU util < 30% for 10 minutes"
    conditionThreshold:
      filter: >
        metric.type="custom.googleapis.com/training/gpu_utilization"
        resource.type="k8s_container"
      comparison: COMPARISON_LT
      thresholdValue: 30
      duration: 600s
      aggregations:
        - alignmentPeriod: 60s
          perSeriesAligner: ALIGN_MEAN
alertStrategy:
  autoClose: 1800s
combiner: OR
notificationChannels: []
POLICY

# Alert: VRAM usage above 95% (OOM risk)
gcloud monitoring policies create \
  --policy-from-file=- << 'POLICY'
displayName: "GPU VRAM Critical (>95%)"
conditions:
  - displayName: "VRAM used > 95% of total"
    conditionThreshold:
      filter: >
        metric.type="custom.googleapis.com/training/vram_used_gb"
        resource.type="k8s_container"
      comparison: COMPARISON_GT
      thresholdValue: 76
      duration: 120s
      aggregations:
        - alignmentPeriod: 60s
          perSeriesAligner: ALIGN_MEAN
alertStrategy:
  autoClose: 600s
combiner: OR
notificationChannels: []
POLICY

echo "[2/4] ✓ Alert policies created"

# ── Step 3: Create monitoring dashboard ──────────────────────────────────────
echo ""
echo "[3/4] Creating monitoring dashboard..."

gcloud monitoring dashboards create --config-from-file=- << 'DASHBOARD'
displayName: "Mistral Training — GPU vs TPU"
mosaicLayout:
  tiles:
    # ── GPU Utilization ──────────────────────────────────────────────────────
    - width: 6
      height: 4
      widget:
        title: "GPU Utilization (%)"
        xyChart:
          dataSets:
            - timeSeriesQuery:
                timeSeriesFilter:
                  filter: metric.type="custom.googleapis.com/training/gpu_utilization"
                unitOverride: "%"
              plotType: LINE
              targetAxis: Y1
          yAxis:
            label: "Utilization %"
            scale: LINEAR

    # ── Training Throughput ──────────────────────────────────────────────────
    - width: 6
      height: 4
      xPos: 6
      widget:
        title: "Training Throughput (tokens/sec)"
        xyChart:
          dataSets:
            - timeSeriesQuery:
                timeSeriesFilter:
                  filter: metric.type="custom.googleapis.com/training/tokens_per_sec"
              plotType: LINE
          yAxis:
            label: "Tokens/sec"

    # ── VRAM Usage ───────────────────────────────────────────────────────────
    - width: 6
      height: 4
      yPos: 4
      widget:
        title: "VRAM Usage (GB)"
        xyChart:
          dataSets:
            - timeSeriesQuery:
                timeSeriesFilter:
                  filter: metric.type="custom.googleapis.com/training/vram_used_gb"
              plotType: LINE
          yAxis:
            label: "GB"

    # ── Power Draw ───────────────────────────────────────────────────────────
    - width: 6
      height: 4
      xPos: 6
      yPos: 4
      widget:
        title: "GPU Power Draw (Watts)"
        xyChart:
          dataSets:
            - timeSeriesQuery:
                timeSeriesFilter:
                  filter: metric.type="custom.googleapis.com/training/power_watts"
              plotType: LINE
          yAxis:
            label: "Watts"

    # ── Cumulative Cost ──────────────────────────────────────────────────────
    - width: 12
      height: 4
      yPos: 8
      widget:
        title: "Estimated Cost (USD) — GPU vs TPU"
        xyChart:
          dataSets:
            - timeSeriesQuery:
                timeSeriesFilter:
                  filter: >
                    metric.type="custom.googleapis.com/training/cost_usd"
                    metric.labels.hardware="gpu"
              legendTemplate: "GPU"
              plotType: LINE
            - timeSeriesQuery:
                timeSeriesFilter:
                  filter: >
                    metric.type="custom.googleapis.com/training/cost_usd"
                    metric.labels.hardware="tpu"
              legendTemplate: "TPU"
              plotType: LINE
          yAxis:
            label: "USD"
DASHBOARD

echo "[3/4] ✓ Dashboard created"

# ── Step 4: Cost comparison report ───────────────────────────────────────────
echo ""
echo "[4/4] Fetching cost comparison from GCS reports..."
compare_costs() {
  local bucket="gs://${PROJECT_ID}-mistral-training/reports"

  # Find most recent GPU and TPU cost reports
  GPU_REPORT=$(gsutil ls "${bucket}/gpu_cost_*.json" 2>/dev/null | tail -1)
  TPU_REPORT=$(gsutil ls "${bucket}/tpu_cost_*.json" 2>/dev/null | tail -1)

  if [ -z "${GPU_REPORT}" ] && [ -z "${TPU_REPORT}" ]; then
    echo "No cost reports found yet. Run training jobs first."
    return
  fi

  echo ""
  echo "========================================================"
  echo "  TRAINING COST COMPARISON"
  echo "========================================================"

  if [ -n "${GPU_REPORT}" ]; then
    echo ""
    echo "GPU Training:"
    gsutil cat "${GPU_REPORT}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Hardware:     {d[\"gpu_type\"]} × {d[\"num_gpus\"]}')
print(f'  Wall time:    {d[\"wall_time_min\"]} minutes')
print(f'  Cost:         \${d[\"estimated_cost_usd\"]}')
print(f'  Price/hr:     \${d[\"price_per_gpu_hr\"]} × {d[\"num_gpus\"]} GPU(s)')
"
  fi

  if [ -n "${TPU_REPORT}" ]; then
    echo ""
    echo "TPU Training:"
    gsutil cat "${TPU_REPORT}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Hardware:     TPU {d[\"tpu_type\"]} ({d[\"num_chips\"]} chips)')
print(f'  Wall time:    {d[\"wall_time_min\"]} minutes')
print(f'  Cost:         \${d[\"estimated_cost_usd\"]}')
print(f'  Price/hr:     \${d[\"price_per_chip_hr\"]} × {d[\"num_chips\"]} chips')
"
  fi

  if [ -n "${GPU_REPORT}" ] && [ -n "${TPU_REPORT}" ]; then
    echo ""
    echo "Comparison:"
    gsutil cat "${GPU_REPORT}" "${TPU_REPORT}" | python3 -c "
import sys, json

data = []
for line in sys.stdin:
    try:
        data.append(json.loads(line))
    except:
        pass

if len(data) == 2:
    gpu = next(d for d in data if d['hardware'] == 'gpu')
    tpu = next(d for d in data if d['hardware'] == 'tpu')

    cost_saving   = gpu['estimated_cost_usd'] - tpu['estimated_cost_usd']
    cost_saving_pct = cost_saving / gpu['estimated_cost_usd'] * 100
    time_diff_min = gpu['wall_time_min'] - tpu['wall_time_min']
    speedup       = gpu['wall_time_min'] / tpu['wall_time_min']

    print(f'  GPU cost:     \${gpu[\"estimated_cost_usd\"]:.2f}')
    print(f'  TPU cost:     \${tpu[\"estimated_cost_usd\"]:.2f}')
    print(f'  Saving:       \${cost_saving:.2f} ({cost_saving_pct:.0f}% cheaper with TPU)')
    print(f'  GPU time:     {gpu[\"wall_time_min\"]} min')
    print(f'  TPU time:     {tpu[\"wall_time_min\"]} min')
    print(f'  TPU speedup:  {speedup:.1f}x faster')
    print()
    if cost_saving > 0:
        print('  Recommendation: TPU is cheaper and faster for this workload.')
    else:
        print('  Recommendation: GPU is more cost-effective for this workload.')
"
  fi

  echo "========================================================"
}

compare_costs

echo ""
echo "Cloud Monitoring dashboard URL:"
echo "https://console.cloud.google.com/monitoring/dashboards?project=${PROJECT_ID}"
echo ""
echo "GCS cost reports:"
gsutil ls "gs://${PROJECT_ID}-mistral-training/reports/" 2>/dev/null || \
  echo "No reports yet — run training jobs first"