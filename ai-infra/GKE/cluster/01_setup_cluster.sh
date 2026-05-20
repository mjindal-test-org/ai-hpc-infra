#!/bin/bash
# =============================================================================
# 01_setup_cluster.sh — Create GKE cluster with GPU and TPU node pools
#
# What this creates:
#   - 1 GKE Autopilot-style Standard cluster (us-central1)
#   - 1 GPU node pool  : 2× n1-standard-16 machines with 1× A100 80GB each
#   - 1 TPU node pool  : 1× TPU v4-8 node (8 chips on 1 host machine)
#   - 1 CPU node pool  : for system workloads (monitoring, inference proxy)
#
# Cost estimate (us-central1, on-demand):
#   GPU node pool : $4.26/hr per node × 2 nodes  = ~$8.52/hr
#   TPU v4-8      : $3.22/chip/hr × 8 chips      = ~$25.76/hr
#   CPU pool      : $0.48/hr × 2 nodes            = ~$0.96/hr
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#   gcloud services enable container.googleapis.com
#   gcloud services enable monitoring.googleapis.com
#   gcloud services enable tpu.googleapis.com
# =============================================================================

set -euo pipefail

# ── Configuration — edit these ────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:-your-project-id}"
CLUSTER_NAME="mistral-training-cluster"
REGION="us-central1"
ZONE="${REGION}-a"

# GPU node pool
GPU_NODE_POOL="gpu-pool"
GPU_MACHINE_TYPE="a2-highgpu-1g"   # 1× A100 80GB per node
GPU_NUM_NODES=2                     # 2 nodes = 2 A100s for multi-GPU training
GPU_DISK_SIZE="200"                 # GB — needs space for model cache

# TPU node pool
TPU_NODE_POOL="tpu-pool"
TPU_TYPE="v4-8"                     # 8 chips on 1 host machine
TPU_TOPOLOGY="2x2x2"               # physical arrangement of the 8 chips

# CPU node pool (for system / inference proxy workloads)
CPU_NODE_POOL="cpu-pool"
CPU_MACHINE_TYPE="n1-standard-4"
CPU_NUM_NODES=2

echo "============================================================"
echo "Creating GKE cluster: ${CLUSTER_NAME}"
echo "Project: ${PROJECT_ID} | Region: ${REGION}"
echo "============================================================"

# ── Step 1: Create the cluster (control plane only, no default node pool) ────
echo ""
echo "[1/5] Creating cluster control plane..."
gcloud container clusters create "${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --release-channel="stable" \
  --workload-pool="${PROJECT_ID}.svc.id.goog" \
  --enable-ip-alias \
  --no-enable-basic-auth \
  --no-create-subnetwork \
  --num-nodes=1 \
  --no-enable-autoupgrade \
  --logging=SYSTEM,WORKLOAD \
  --monitoring=SYSTEM,WORKLOAD \
  --addons=GcePersistentDiskCsiDriver \
  --no-enable-master-authorized-networks \
  --machine-type="e2-medium" \   # control plane nodes — small is fine
  --disk-size="50" \
  --tags="gke-cluster"

echo "[1/5] ✓ Cluster created"

# ── Step 2: Create CPU node pool ──────────────────────────────────────────────
echo ""
echo "[2/5] Creating CPU node pool (system workloads)..."
gcloud container node-pools create "${CPU_NODE_POOL}" \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --machine-type="${CPU_MACHINE_TYPE}" \
  --num-nodes="${CPU_NUM_NODES}" \
  --disk-size="100" \
  --disk-type="pd-ssd" \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=4 \
  --node-labels="pool-type=cpu" \
  --no-enable-autoupgrade

echo "[2/5] ✓ CPU pool created"

# ── Step 3: Create GPU node pool ─────────────────────────────────────────────
echo ""
echo "[3/5] Creating GPU node pool (A100 80GB)..."
gcloud container node-pools create "${GPU_NODE_POOL}" \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --machine-type="${GPU_MACHINE_TYPE}" \
  --accelerator="type=nvidia-tesla-a100,count=1,gpu-driver-version=latest" \
  --num-nodes="${GPU_NUM_NODES}" \
  --disk-size="${GPU_DISK_SIZE}" \
  --disk-type="pd-ssd" \
  --enable-autoscaling \
  --min-nodes=0 \           # scale to 0 when idle — saves cost
  --max-nodes=8 \
  --node-labels="pool-type=gpu,accelerator=a100" \
  --node-taints="nvidia.com/gpu=present:NoSchedule" \  # only GPU pods land here
  --no-enable-autoupgrade \
  --spot                    # use spot VMs — ~70% cheaper, may be preempted

echo "[3/5] ✓ GPU pool created"

# ── Step 4: Create TPU node pool ─────────────────────────────────────────────
echo ""
echo "[4/5] Creating TPU node pool (v4-8)..."
gcloud container node-pools create "${TPU_NODE_POOL}" \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --machine-type="ct4p-hightpu-4t" \  # TPU v4 host machine type
  --tpu-topology="${TPU_TOPOLOGY}" \
  --num-nodes=1 \
  --disk-size="200" \
  --disk-type="pd-ssd" \
  --node-labels="pool-type=tpu,tpu-type=v4-8" \
  --node-taints="google.com/tpu=present:NoSchedule" \  # only TPU pods land here
  --no-enable-autoupgrade \
  --spot                    # TPU spot = ~60% cheaper

echo "[4/5] ✓ TPU pool created"

# ── Step 5: Configure kubectl ─────────────────────────────────────────────────
echo ""
echo "[5/5] Configuring kubectl..."
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}"

# Verify nodes are ready
echo ""
echo "Node pool status:"
kubectl get nodes -L pool-type,accelerator --sort-by='.metadata.labels.pool-type'

echo ""
echo "============================================================"
echo "✓ Cluster setup complete!"
echo ""
echo "Next steps:"
echo "  1. Create GCS bucket:  gsutil mb gs://${PROJECT_ID}-mistral-training"
echo "  2. Deploy GPU training: kubectl apply -f gpu/training_job.yaml"
echo "  3. Deploy TPU training: kubectl apply -f tpu/training_job.yaml"
echo "  4. View monitoring:     ./monitoring/setup_dashboard.sh"
echo "============================================================"

# ── Create GCS bucket for model storage ──────────────────────────────────────
echo ""
echo "Creating GCS bucket for model artifacts..."
gsutil mb -l "${REGION}" "gs://${PROJECT_ID}-mistral-training" || \
  echo "Bucket already exists — skipping"

gsutil lifecycle set monitoring/gcs_lifecycle.json \
  "gs://${PROJECT_ID}-mistral-training" || true

echo "✓ GCS bucket: gs://${PROJECT_ID}-mistral-training"