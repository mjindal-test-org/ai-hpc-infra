#!/bin/bash
# =============================================================================
# RUNBOOK.sh — Complete end-to-end deployment for GPU and TPU training on GKE
#
# Read this file top to bottom. Each section is a numbered step.
# Run each step manually in order — do not run the whole file at once.
# =============================================================================

export PROJECT_ID="your-project-id"   # ← CHANGE THIS
export REGION="us-central1"
export CLUSTER_NAME="mistral-training-cluster"
export GCS_BUCKET="${PROJECT_ID}-mistral-training"

# =============================================================================
# STEP 1: Prerequisites
# =============================================================================

# Install required tools
gcloud components install kubectl gke-gcloud-auth-plugin
gcloud components install beta

# Enable required GCP APIs
gcloud services enable \
  container.googleapis.com \
  monitoring.googleapis.com \
  tpu.googleapis.com \
  storage.googleapis.com \
  iamcredentials.googleapis.com \
  --project="${PROJECT_ID}"

echo "✓ Step 1 complete"

# =============================================================================
# STEP 2: Create cluster
# =============================================================================

bash cluster/01_setup_cluster.sh

# Verify cluster is up
kubectl get nodes -L pool-type,accelerator

echo "✓ Step 2 complete"

# =============================================================================
# STEP 3: Set up IAM and service accounts
# =============================================================================

# Create GCP service account for pods
gcloud iam service-accounts create training-sa \
  --display-name="Mistral Training Service Account" \
  --project="${PROJECT_ID}" 2>/dev/null || echo "SA already exists"

# Grant GCS access
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:training-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Grant Cloud Monitoring write access
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:training-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/monitoring.metricWriter"

# Workload Identity binding: allow the Kubernetes SA to impersonate GCP SA
gcloud iam service-accounts add-iam-policy-binding \
  "training-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[default/training-sa]"

# Apply RBAC
kubectl apply -f cluster/rbac.yaml

echo "✓ Step 3 complete"

# =============================================================================
# STEP 4: Upload training code to GCS
# =============================================================================

# Create code directory in GCS
gsutil mb -l "${REGION}" "gs://${GCS_BUCKET}" 2>/dev/null || true

# Upload all Python training files
gsutil -m cp \
  ../config.py \
  ../dataset.py \
  ../model_utils.py \
  ../train_gpu.py \
  ../train_tpu.py \
  ../inference_gpu.py \
  ../inference_tpu.py \
  monitoring/gpu_metrics_exporter.py \
  monitoring/tpu_metrics_exporter.py \
  "gs://${GCS_BUCKET}/code/"

echo "Files uploaded:"
gsutil ls "gs://${GCS_BUCKET}/code/"

echo "✓ Step 4 complete"

# =============================================================================
# STEP 5: Run GPU training job
# =============================================================================

# Replace PROJECT_ID placeholder in YAML
sed "s/\$(PROJECT_ID)/${PROJECT_ID}/g" gpu/training_job.yaml | \
  kubectl apply -f -

echo "GPU training job submitted"
echo "Watch logs: kubectl logs -f job/mistral-gpu-training -c trainer"
echo "Check status: kubectl get job mistral-gpu-training"

# Wait for job to start
kubectl wait --for=condition=ready \
  pod -l job-name=mistral-gpu-training \
  --timeout=300s

# Stream logs
kubectl logs -f job/mistral-gpu-training -c trainer &
GPU_LOG_PID=$!

echo ""
echo "Waiting for GPU job to complete..."
kubectl wait --for=condition=complete \
  job/mistral-gpu-training \
  --timeout=7200s    # 2 hour timeout

kill ${GPU_LOG_PID} 2>/dev/null || true
echo "✓ Step 5 complete — GPU training done"

# =============================================================================
# STEP 6: Run TPU training job
# =============================================================================

sed "s/\$(PROJECT_ID)/${PROJECT_ID}/g" tpu/training_job.yaml | \
  kubectl apply -f -

echo "TPU training job submitted"
echo "Watch logs: kubectl logs -f job/mistral-tpu-training -c trainer"

kubectl wait --for=condition=ready \
  pod -l job-name=mistral-tpu-training \
  --timeout=300s

kubectl logs -f job/mistral-tpu-training -c trainer &
TPU_LOG_PID=$!

echo "Waiting for TPU job to complete..."
kubectl wait --for=condition=complete \
  job/mistral-tpu-training \
  --timeout=7200s

kill ${TPU_LOG_PID} 2>/dev/null || true
echo "✓ Step 6 complete — TPU training done"

# =============================================================================
# STEP 7: Set up monitoring and compare costs
# =============================================================================

bash monitoring/setup_dashboard.sh

echo ""
echo "Cost reports in GCS:"
gsutil ls "gs://${GCS_BUCKET}/reports/"

echo ""
echo "Fetch GPU cost report:"
gsutil cat "gs://${GCS_BUCKET}/reports/gpu_cost_*.json" | python3 -m json.tool

echo ""
echo "Fetch TPU cost report:"
gsutil cat "gs://${GCS_BUCKET}/reports/tpu_cost_*.json" | python3 -m json.tool

echo "✓ Step 7 complete"

# =============================================================================
# STEP 8: Deploy inference (GPU)
# =============================================================================

sed "s/\$(PROJECT_ID)/${PROJECT_ID}/g" gpu/inference_deployment.yaml | \
  kubectl apply -f -

echo "Waiting for inference deployment to be ready..."
kubectl wait --for=condition=available \
  deployment/mistral-gpu-inference \
  --timeout=600s

INFERENCE_IP=$(kubectl get svc mistral-inference-svc \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo ""
echo "✓ Inference endpoint: http://${INFERENCE_IP}/v1/chat/completions"
echo ""
echo "Test it:"
cat << EOF
curl http://${INFERENCE_IP}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "mistral",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "max_tokens": 200
  }'
EOF

echo "✓ Step 8 complete"

# =============================================================================
# STEP 9: Clean up (IMPORTANT — stops billing)
# =============================================================================

echo ""
echo "To clean up all resources and STOP BILLING:"
echo ""
echo "# Delete training jobs"
echo "kubectl delete job mistral-gpu-training mistral-tpu-training"
echo ""
echo "# Delete inference deployment"
echo "kubectl delete -f gpu/inference_deployment.yaml"
echo ""
echo "# Scale node pools to 0 (stops VM billing, keeps cluster)"
echo "gcloud container node-pools update ${GPU_NODE_POOL} \\"
echo "  --cluster=${CLUSTER_NAME} --region=${REGION} \\"
echo "  --min-nodes=0 --max-nodes=0"
echo ""
echo "# Or delete the entire cluster (full cleanup)"
echo "gcloud container clusters delete ${CLUSTER_NAME} \\"
echo "  --region=${REGION} --quiet"
echo ""
echo "# Empty and delete GCS bucket"
echo "gsutil -m rm -r gs://${GCS_BUCKET}"