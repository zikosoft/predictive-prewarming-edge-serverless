#!/usr/bin/env bash
# Builds the application image (the exact same app/telemetry_app.py
# used in the process-level emulation) and deploys both the container
# (always-on) and Knative (serverless) architectures. Docker Desktop's
# Kubernetes shares the same Docker image store as `docker build`, so
# no push/registry/`kind load` step is needed -- the image is visible
# to the cluster the moment `docker build` finishes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "[*] Building telemetry-app:bench from ${ROOT}/app ..."
docker build -t telemetry-app:bench "${ROOT}/app"

echo "[*] Deploying the container (always-on) architecture..."
kubectl apply -f "${ROOT}/real-infra/k8s/container-deployment.yaml"

echo "[*] Deploying the Knative (serverless) architecture..."
kubectl apply -f "${ROOT}/real-infra/k8s/knative-service.yaml"

echo "[*] Waiting for the container Deployment to be ready..."
kubectl rollout status deployment/telemetry-container --timeout=120s

echo "[*] Waiting for the Knative Service to report Ready..."
kubectl wait --for=condition=Ready ksvc/telemetry-serverless --timeout=180s

echo "[*] Waiting for the container Service's LoadBalancer external IP..."
for i in $(seq 1 30); do
  ip=$(kubectl get svc telemetry-container-svc -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
  [[ -n "$ip" ]] && break
  sleep 2
done

KNATIVE_URL=$(kubectl get ksvc telemetry-serverless -o jsonpath='{.status.url}')
echo
echo "=================================================================="
echo "Container endpoint:  http://localhost:8080/ingest   (health: /health)"
echo "Knative endpoint:     ${KNATIVE_URL}/ingest   (health: ${KNATIVE_URL}/health)"
echo "=================================================================="
echo
echo "Sanity check both endpoints now:"
echo "  curl -s http://localhost:8080/health"
echo "  curl -s ${KNATIVE_URL}/health"
echo
echo "If the Knative curl hangs or fails to resolve, DNS for *.sslip.io may be"
echo "blocked on this network -- see real-infra/README.md 'Troubleshooting'."
