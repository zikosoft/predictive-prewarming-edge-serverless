#!/usr/bin/env bash
# Installs Knative Serving v1.23.0 + Kourier networking + magic-DNS
# default-domain onto whatever Kubernetes cluster `kubectl` currently
# points at (intended target: Docker Desktop's built-in Kubernetes).
#
# Run this from WSL2 (Ubuntu) or Git Bash, with Docker Desktop's
# "Enable Kubernetes" turned on first (Settings -> Kubernetes) and
# kubectl context set to docker-desktop (`kubectl config use-context
# docker-desktop`). See real-infra/README.md for the full walkthrough.
set -euo pipefail

KNATIVE_VERSION="knative-v1.23.0"

echo "[*] kubectl context: $(kubectl config current-context)"
read -p "    Continue installing Knative onto this context? [y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Aborted."; exit 1; }

echo "[*] Installing Knative Serving CRDs..."
kubectl apply -f "https://github.com/knative/serving/releases/download/${KNATIVE_VERSION}/serving-crds.yaml"

echo "[*] Installing Knative Serving core components..."
kubectl apply -f "https://github.com/knative/serving/releases/download/${KNATIVE_VERSION}/serving-core.yaml"

echo "[*] Installing Kourier networking layer..."
kubectl apply -f "https://github.com/knative-extensions/net-kourier/releases/download/${KNATIVE_VERSION}/kourier.yaml"

echo "[*] Configuring Knative to use Kourier as its ingress..."
kubectl patch configmap/config-network \
  --namespace knative-serving \
  --type merge \
  --patch '{"data":{"ingress-class":"kourier.ingress.networking.knative.dev"}}'

echo "[*] Installing magic-DNS (sslip.io) default domain..."
kubectl apply -f "https://github.com/knative/serving/releases/download/${KNATIVE_VERSION}/serving-default-domain.yaml"

echo "[*] Patching config-autoscaler to a 6s stable-window / grace-period..."
kubectl patch configmap/config-autoscaler \
  --namespace knative-serving \
  --type merge \
  --patch-file "$(dirname "$0")/../k8s/autoscaler-configmap-patch.yaml"

echo "[*] Waiting for Knative Serving deployments to become ready (this can take a few minutes)..."
kubectl wait --for=condition=Available deployment --all -n knative-serving --timeout=300s
kubectl wait --for=condition=Available deployment --all -n kourier-system --timeout=300s

echo "[*] Waiting for the Kourier LoadBalancer to get an external IP (Docker Desktop maps this to localhost)..."
for i in $(seq 1 30); do
  ip=$(kubectl get svc kourier -n kourier-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
  [[ -n "$ip" ]] && break
  sleep 2
done
echo "[*] Kourier external IP: ${ip:-<not yet assigned, check `kubectl get svc kourier -n kourier-system`>}"

echo
echo "Done. Verify with:"
echo "  kubectl get pods -n knative-serving"
echo "  kubectl get pods -n kourier-system"
echo "  curl -s -o /dev/null -w '%{http_code}\n' http://localhost/  (expect 404 from Kourier itself -- that's fine, it means it's up)"
