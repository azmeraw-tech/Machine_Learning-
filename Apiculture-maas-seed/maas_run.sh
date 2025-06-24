#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0)  Sanity-check: make sure we can talk to AWS CLI first
###############################################################################
command -v aws >/dev/null || {
  echo "aws CLI not found. Install AWS CLI v2 first: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
  exit 1
}

###############################################################################
# 1)  Ensure jq exists, install if missing
###############################################################################
if ! command -v jq >/dev/null 2>&1; then
  echo "🔍 jq not found — attempting to install..."

  if command -v apt-get >/dev/null 2>&1; then                # Debian/Ubuntu
    sudo apt-get update -qq
    sudo apt-get install -y -qq jq                           # :contentReference[oaicite:0]{index=0}
  elif command -v yum >/dev/null 2>&1; then                  # RHEL/CentOS 7
    sudo yum install -y epel-release >/dev/null              # enable EPEL repo :contentReference[oaicite:1]{index=1}
    sudo yum install -y jq >/dev/null
  elif command -v dnf >/dev/null 2>&1; then                  # RHEL 8+/Fedora
    sudo dnf install -y jq >/dev/null
  elif command -v apk >/dev/null 2>&1; then                  # Alpine
    sudo apk add --no-cache jq >/dev/null
  elif command -v brew >/dev/null 2>&1; then                 # macOS / Linuxbrew
    brew install jq                                          # :contentReference[oaicite:2]{index=2}
  else
    echo "Unsupported package manager. Install jq manually: https://jqlang.org/download/" # :contentReference[oaicite:3]{index=3}
    exit 1
  fi

  command -v jq >/dev/null || { echo "jq install failed"; exit 1; }
  echo "jq installed successfully."
fi

###############################################################################
# 2)  Pull AWS creds from Secrets Manager
###############################################################################
SECRET_JSON=$(aws secretsmanager get-secret-value \
  --secret-id maas/maas-seed/s3-cred \
  --query SecretString --output text)

export AWS_ACCESS_KEY_ID=$(echo "$SECRET_JSON" | jq -r .AWS_ACCESS_KEY_ID)
export AWS_SECRET_ACCESS_KEY=$(echo "$SECRET_JSON" | jq -r .AWS_SECRET_ACCESS_KEY)

###############################################################################
# 3)  Build & launch your stack
###############################################################################
docker compose up -d --build