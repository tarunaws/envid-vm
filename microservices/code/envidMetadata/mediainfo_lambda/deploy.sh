#!/usr/bin/env bash
set -euo pipefail

# Deploy MediaInfo Lambda (container image) to us-east-1.
# Requires: aws cli, docker

HERE="$(cd "$(dirname "$0")" && pwd)"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
FUNCTION_NAME="${MEDIAINFO_LAMBDA_FUNCTION_NAME:-envid-mediainfo}"
REPO_NAME="${MEDIAINFO_LAMBDA_ECR_REPO_NAME:-envid-mediainfo-lambda}"
ROLE_NAME="${MEDIAINFO_LAMBDA_ROLE_NAME:-envid-mediainfo-lambda-role}"
MAX_BYTES="${MEDIAINFO_MAX_BYTES:-524288000}"  # 500MB

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI not found. Install AWS CLI v2." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found. Install Docker Desktop (or run this in CI with Docker)." >&2
  exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text --region "$REGION")"
if [[ -z "$ACCOUNT_ID" ]]; then
  echo "ERROR: Could not determine AWS account id. Check credentials." >&2
  exit 1
fi

ECR_HOST="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
ECR_URI="$ECR_HOST/$REPO_NAME"
IMAGE_URI="$ECR_URI:latest"

echo "Region: $REGION"
echo "Account: $ACCOUNT_ID"
echo "ECR: $ECR_URI"
echo "Lambda: $FUNCTION_NAME"

# 1) Ensure ECR repo exists
aws ecr describe-repositories --region "$REGION" --repository-names "$REPO_NAME" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$REGION" --repository-name "$REPO_NAME" >/dev/null

# 2) Login to ECR
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_HOST" >/dev/null

# 3) Build + push
(
  cd "$HERE"
  docker build -t "$REPO_NAME:latest" .
  docker tag "$REPO_NAME:latest" "$IMAGE_URI"
  docker push "$IMAGE_URI"
)

# 4) Ensure Lambda execution role exists
# Trust policy for Lambda service
TRUST_JSON='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

ROLE_ARN=""
if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)"
else
  ROLE_ARN="$(aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST_JSON" --query Role.Arn --output text)"

  # Basic logging
  aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null
fi

# Inline policy for S3 read access.
# Uses S3_BUCKET/ENVID_METADATA_VIDEO_BUCKET/rawvideo bucket env vars if present.
BUCKET_MAIN="${S3_BUCKET:-${ENVID_METADATA_VIDEO_BUCKET:-}}"
BUCKET_RAWVIDEO="${ENVID_METADATA_RAWVIDEO_BUCKET:-${S3_BUCKET:-}}"

if [[ -z "$BUCKET_MAIN" ]]; then
  echo "WARN: S3_BUCKET not set; skipping S3 policy install (Lambda will fail to read)." >&2
else
  S3_POLICY_NAME="${MEDIAINFO_LAMBDA_S3_POLICY_NAME:-envid-mediainfo-s3-read}"
  # Allow read on both buckets (if they differ)
  BUCKETS_JSON="[\"$BUCKET_MAIN\""
  if [[ -n "$BUCKET_RAWVIDEO" && "$BUCKET_RAWVIDEO" != "$BUCKET_MAIN" ]]; then
    BUCKETS_JSON+=",\"$BUCKET_RAWVIDEO\""
  fi
  BUCKETS_JSON+="]"

  python3 - <<PY >/tmp/mediainfo_s3_policy.json
import json
buckets = json.loads('$BUCKETS_JSON')
resources = []
for b in buckets:
  resources.append(f"arn:aws:s3:::{b}")
  resources.append(f"arn:aws:s3:::{b}/*")
policy = {
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:HeadObject", "s3:GetObject"], "Resource": resources},
  ],
}
print(json.dumps(policy))
PY

  aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name "$S3_POLICY_NAME" --policy-document file:///tmp/mediainfo_s3_policy.json >/dev/null
fi

# 5) Create or update Lambda function
if aws lambda get-function --region "$REGION" --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
  aws lambda update-function-code --region "$REGION" --function-name "$FUNCTION_NAME" --image-uri "$IMAGE_URI" >/dev/null
else
  aws lambda create-function \
    --region "$REGION" \
    --function-name "$FUNCTION_NAME" \
    --package-type Image \
    --code ImageUri="$IMAGE_URI" \
    --role "$ROLE_ARN" \
    --timeout 60 \
    --memory-size 2048 >/dev/null
fi

aws lambda update-function-configuration \
  --region "$REGION" \
  --function-name "$FUNCTION_NAME" \
  --timeout 60 \
  --memory-size 2048 \
  --environment "Variables={MEDIAINFO_MAX_BYTES=$MAX_BYTES}" >/dev/null

echo "OK: Deployed $FUNCTION_NAME ($IMAGE_URI) in $REGION"
echo "Next: set ENVID_METADATA_MEDIAINFO_LAMBDA=$FUNCTION_NAME and restart envidMetadata."
