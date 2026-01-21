#!/usr/bin/env bash
set -euo pipefail

# Deploy MediaInfo Lambda without local Docker, using AWS CodeBuild.
# Requires: aws cli, zip

HERE="$(cd "$(dirname "$0")" && pwd)"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
FUNCTION_NAME="${MEDIAINFO_LAMBDA_FUNCTION_NAME:-envid-mediainfo}"
REPO_NAME="${MEDIAINFO_LAMBDA_ECR_REPO_NAME:-envid-mediainfo-lambda}"
PROJECT_NAME="${MEDIAINFO_LAMBDA_CODEBUILD_PROJECT:-envid-mediainfo-lambda-build}"
ROLE_NAME="${MEDIAINFO_LAMBDA_ROLE_NAME:-envid-mediainfo-lambda-role}"
CODEBUILD_ROLE_NAME="${MEDIAINFO_CODEBUILD_ROLE_NAME:-envid-mediainfo-codebuild-role}"
SOURCE_BUCKET="${MEDIAINFO_CODEBUILD_SOURCE_BUCKET:-${S3_BUCKET:-}}"
SOURCE_PREFIX="${MEDIAINFO_CODEBUILD_SOURCE_PREFIX:-envid-metadata/buildsrc}"
MAX_BYTES="${MEDIAINFO_MAX_BYTES:-524288000}"  # 500MB

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing $1" >&2; exit 1; }; }
need aws
need zip

if [[ -z "${SOURCE_BUCKET:-}" ]]; then
  echo "ERROR: S3_BUCKET not set; set MEDIAINFO_CODEBUILD_SOURCE_BUCKET or S3_BUCKET in .env.local" >&2
  exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text --region "$REGION")"
ECR_HOST="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
ECR_URI="$ECR_HOST/$REPO_NAME"
IMAGE_URI="$ECR_URI:latest"

SOURCE_KEY="$SOURCE_PREFIX/mediainfo_lambda_src.zip"
SOURCE_LOCATION="$SOURCE_BUCKET/$SOURCE_KEY"

echo "Region: $REGION"
echo "Account: $ACCOUNT_ID"
echo "ECR: $ECR_URI"
echo "Lambda: $FUNCTION_NAME"
echo "CodeBuild Project: $PROJECT_NAME"
echo "Source: s3://$SOURCE_BUCKET/$SOURCE_KEY"

# 1) Zip source (Dockerfile + handler + buildspec)
TMP_ZIP="/tmp/mediainfo_lambda_src.zip"
rm -f "$TMP_ZIP"
(
  cd "$HERE"
  zip -qr "$TMP_ZIP" Dockerfile app.py buildspec.yml
)

# 2) Upload source to S3
aws s3 cp "$TMP_ZIP" "s3://$SOURCE_BUCKET/$SOURCE_KEY" --region "$REGION" >/dev/null

# 3) Ensure ECR repo exists
aws ecr describe-repositories --region "$REGION" --repository-names "$REPO_NAME" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$REGION" --repository-name "$REPO_NAME" >/dev/null

# 4) Create/update CodeBuild role
# Some accounts require the regional service principal too.
TRUST_CB='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":["codebuild.amazonaws.com","codebuild.us-east-1.amazonaws.com"]},"Action":"sts:AssumeRole"}]}'
if ! aws iam get-role --role-name "$CODEBUILD_ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$CODEBUILD_ROLE_NAME" --assume-role-policy-document "$TRUST_CB" >/dev/null
else
  aws iam update-assume-role-policy --role-name "$CODEBUILD_ROLE_NAME" --policy-document "$TRUST_CB" >/dev/null
fi
CODEBUILD_ROLE_ARN="$(aws iam get-role --role-name "$CODEBUILD_ROLE_NAME" --query Role.Arn --output text)"

# Inline policy for CodeBuild: logs + s3 read of source zip + ECR push
cat > /tmp/mediainfo_codebuild_policy.json <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:GetObjectVersion"
      ],
      "Resource": "arn:aws:s3:::$SOURCE_BUCKET/$SOURCE_KEY"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:CompleteLayerUpload",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
        "ecr:DescribeRepositories",
        "ecr:CreateRepository"
      ],
      "Resource": "arn:aws:ecr:$REGION:$ACCOUNT_ID:repository/$REPO_NAME"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sts:GetCallerIdentity"
      ],
      "Resource": "*"
    }
  ]
}
JSON
aws iam put-role-policy \
  --role-name "$CODEBUILD_ROLE_NAME" \
  --policy-name "envid-mediainfo-codebuild-inline" \
  --policy-document file:///tmp/mediainfo_codebuild_policy.json >/dev/null

# 5) Create/update CodeBuild project
# Use privilegedMode so Docker builds work inside CodeBuild.
if aws codebuild batch-get-projects --region "$REGION" --names "$PROJECT_NAME" --query 'projects[0].name' --output text 2>/dev/null | grep -q "$PROJECT_NAME"; then
  aws codebuild update-project --region "$REGION" \
    --name "$PROJECT_NAME" \
    --service-role "$CODEBUILD_ROLE_ARN" \
    --source "type=S3,location=$SOURCE_LOCATION" \
    --environment "type=LINUX_CONTAINER,image=aws/codebuild/standard:7.0,computeType=BUILD_GENERAL1_MEDIUM,privilegedMode=true,environmentVariables=[{name=AWS_REGION,value=$REGION,type=PLAINTEXT},{name=REPO_NAME,value=$REPO_NAME,type=PLAINTEXT}]" \
    --artifacts "type=NO_ARTIFACTS" >/dev/null
else
  aws codebuild create-project --region "$REGION" \
    --name "$PROJECT_NAME" \
    --service-role "$CODEBUILD_ROLE_ARN" \
    --source "type=S3,location=$SOURCE_LOCATION" \
    --environment "type=LINUX_CONTAINER,image=aws/codebuild/standard:7.0,computeType=BUILD_GENERAL1_MEDIUM,privilegedMode=true,environmentVariables=[{name=AWS_REGION,value=$REGION,type=PLAINTEXT},{name=REPO_NAME,value=$REPO_NAME,type=PLAINTEXT}]" \
    --artifacts "type=NO_ARTIFACTS" >/dev/null
fi

# 6) Start build and wait
BUILD_ID="$(aws codebuild start-build --region "$REGION" --project-name "$PROJECT_NAME" --query build.id --output text)"
echo "Started build: $BUILD_ID"

STATUS="IN_PROGRESS"
while [[ "$STATUS" == "IN_PROGRESS" || "$STATUS" == "QUEUED" ]]; do
  sleep 10
  STATUS="$(aws codebuild batch-get-builds --region "$REGION" --ids "$BUILD_ID" --query 'builds[0].buildStatus' --output text)"
  echo "Build status: $STATUS"
done

if [[ "$STATUS" != "SUCCEEDED" ]]; then
  echo "ERROR: CodeBuild failed ($STATUS). Check logs in CodeBuild console." >&2
  exit 1
fi

echo "OK: Image pushed: $IMAGE_URI"

# 7) Ensure Lambda execution role exists and has S3 read
TRUST_LAMBDA='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST_LAMBDA" >/dev/null
  aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null
fi
ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)"

# Lambda S3 read policy (use S3_BUCKET)
cat > /tmp/mediainfo_lambda_s3_policy.json <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:HeadObject","s3:GetObject"],
      "Resource": [
        "arn:aws:s3:::$SOURCE_BUCKET/*",
        "arn:aws:s3:::${ENVID_METADATA_VIDEO_BUCKET:-$SOURCE_BUCKET}/*"
      ]
    }
  ]
}
JSON
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name "envid-mediainfo-s3-read" --policy-document file:///tmp/mediainfo_lambda_s3_policy.json >/dev/null

# 8) Create or update Lambda
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

echo "OK: Lambda deployed: $FUNCTION_NAME"
echo "Backend env: ENVID_METADATA_MEDIAINFO_LAMBDA=$FUNCTION_NAME"
