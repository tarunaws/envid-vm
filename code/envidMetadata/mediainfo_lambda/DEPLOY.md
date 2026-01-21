# MediaInfo Lambda (us-east-1) — Deploy

Goal: run `mediainfo` in an AWS Linux runtime so the backend is OS-independent.

## 1) Prereqs
- AWS CLI configured (`aws sts get-caller-identity` works)
- Docker installed
- An ECR repo name (example: `envid-mediainfo-lambda`)

Quick path (recommended): run the included deploy script.

```bash
cd code/envidMetadata/mediainfo_lambda
./deploy.sh
```

If you don't have Docker locally, use CodeBuild (Docker-less on your machine):

```bash
cd code/envidMetadata/mediainfo_lambda
./deploy_codebuild.sh
```

## 2) Create ECR repo (us-east-1)
```bash
aws ecr create-repository --region us-east-1 --repository-name envid-mediainfo-lambda
```

## 3) Build + push container image
From `code/envidMetadata/mediainfo_lambda/`:
```bash
export AWS_REGION=us-east-1
export REPO_NAME=envid-mediainfo-lambda
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME"

aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker build -t "$REPO_NAME:latest" .
docker tag "$REPO_NAME:latest" "$ECR_URI:latest"
docker push "$ECR_URI:latest"
```

## 4) Create the Lambda function (container image)
```bash
aws lambda create-function \
  --region us-east-1 \
  --function-name envid-mediainfo \
  --package-type Image \
  --code ImageUri="$ECR_URI:latest" \
  --role "arn:aws:iam::<ACCOUNT_ID>:role/<LAMBDA_EXEC_ROLE>" \
  --timeout 60 \
  --memory-size 2048
```

Optional: increase ephemeral storage if needed (videos can be large):
```bash
aws lambda update-function-configuration \
  --region us-east-1 \
  --function-name envid-mediainfo \
  --ephemeral-storage Size=2048
```

Optional: size guard (bytes) for downloads to /tmp:
```bash
aws lambda update-function-configuration \
  --region us-east-1 \
  --function-name envid-mediainfo \
  --environment "Variables={MEDIAINFO_MAX_BYTES=524288000}"
```

## 5) IAM permissions

### Lambda execution role needs
- `s3:HeadObject`, `s3:GetObject` on the buckets/keys you’ll analyze

### Backend/service credentials need
- `lambda:InvokeFunction` on `envid-mediainfo` (or its ARN)

## 6) Backend configuration
Set this env var for the `envidMetadata` service:
- `ENVID_METADATA_MEDIAINFO_LAMBDA` = `envid-mediainfo` (or full ARN)
- `ENVID_METADATA_MEDIAINFO_LAMBDA_REGION` = `us-east-1`

Example in `code/.env.local`:
```env
ENVID_METADATA_MEDIAINFO_LAMBDA=envid-mediainfo
ENVID_METADATA_MEDIAINFO_LAMBDA_REGION=us-east-1
```

## 7) What the Lambda returns
The Lambda returns JSON like:
```json
{"ok": true, "mediainfo": {"available": true, "container_format": "MPEG-4", "duration_seconds": 30.1, "raw_full": {"media": {"track": []}}}}
```

The backend stores it under `technical_mediainfo` and exposes it in the categorized metadata under `technical_metadata.mediainfo`.
