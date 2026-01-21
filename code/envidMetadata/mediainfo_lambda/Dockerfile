# MediaInfo Lambda (container image)
# Build/push to ECR and create Lambda in us-east-1.
# Base image includes the Lambda Runtime Interface Client.
FROM public.ecr.aws/lambda/python:3.11

# Install mediainfo on Amazon Linux.
# Base image is AL2-compatible; yum is available.
RUN yum -y update \
  && yum -y install mediainfo \
  && yum clean all \
  && rm -rf /var/cache/yum

# Copy handler
COPY app.py ${LAMBDA_TASK_ROOT}/app.py

# Set the handler
CMD ["app.handler"]
