#!/bin/bash
# QuNtra AWS EC2 deployment — ap-south-1 (Mumbai), lowest latency to NSE.
# Instance: t3.medium (2 vCPU, 4 GB) · Ubuntu 22.04 LTS
#
# Prereqs on your Mac:
#   brew install awscli && aws configure   (region: ap-south-1)
# Then:
#   bash infra/deploy_aws.sh
#
# NOTE: do NOT migrate mid-gate. Scripts are ready; the move happens
# after the 40-day paper gate completes (see RUNBOOK.md).

set -euo pipefail
cd "$(dirname "$0")/.."

REGION="ap-south-1"
INSTANCE_TYPE="t3.medium"
KEY_NAME="quntra-key"
SECURITY_GROUP="quntra-sg"
INSTANCE_NAME="quntra-paper-trading"

echo "=== QuNtra AWS Deployment ==="

# 0. Resolve the latest Ubuntu 22.04 LTS AMI dynamically (never hardcode:
#    AMI ids rotate and differ per account)
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
              "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' \
    --output text)
echo "Ubuntu 22.04 AMI: $AMI_ID"

# 1. Key pair
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" \
        >/dev/null 2>&1; then
    echo "Creating key pair…"
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --region "$REGION" \
        --query 'KeyMaterial' \
        --output text > ~/.ssh/quntra-key.pem
    chmod 400 ~/.ssh/quntra-key.pem
    echo "Key saved to ~/.ssh/quntra-key.pem"
fi

# 2. Security group — SSH from the current IP only
SG_ID=$(aws ec2 create-security-group \
    --group-name "$SECURITY_GROUP" \
    --description "QuNtra trading system" \
    --region "$REGION" \
    --query 'GroupId' \
    --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
        --group-names "$SECURITY_GROUP" \
        --region "$REGION" \
        --query 'SecurityGroups[0].GroupId' \
        --output text)

MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 22 \
    --cidr "${MY_IP}/32" \
    --region "$REGION" 2>/dev/null || true
echo "Security group: $SG_ID (SSH restricted to $MY_IP)"

# 3. Launch
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --region "$REGION" \
    --tag-specifications \
        "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --user-data file://infra/ec2_userdata.sh \
    --query 'Instances[0].InstanceId' \
    --output text)
echo "Instance launched: $INSTANCE_ID"

echo "Waiting for running state…"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

# 4. Elastic IP — survives instance stop/start
ALLOC_ID=$(aws ec2 allocate-address \
    --domain vpc \
    --region "$REGION" \
    --query 'AllocationId' \
    --output text)
aws ec2 associate-address \
    --instance-id "$INSTANCE_ID" \
    --allocation-id "$ALLOC_ID" \
    --region "$REGION" >/dev/null
PUBLIC_IP=$(aws ec2 describe-addresses \
    --allocation-ids "$ALLOC_ID" \
    --region "$REGION" \
    --query 'Addresses[0].PublicIp' \
    --output text)

{
    echo "ELASTIC_IP=$PUBLIC_IP"
    echo "INSTANCE_ID=$INSTANCE_ID"
    echo "ALLOCATION_ID=$ALLOC_ID"
    echo "SG_ID=$SG_ID"
} > infra/.ec2_info

echo ""
echo "=== DEPLOYMENT COMPLETE ==="
echo "Elastic IP: $PUBLIC_IP"
echo "SSH:  ssh -i ~/.ssh/quntra-key.pem ubuntu@$PUBLIC_IP"
echo "Next: bash infra/sync_to_ec2.sh"
