PROJECT_ID=${PROJECT_ID:-"civic-pulsar-459907-t2"}
ACCELERATOR_TYPE=${ACCELERATOR_TYPE:-"v4-8"} # v4-*, v3-*, v2-* available
ZONE=${ZONE:-"us-central2-b"}    # us-central2-b for v4, europe-west4-a for v3, us-central1-f for v2
RUNTIME_VERSION=${RUNTIME_VERSION:-"tpu-ubuntu2204-base"}
TPU_NAME=${TPU_NAME:-"my_tpu"}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-"trcloud@civic-pulsar-459907-t2.iam.gserviceaccount.com"}
NETWORK=${NETWORK:-"trc-vpc"}
SUB_NETWORK=${SUB_NETWORK:="trc-us-central2-subnet"}


# create a TPU VM
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION \
    --network $NETWORK \
    --subnetwork $SUB_NETWORK \
    --service-account=$SERVICE_ACCOUNT \
    --metadata startup-script="#! /bin/bash
        pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'"
# `--preemptible` if use preemptible TPUs


# connect to TPU VM with ssh
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE
    
