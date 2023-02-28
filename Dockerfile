FROM python:3.9-slim

RUN apt-get update && apt-get install build-essential gcc -y

WORKDIR /root/app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY sample_data sample_data
COPY src src
COPY main.py main.py

ARG gcs_bucket_name="tfx-recommender-artifacts"
ARG google_cloud_project="tfx-recommender-playground"
ARG gcp_artifact_registry_docker_tag="master"

ENV GCS_BUCKET_NAME=$gcs_bucket_name
ENV GOOGLE_CLOUD_PROJECT=$google_cloud_project
ENV GCP_ARTIFACT_REGISTRY_DOCKER_TAG=$gcp_artifact_registry_docker_tag

CMD ["--runner-env", "local", "--use-local-sample-data"]
ENTRYPOINT ["python", "main.py"]
