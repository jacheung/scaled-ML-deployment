#!/bin/bash -e
image_name=jacheung6/tutorial-training-container
image_tag=v0.1
full_image_name=${image_name}:${image_tag}
latest_image_name=${image_name}:latest

# cd "$(dirname "$0")" 
# echo "dirname : [$(dirname "$0")]"
docker build -t "${full_image_name}" -t "${latest_image_name}" -f pipelines/training/Dockerfile .
docker push "${image_name}" --all-tags

# Output the strict image name, which contains the sha256 image digest
docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"