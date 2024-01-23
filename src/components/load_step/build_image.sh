#!/bin/bash -e
image_name=localhost:5000/mnist-load-step
image_tag=latest
full_image_name=${image_name}:${image_tag}

cd "$(dirname "$0")" 
echo "dirname : [$(dirname "$0")]"
docker build -t "${full_image_name}" . -f Dockerfile
docker push ${full_image_name}

# Output the strict image name, which contains the sha256 image digest
docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"