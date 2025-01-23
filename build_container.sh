#!/bin/bash

set -eou pipefail

cd $(git rev-parse --show-toplevel)

GITHUB_REPOSITORY_OWNER=${GITHUB_REPOSITORY_OWNER:-"ghcr.io/taranis-ai"}
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD | sed 's/[^a-zA-Z0-9_.-]/_/g')

echo "Building containers for branch ${CURRENT_BRANCH} on ${GITHUB_REPOSITORY_OWNER}"

docker buildx build --file Containerfile \
  --build-arg GITHUB_REPOSITORY_OWNER="${GITHUB_REPOSITORY_OWNER}" \
  --tag "${GITHUB_REPOSITORY_OWNER}/taranis-story-clustering-bot:latest" \
  --tag "${GITHUB_REPOSITORY_OWNER}/taranis-story-clustering-bot:${CURRENT_BRANCH}" \
  --load .

