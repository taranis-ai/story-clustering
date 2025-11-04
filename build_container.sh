#!/bin/bash

set -eou pipefail

if [ ! -d .git ]; then
    echo "This script must be run from the root of a git repository"
    exit 1
fi

cd $(git rev-parse --show-toplevel)

GITHUB_REPOSITORY_OWNER=${GITHUB_REPOSITORY_OWNER:-"ghcr.io/taranis-ai"}
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD | sed 's/[^a-zA-Z0-9_.-]/_/g')

MODEL=${MODEL:-"tfidf"}
REPO_NAME=$(grep 'url =' .git/config | sed -E 's/.*[:\/]([^\/]+)\.git/\1/')

echo "Building containers for branch ${CURRENT_BRANCH} with model ${MODEL} on ${GITHUB_REPOSITORY_OWNER}"

docker buildx build --file Containerfile \
  --build-arg GITHUB_REPOSITORY_OWNER="${GITHUB_REPOSITORY_OWNER}" \
  --build-arg MODEL="${MODEL}" \
  --tag "${GITHUB_REPOSITORY_OWNER}/${REPO_NAME}:latest" \
  --tag "${GITHUB_REPOSITORY_OWNER}/${REPO_NAME}:${MODEL}" \
  --load .
