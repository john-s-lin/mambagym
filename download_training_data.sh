#!/bin/bash
# Check if the target directory is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <target-directory>"
  exit 1
fi

TARGET_DIR=$1

# Use the specified target directory in the curl command
curl -L -o "$TARGET_DIR/archive.zip" \
https://www.kaggle.com/api/v1/datasets/download/andrewmvd/ct-low-dose-reconstruction