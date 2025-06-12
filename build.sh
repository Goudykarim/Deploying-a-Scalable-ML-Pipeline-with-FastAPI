#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# We no longer run the training script here.
# The server will use the pre-trained model that has been committed to the repo.