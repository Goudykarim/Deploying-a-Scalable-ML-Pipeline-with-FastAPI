    #!/usr/bin/env bash
    # exit on error
    set -o errexit

    pip install -r requirements.txt

    # Run the training pipeline to generate the model artifacts
    # Render has a small instance size, so we need to be mindful of memory.
    # If this step fails on Render, you may need to pre-train the model
    # and upload the .pkl files directly to your repo.
    python train_model.py