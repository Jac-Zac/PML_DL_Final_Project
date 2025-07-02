#!/bin/bash

# Set your common config
# NOTE: Train those two
# CKPT_FLOW_MNIST="jac-zac/bayesflow-project/best-model:v145"
# CKPT_DIFF_MNIST="jac-zac/bayesflow-project/best-model:v145"

CKPT_FLOW_FASHION="jac-zac/bayesflow-project/best-model:v127"
CKPT_DIFF_FASHION="jac-zac/bayesflow-project/best-model:v145"

METHODS=("flow" "diffusion")
# DATASETS=("MNIST" "FashionMNIST")
DATASETS=("FashionMNIST")

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do

    # Choose checkpoint based on dataset and method
    if [ "$dataset" == "MNIST" ]; then
      if [ "$method" == "flow" ]; then
        CKPT=$CKPT_FLOW_MNIST
      else
        CKPT=$CKPT_DIFF_MNIST
      fi
    else
      if [ "$method" == "flow" ]; then
        CKPT=$CKPT_FLOW_FASHION
      else
        CKPT=$CKPT_DIFF_FASHION
      fi
    fi

    # Set the number of steps based on method
    if [ "$method" == "flow" ]; then
      STEPS=15
    else
      STEPS=50
    fi

    # Set the output directory
    SAVE_DIR="plots/$(echo "${dataset}" | tr '[:upper:]' '[:lower:]')_${method}"

    echo "Generating plots for $dataset with $method (steps=$STEPS)..."
    python -m src.eval.llla \
      --ckpt "$CKPT" \
      --dataset-name "$dataset" \
      --method "$method" \
      --save-dir "$SAVE_DIR" \
      --batch-size 16 \
      --steps "$STEPS" \
      --cov-samples 100
  done
done
