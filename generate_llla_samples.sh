#!/bin/bash

# Checkpoints for each dataset + method
CKPT_FLOW_MNIST="jac-zac/bayesflow-project/best-model:v204"
CKPT_DIFF_MNIST="jac-zac/bayesflow-project/best-model:v174"
CKPT_FLOW_FASHION="jac-zac/bayesflow-project/best-model:v127"
CKPT_DIFF_FASHION="jac-zac/bayesflow-project/best-model:v145"

METHODS=("flow" "diffusion")
DATASETS=("FashionMNIST" "MNIST")

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do

    # Select checkpoint
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

    # Set diffusion or flow step count
    if [ "$method" == "flow" ]; then
      if [ "$dataset" == "MNIST" ]; then
        STEPS=10
      else
        STEPS=15
      fi
    else
      STEPS=50
    fi

    # Set slice indices based on dataset and method
    if [ "$dataset" == "FashionMNIST" ]; then
      SLICE_START=1
      SLICE_END=2
    else
      SLICE_START=0
      SLICE_END=2
    fi

    # Output directory
    SAVE_DIR="plots/$(echo "$dataset" | tr '[:upper:]' '[:lower:]')_${method}"

    echo "Running $dataset with $method, steps=$STEPS, slice=[$SLICE_START:$SLICE_END]"

    python -m src.eval.llla \
      --ckpt "$CKPT" \
      --dataset-name "$dataset" \
      --method "$method" \
      --save-dir "$SAVE_DIR" \
      --batch-size 16 \
      --steps "$STEPS" \
      --cov-samples 100 \
      --slice-start "$SLICE_START" \
      --slice-end "$SLICE_END"

  done
done
