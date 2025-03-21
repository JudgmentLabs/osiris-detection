#!/bin/bash

RESULTS_DIR="/path/to/your/results/directory"
echo "Processing raw predictions in $RESULTS_DIR..."
python format_predictions.py --input_dir "$RESULTS_DIR"

