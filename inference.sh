#!/bin/bash

# Parse command line arguments
INPUT_DIR=""
OUTPUT_DIR="/output"

# Function to display usage
usage() {
    echo "Usage: $0 -i <input_directory> [-o <output_directory>]"
    echo "  -i: Input data directory (structured as released validation data)"
    echo "  -o: Output directory (default: /output)"
    echo "Example: $0 -i /input -o /output"
    exit 0
}

# Parse command line options
while getopts "i:o:h" opt; do
    case $opt in
        i)
            INPUT_DIR="$OPTARG"
            ;;
        o)
            OUTPUT_DIR="$OPTARG"
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo ""
            echo "Usage: $0 -i <input_directory> [-o <output_directory>]"
            echo "  -i: Input data directory (structured as released validation data)"
            echo "  -o: Output directory (default: /output)"
            echo "Example: $0 -i /input -o /output"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$INPUT_DIR" ]; then
    echo "Error: Input directory is required"
    echo ""
    echo "Usage: $0 -i <input_directory> [-o <output_directory>]"
    echo "  -i: Input data directory (structured as released validation data)"
    echo "  -o: Output directory (default: /output)"
    echo "Example: $0 -i /input -o /output"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting inference pipeline..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Prepare data for inference
echo "Step 1: Preparing data for inference..."
uv run scripts/inference_data_preparation.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"

# Run inference using nnUNet (optimized for CPU)
echo "Step 2: Running nnUNet inference..."
nnUNetv2_predict -i "$OUTPUT_DIR/renamed_data" \
               -o "$OUTPUT_DIR/inference" \
               -d 17 \
               -c 3d_fullres \
               -f 2 \
               -step_size 0.8 \
               -p nnUNetPlans \
               -npp 2 \
               -nps 2

# Prepare submission folder (using inference results directly, skipping post-processing)
echo "Step 3: Preparing submission folder..."
uv run scripts/submission_preparation.py --submission_dir "$OUTPUT_DIR/CARE-Liver-Submission" \
                                          --validation_dir "$OUTPUT_DIR/inference"

# Run stage prediction
echo "Step 4: Running stage prediction..."
uv run scripts/stage_prediction.py --input_dir "$INPUT_DIR" \
                                    --mask_dir "$OUTPUT_DIR/CARE-Liver-Submission/LiSeg_pred" \
                                    --output_dir "$OUTPUT_DIR/CARE-Liver-Submission"

echo "Inference pipeline completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
