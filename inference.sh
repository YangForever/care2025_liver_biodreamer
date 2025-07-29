# Prepare the environment
#!/bin/bash
source .venv/bin/activate

# Prepare data for inference
echo -n "Enter test data directory (strctured as released validation data), for example: care2025/LiQA_val:"
read input_dir
uv run scripts/inference_data_preparation.py --input_dir "$input_dir" 

# Run inference
nnUNetv2_predict -i output/renamed_data \
               -o output/inference \
               -d 17 \
               -c 3d_fullres \
               -f 2 \
               -step_size 0.5 \
               -p nnUNetPlans

uv run scripts/post_processing.py --inference_dir output/inference \
                                --output_dir output/post_processing

uv run scripts/submission_preparation.py --submission_dir output/CARE-Liver-Submission \
                                          --validation_dir output/post_processing 