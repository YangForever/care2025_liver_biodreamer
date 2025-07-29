# Prepare the environment
#!/bin/bash
source .venv/bin/activate

# Prepare data for inference
uv run scripts/inference_data_preparation.py --input_dir /hdd/yang/data/care2025/LiQA_val/Data

# Run inference
nnUNetv2_predict -i output/renamed_data \
               -o output/inference \
               -d 17 \
               -c 3d_fullres \
               -f 2 \
               -step_size 0.5 \
               -p nnUNetPlans

uv run scripts/submission_preparation.py --submission_dir output/CARE-Liver-Submission \
                                          --validation_dir output/inference