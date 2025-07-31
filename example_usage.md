# Docker Usage Examples for CARE2025 Liver Inference

This document provides detailed examples of how to use the Docker-based inference pipeline for liver segmentation and fibrosis stage prediction.

## Quick Start

### 1. Build the Docker Image

```bash
# Option 1: Use the automated build script (recommended)
./build_docker.sh

# Option 2: Manual build
docker build -t liver-inference .
```

### 2. Prepare Your Data

Ensure your test data follows the expected structure:
```
your_test_data/
    Data/
        VendorA/
            case001/
                DWI_800.nii.gz
                GED1.nii.gz
                GED2.nii.gz
                GED3.nii.gz
                GED4.nii.gz
                T1.nii.gz
                T2.nii.gz
            case002/
                ...
        VendorB/
            case003/
                ...
```

### 3. Run Inference

```bash
# Basic usage
docker run \
  -v /path/to/your_test_data:/input:ro \
  -v /path/to/output_directory:/output \
  liver-inference -i /input -o /output
```

## Detailed Examples

### Example 1: Local Data Directory

If your test data is in the current directory:

```bash
# Create output directory
mkdir -p ./results

# Run inference
docker run \
  -v $(pwd)/LiQA_val:/input:ro \
  -v $(pwd)/results:/output \
  liver-inference -i /input -o /output
```

### Example 2: Absolute Paths

```bash
docker run \
  -v /home/user/datasets/liver_test:/input:ro \
  -v /home/user/outputs/liver_results:/output \
  liver-inference -i /input -o /output
```

### Example 3: Interactive Mode for Debugging

```bash
# Run container interactively to debug issues
docker run -it \
  -v $(pwd)/test_data:/input:ro \
  -v $(pwd)/results:/output \
  liver-inference /bin/bash

# Inside the container, you can run:
# ./inference.sh -i /input -o /output
# Or run individual scripts for debugging
```

### Example 4: Using Saved Docker Image

```bash
# Load a previously saved image
docker load -i liver_inference_image.tar.gz

# Run the loaded image
docker run \
  -v /path/to/test_data:/input:ro \
  -v /path/to/results:/output \
  liver-inference -i /input -o /output
```

## Output Structure

After successful execution, your output directory will contain:

```
output_directory/
├── renamed_data/                    # Preprocessed input data for nnUNet
│   ├── case001_0000.nii.gz        # DWI modality
│   ├── case001_0001.nii.gz        # GED1 modality
│   ├── case001_0002.nii.gz        # GED2 modality
│   ├── case001_0003.nii.gz        # GED3 modality
│   ├── case001_0004.nii.gz        # GED4 modality
│   ├── case001_0005.nii.gz        # T1 modality
│   └── case001_0006.nii.gz        # T2 modality
├── inference/                       # Raw nnUNet segmentation results
│   ├── case001.nii.gz
│   ├── case002.nii.gz
│   └── ...
├── post_processing/                 # Post-processed segmentations
│   ├── case001_DWI_processed.nii.gz
│   ├── case001_GED1_processed.nii.gz
│   ├── case001_GED2_processed.nii.gz
│   ├── case001_GED3_processed.nii.gz
│   ├── case001_GED4_processed.nii.gz
│   ├── case001_T1_processed.nii.gz
│   ├── case001_T2_processed.nii.gz
│   └── ...
└── CARE-Liver-Submission/           # Final submission files
    ├── LiSeg_pred/                  # Segmentation predictions
    │   ├── case001/
    │   │   ├── DWI_800_pred.nii.gz
    │   │   ├── GED1_pred.nii.gz
    │   │   ├── GED2_pred.nii.gz
    │   │   ├── GED3_pred.nii.gz
    │   │   ├── GED4_pred.nii.gz
    │   │   ├── T1_pred.nii.gz
    │   │   └── T2_pred.nii.gz
    │   └── case002/
    │       └── ...
    ├── LiFS_pred_contrast.csv       # Fibrosis prediction (contrast setting)
    └── LiFS_pred_non_contrast.csv   # Fibrosis prediction (non-contrast setting)
```

## Troubleshooting

### Common Issues and Solutions

1. **Permission Denied Errors**
   ```bash
   # Ensure output directory has proper permissions
   sudo chown -R $USER:$USER /path/to/output_directory
   
   # Or run with user mapping
   docker run --user $(id -u):$(id -g) \
     -v $(pwd)/test_data:/input:ro \
     -v $(pwd)/results:/output \
     liver-inference -i /input -o /output
   ```

2. **Input Directory Not Found**
   ```bash
   # Check if your input path is correct
   ls -la /path/to/your/test/data
   
   # Ensure the path structure is correct
   docker run liver-inference -h  # Show help
   ```

3. **Out of Memory Errors**
   ```bash
   # Increase Docker memory limit or use smaller batch sizes
   docker run --memory=8g \
     -v $(pwd)/test_data:/input:ro \
     -v $(pwd)/results:/output \
     liver-inference -i /input -o /output
   ```

4. **CUDA/GPU Issues**
   ```bash
   # For GPU support (if available)
   docker run --gpus all \
     -v $(pwd)/test_data:/input:ro \
     -v $(pwd)/results:/output \
     liver-inference -i /input -o /output
   ```

### Viewing Logs

```bash
# Run with verbose output
docker run \
  -v $(pwd)/test_data:/input:ro \
  -v $(pwd)/results:/output \
  liver-inference -i /input -o /output 2>&1 | tee inference.log
```

### Container Inspection

```bash
# List running containers
docker ps

# View container logs
docker logs <container_id>

# Execute commands in running container
docker exec -it <container_id> /bin/bash
```

## Performance Tips

1. **Use SSD storage** for input and output directories
2. **Allocate sufficient memory** (recommend 8GB+)
3. **Use local directories** instead of network shares when possible
4. **Clean up old containers** regularly: `docker system prune`

## Integration with CI/CD

Example for automated testing:

```bash
#!/bin/bash
# test_pipeline.sh

set -e

# Build image
docker build -t liver-inference .

# Test with sample data
docker run --rm \
  -v $(pwd)/test_samples:/input:ro \
  -v $(pwd)/test_output:/output \
  liver-inference -i /input -o /output

# Verify outputs exist
if [ -f "test_output/CARE-Liver-Submission/LiFS_pred_contrast.csv" ]; then
    echo "✅ Pipeline test passed"
else
    echo "❌ Pipeline test failed"
    exit 1
fi
``` 