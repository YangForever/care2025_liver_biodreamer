# care2025_liver_biodreamer

This project utilise [uv](https://docs.astral.sh/uv/) to manage the dependencies.

## Local Installation

To install all the dependencies, run the commands:
````
cd care2025_liver_biodreamer/
uv sync
cd nnUNet/
uv pip install -e .
````

*The packages used are listed in the pyproject.toml which can used to create a environment by conda if required* 

## Docker Usage (Recommended for Inference)

For easy deployment and inference, you can use Docker to run the complete pipeline without local setup.

### Building the Docker Image

```bash
# Build the Docker image
docker build -t liver-inference .
```

### Running Inference with Docker

```bash
# Run inference using Docker
docker run \
  -v /path/to/your/test/dataset:/input:ro \
  -v /path/to/output/directory:/output \
  liver-inference -i /input -o /output
```

**Example:**
```bash
# If your test data is in ./LiQA_val and you want output in ./results
docker run \
  -v $(pwd)/LiQA_val:/input:ro \
  -v $(pwd)/results:/output \
  liver-inference -i /input -o /output
```

### Creating a Docker Image Archive

To create a distributable Docker image file:

```bash
# Save the Docker image to a tar.gz file
docker save liver-inference | gzip > liver_inference_image.tar.gz

# Load the image on another machine
docker load -i liver_inference_image.tar.gz

# Run the loaded image
docker run \
  -v /path/to/Liver-Test-Dataset:/input:ro \
  -v /path/to/output_dir:/output \
  liver-inference -i /input -o /output
```

### Docker Command Options

- `-i <input_directory>`: Required. Path to input data directory (structured as released validation data)
- `-o <output_directory>`: Optional. Path to output directory (default: /output)
- `-h`: Show help message

**Input Data Structure:**
The input directory should be structured like the released validation dataset:
```
/input/
    Data/
        VendorA/
            case001/
            case002/
            ...
        VendorB/
            ...
```

**Output Structure:**
The Docker container will create the following output structure:
```
/output/
    renamed_data/           # Preprocessed input data
    inference/              # Raw nnUNet predictions
    post_processing/        # Post-processed segmentations
    CARE-Liver-Submission/  # Final submission files
        LiSeg_pred/         # Segmentation predictions
        LiFS_pred_contrast.csv      # Contrast fibrosis predictions
        LiFS_pred_non_contrast.csv  # Non-contrast fibrosis predictions
```

# Local Development Usage

## Segmentation Inference
Once the test data are organised as the release validation dataset:
````
LiQA_val/
    -Data/
        -VendorA/
            -xxxx/
            -yyyy/
            -zzzz/
````

### Using the Updated Inference Script
```bash
# Run inference with command-line arguments (no interactive input)
./inference.sh -i /path/to/LiQA_val -o /path/to/output

# Or for local development
./inference.sh -i ./LiQA_val -o ./output
```

### Legacy Usage (Interactive)
Replace the *--input_dir* to the *LiQA_val* path, then run
````
./inference.sh
````

# nnUNet Usage
please refer to the original documentation for more details 
## nnUNet data preprocessed
After running the code and obtain the nnUNet_raw data

run the following code
````
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
````
Example:
```Console
nnUNetv2_plan_and_preprocess -d 13 --verify_dataset_integrity
```

## nnUNet training
```Console
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```
*DATASET_NAME_OR_ID: dataset id without '00'* \
*UNET_CONFIGURATION: 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres*\
*FOLD: 0, 1, 2, 3, 4 (default 5-fold)*

Example:
```Console
nnUNetv2_train 13 3d_fullres 0
```

## nnUNet inference
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION -f FOLD --save_probabilities
```

Example:
```Console
nnUNetv2_predict -i ./Dataset013_care2025_benchmark1/imagesTs -o ./nnUNet_results/Dataset013_care2025_benchmark1/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/inference_test -d 13 -c 3d_fullres -f 0
```

# Fibrosis Classfication
Using the jupyter notebook
```Console
script/classification.ipynb
```