# care2025_liver_biodreamer

This project utilise [uv](https://docs.astral.sh/uv/) to manage the dependencies.

To install all the dependencies, run the commands:
````
cd care2025_liver_biodreamer/
uv sync
cd nnUNet/
uv pip install -e .
````

*The packages used are listed in the pyproject.toml which can used to create a environment by conda if required* 

# Segmentation Inference
Once the test data are organised as the release validation dataset:
````
LiQA_val/
    -Data/
        -VendorA/
            -xxxx/
            -yyyy/
            -zzzz/
````
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