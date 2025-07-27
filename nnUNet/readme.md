# nnUNet for glomeruli segmentation using HiP-CT
The repository is a fork from nnUNet. The original README can be accessed [here](https://github.com/MIC-DKFZ/nnUNet)

# Installation
This repository has been updated with several customised functions. Therefore, a reinstallation or update of nnUNet is required to compile the customised functions.

### 1. Path setup
Setup your path so the nnUNet finds the data and training plans
- Open the files in the path: *nnUNet/nnunetv2/path.py*
- Modify the fowllowing lines to your customised folders
```
data_dir = ''
result_dir = ''
nnUNet_raw = os.path.join(data_dir, 'nnUNet_raw') 
nUNet_preprocessed = os.path.join(data_dir, 'nnUNet_preprocessed') 
nnUNet_results = os.path.join(result_dir, 'nnUNet_results')
```

This setup works for any systems e.g. Windows, Linux. In Linux, you may want to set the path in the ./bashrc, please refer to the original setup [link](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md)

### 2. Install the nnUNet
Installation is recommended in a conda environment or a venv environment. (example below in conda)

If you haven't got a nnunet environment on conda: (higher python version is also working)

```
conda create -n nnunet python=3.9
```

Then (if you have a nnunet environment):

```
conda activate nnunet
```

Pytorch is recommended to be installed before nnUNet (version 2.0.1 is recommended and tested without error)
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Install (or update) the repository:

```
cd nnUNet
pip install -e .
```

# Training
Training details are same as in the original nnUNet guideline, for more details: [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)

### 1. Pre-process the dataset
Before training, extract dataset fingerprint(im size, voxel spacings, intensity information etc.) to build network structure
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
e.g.
```
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

### 2. Training
Train the model:
 ```
 nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
 ```
 e.g.
 ```
 nnUNetv2_train 1 3d_fullres 0 
 ```

# Correlative training on lower resolution
This is the key fine-tunining step proposed in the multiscale correlative glomeruli segmentation. The model which is trained on the higher resolution will be fine-tuned with lower resolution data.

### 1. Prepare the dataset of lower resolution
Before training, extract dataset fingerprint(im size, voxel spacings, intensity information etc.) to build network structure
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
*DATASET_ID: dataset id without '00'*

e.g. (dataset 2 is a lower resolution dataset compared to dataset 1)
```
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
```
### 2. Move the training plan from high resolution dataset to low resolution dataset
```
nnUNetv2_move_plans_between_datasets -s HIGH_RES_DATASET -t LOW_RES_DATASET -sp HIGH_RES_PLAN -tp LOW_RES_PLAN
```

e.g.
```
nnUNetv2_move_plans_between_datasets -s 1 -t 2 -sp nnUNetPlans -tp nnUNetPlans_new
# This command moves the nnUNetsPlans from datsaet 1 to dataset 2 under a new name as nnUNetPlans_new
```
After moving the training plans, the dataset statistics from step 1 should be copy to the nnUNetPlan_new

e.g.
```
"max": 255.0,
"mean": 132.55918884277344,
"median": 132.0,
"min": 0.0,
"percentile_00_5": 5.0,
"percentile_99_5": 252.0,
"std": 59.31269073486328
```

### 3. Prepare the data from the new plan
```
nnUNetv2_preprocess -d 2 -plans_name nnUNetPlans_new
```

### 4. Training / Finetune

```
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD -p PLANS -pretrained_weights PATH_TO_HIGH_RES_MODEL
```

e.g.
```
nnUNetv2_train 2 3d_fullres 0 -p nnUNetPlans_new -pretrained_weights /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/checkpoint_best.pth
```

# Inference
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION -f FOLD --save_probabilities
```

Example:
```
nnUNetv2_predict -i /hdd/yang/data/kidney_seg/12.1um/full_8bits_tif/whole_vol -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset005_12-1Glom_search_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/inference_whole_vol -d 5 -c 3d_fullres -f 0 -step_size 0.5 -p nnUNetPlans_new 
```