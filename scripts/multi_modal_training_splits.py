from pathlib import Path
import json
import random
random.seed(42)


def split_5_fold_based_on_modality(train_image_dir, json_output_path):
    """
    Splits the training images into 5 folds based on modality.
    
    Args:
        train_image_dir (Path): Path to the training images directory.
        json_output_path (Path): Path to save the JSON output.
    """
    modality_list = ["GED4","GED3","GED2","GED1","T1","T2","DWI_800","augmented"]
    train_image_dir = Path(train_image_dir)
    json_output_path = Path(json_output_path)
    if not train_image_dir.exists():
        raise FileNotFoundError(f"Training image directory {train_image_dir} does not exist.")
    if not json_output_path.parent.exists():
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    images_per_modality = {modality: [] for modality in modality_list}
    found = 0
    for file in train_image_dir.glob('*.gz'):
        for modality in modality_list:
            if modality in file.name:
                images_per_modality[modality].append(file.name.split('.')[0][:-5])
                found = 1
                break
        if found == 0:
            images_per_modality['augmented'].append(file.name.split('.')[0][:-5])
        found = 0
    
    for modality in modality_list:
        print(f"Number of images for {modality}: {len(images_per_modality[modality])}")
        print(f"Example image for {modality}: {images_per_modality[modality][15]}")
    
    # Create folds
    folds = []
    for i in range(5):
        fold = {"train": [], "val": []}
        for modality in modality_list:
            # randomly select 80% for training and 20% for validation
            num_images = len(images_per_modality[modality])
            train_count = int(num_images * 0.8)
            fold["train"].extend(images_per_modality[modality][:train_count])
            fold["val"].extend(images_per_modality[modality][train_count:])
        # Shuffle the training and validation images
        random.shuffle(fold["train"])
        random.shuffle(fold["val"])
        folds.append(fold)
    print(f"Total number of images in each fold: {len(folds[0]['train']) + len(folds[0]['val'])}")

    with open(json_output_path, 'w') as f:
        json.dump(folds, f, indent=4)

if __name__ == "__main__":
    train_image_dir = '/hdd/yang/data/kidney_seg_nnunet/nnUNet_raw/Dataset015_Care2025_benchmark3/imagesTr'
    json_output_path = '/hdd/yang/data/kidney_seg_nnunet/nnUNet_preprocessed/Dataset015_Care2025_benchmark3/folds.json'
    split_5_fold_based_on_modality(train_image_dir, json_output_path)
            
    
   