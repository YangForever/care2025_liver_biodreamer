import numpy as np
import nibabel as nib
from pathlib import Path

def construct_meta_data_dictionay(base_dir):
    """
    Construct a JSON file with voxel sizes for each image in the dataset.
    """
    base_dir = Path(base_dir)
    
    meta_data_dict = {}
    
    for image_file in base_dir.glob("images/*.nii.gz"):
        img = nib.load(image_file)
        # voxel_size = img.header.get_zooms()[:3]  # Get the first three dimensions (x, y, z)clear
        if image_file.stem.split('_')[0] not in meta_data_dict:
            meta_data_dict[image_file.stem.split('_')[0]] = {}
        meta_data_dict[image_file.stem.split('_')[0]]['affine'] = img.affine
        meta_data_dict[image_file.stem.split('_')[0]]['header'] = img.header.copy()
        meta_data_dict[image_file.stem.split('_')[0]]['extra'] = img.extra.copy()
        meta_data_dict[image_file.stem.split('_')[0]]['file_map'] = img.file_map.copy()
    
    for key in meta_data_dict.keys():
        print(key)

    return meta_data_dict

def update_voxel_size_for_aug(aug_base_dir, meta_data_dict, save_dir):
    """
    Update the voxel size for each augmented image in the dataset.
    """
    aug_base_dir = Path(aug_base_dir)
    im_save_dir = Path(save_dir, 'images')
    lbl_save_dir = Path(save_dir, 'labels')
    im_save_dir.mkdir(parents=True, exist_ok=True)
    lbl_save_dir.mkdir(parents=True, exist_ok=True)

    # for aug_image_file in aug_base_dir.glob("images/*.nii.gz"):
    #     aug_img = nib.load(aug_image_file)
    #     target_img = aug_image_file.name.split('_')[1]
    #     if target_img == 'GED4.nii.gz':
    #         continue
    #     if target_img not in voxel_size_dict:
    #         raise ValueError(f"Voxel size for {target_img} not found in the provided dictionary.")
        
    #     aug_img_data = aug_img.get_fdata(dtype=np.float32)
    #     new_aug_img = nib.Nifti1Image(aug_img_data, 
    #                                   meta_data_dict[target_img]['affine'],
    #                                   header=meta_data_dict[target_img]['header'],
    #                                   extra=meta_data_dict[target_img]['extra'])
    #     nib.save(new_aug_img, im_save_dir / aug_image_file.name)
    #     print(f"Processed {aug_image_file.name}")
    
    # Save the corresponding label file
    for aug_label_file in aug_base_dir.glob("labels/*.nii.gz"):
        aug_lbl = nib.load(aug_label_file)
        target_img = aug_label_file.name.split('_')[1]
        if target_img == 'mask':
            continue
        if target_img not in voxel_size_dict:
            print(target_img)
            raise ValueError(f"Voxel size for {target_img} not found in the provided dictionary.")
        
        aug_lbl_data = aug_lbl.get_fdata().astype(np.int16)
        meta_data_dict[target_img]['header'].set_data_dtype(np.int16)
        new_aug_lbl = nib.Nifti1Image(aug_lbl_data, 
                                      meta_data_dict[target_img]['affine'],
                                      header=meta_data_dict[target_img]['header'],
                                      extra=meta_data_dict[target_img]['extra'])
        nib.save(new_aug_lbl, lbl_save_dir / aug_label_file.name)
        print(f"Processed {aug_label_file.name}")

    

if __name__ == "__main__":
    base_dir = '/hdd/yang/data/care2025/training_dataset/benchmark1'
    voxel_size_dict = construct_meta_data_dictionay(base_dir)
    aug_base_dir = '/hdd/yang/data/care2025/training_dataset/benchmark2'
    save_dir = '/hdd/yang/data/care2025/training_dataset/benchmark2/corrected'
    update_voxel_size_for_aug(aug_base_dir, voxel_size_dict, save_dir)
