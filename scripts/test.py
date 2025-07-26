import nibabel
from pathlib import Path
from skimage.morphology import label, cube, closing

# folder = '/hdd/yang/data/care2025/validation_dataset/train_all'
# folder = Path(folder)
# for file in folder.glob('*.gz'):
#     img = nibabel.load(file)
#     not_3d = []
#     if img.ndim != 3:
#         not_3d.append(file.name)
#         print(f"File {file.name} is not 3D, it has {img.ndim} dimensions.")
#         print(img.shape)
#         img3d = nibabel.funcs.squeeze_image(img)
#         print(img3d.shape)
#         img3d.to_filename(file.name.replace('.nii.gz', '.nii.gz'))


# test connected components of preds

pred_dir = Path('/hdd/yang/data/care2025/Submission/CARE-Liver-Submission3/LiSeg_pred/')
for patient_dir in pred_dir.glob('*/'):
    print(f"Processing patient directory: {patient_dir}")
    for file in patient_dir.glob('*.gz'):
        if 'DWI' in file.name:
            img = nibabel.load(file)
            data = img.get_fdata()
            data = closing(data, cube(10))
            data = label(data)
            unique_labels = set(data.flatten())
            print(f"File {file.name} has {len(unique_labels)} unique labels.")
