import nibabel as nib
import numpy as np

def normalise_image(im_path):
#This function normalises the image slice by slice.
    '''
    Args:
        im_path: str, path to the image
    Returns:
        norm_slice: np.array, normalised slice
    '''
    image = nib.load(im_path)
    if image.ndim != 3:
        image = nib.funcs.squeeze_image(image)

    data_type = image.get_data_dtype()
    print(f'Normalising image {im_path} of type {data_type}.')
    
    image_data = image.get_fdata()

    # normalise by max and min
    new_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    new_image = (new_image * 255).astype(np.uint8)
    # change image header to reflect the new data type
    image.header.set_data_dtype(np.uint8)
    # create new image with the same affine and header
    norm_data = nib.Nifti1Image(new_image, image.affine, image.header)

    return norm_data

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