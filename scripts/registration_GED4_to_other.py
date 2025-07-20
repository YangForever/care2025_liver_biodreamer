import nibabel as nib
import SimpleITK as sitk
from pathlib import Path

def loading_data(fixed_image_path, moving_image_path):
    """
    Load the fixed and moving images using SimpleITK.

    Parameters:
    fixed_image_path (str): Path to the fixed image (Tx).
    moving_image_path (str): Path to the moving image (GED4).

    Returns:
    tuple: Fixed and moving images as SimpleITK images.
    """
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    
    return fixed_image, moving_image

def rigid_registration(fixed_image, moving_image):
    """
    Perform registration between two images with different spacings.
    
    Parameters:
    fixed_image (SimpleITK.Image): Fixed image (Other modalities).
    moving_image (SimpleITK.Image): Moving image (GED4).
    
    Returns:
    """
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsJointHistogramMutualInformation()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01, seed=42)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=5.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    rigid_transform = sitk.Euler3DTransform(initial_transform)
    registration_method.SetInitialTransform(rigid_transform)

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    rigid_final_transform = registration_method.Execute(fixed_image, moving_image)
    return rigid_final_transform



def registration(patient_folder):

    for file in patient_folder.iterdir():
        if file.suffix == ".gz" and 'mask' not in file.name and 'GED4' not in file.name:
            print(f"\t processing: {file.name}")
            file_name = file.stem.split('.')[0]
            fixed_image_path = patient_folder / f'{file_name}.nii.gz'
            moving_image_path = patient_folder / 'GED4.nii.gz'
            moving_image_mask_path = patient_folder / 'mask_GED4.nii.gz'

            fixed_image, moving_image = loading_data(fixed_image_path, moving_image_path)
            moving_image_mask = sitk.ReadImage(moving_image_mask_path, sitk.sitkUInt8)

            
            transform = rigid_registration(fixed_image, moving_image)
            # Save the resulting transform
            output_transform_path = patient_folder / f'transform_GED4_to_{file_name}.tfm'
            sitk.WriteTransform(transform, str(output_transform_path))

            resampled_image = sitk.Resample(
                moving_image,
                fixed_image,
                transform,
                sitk.sitkLinear,
                0.0,
                moving_image.GetPixelID()
            )
            output_resampled_path = patient_folder / f'GED4_registered_to_{file_name}.nii.gz'
            sitk.WriteImage(resampled_image, str(output_resampled_path))

            resampled_label = sitk.Resample(
                moving_image_mask,
                fixed_image,
                transform,
                sitk.sitkNearestNeighbor,
                0.0,
                moving_image_mask.GetPixelID()
            )
            output_label_path = patient_folder / f'mask_GED4_registered_to_{file_name}.nii.gz'
            sitk.WriteImage(resampled_label, str(output_label_path))
    print(f"Resampled label saved for {patient_folder.name}")





def registration_GED4_to_others(base_dir):
    """
    Perform non-rigid registration of GED4 to .

    Parameters:
    fixed_image (SimpleITK.Image): Fixed image (others).
    moving_image (SimpleITK.Image): Moving image (GED4).

    Returns:
    SimpleITK.Transform: The resulting transformation.
    """
    base_dir = Path(base_dir)
    for vendor in base_dir.iterdir():
        if vendor.is_dir():
            for patient in vendor.iterdir():
                if patient.is_dir():
                    for file in patient.iterdir():
                        if file.suffix == ".gz" and file.name.startswith("mask"):
                            print(f"Processing patient: {patient.name}")
                            registration(patient)
                            break



if __name__ == "__main__":
    base_dir = '/hdd/yang/data/care2025/LiQA_training_data'
    registration_GED4_to_others(base_dir)
    print("Registration completed.")