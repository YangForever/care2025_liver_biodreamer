#!/usr/bin/env python3
"""
Liver Fibrosis Stage Prediction Script
Converted from inference.ipynb

This script processes medical imaging data to predict liver fibrosis stages
using trained RandomForest models.
"""

import nibabel as nib
import os
from augmentation import *
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
import csv
import pandas as pd
import joblib
from skimage import measure
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import skew, kurtosis
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import argparse


# Configuration
mod1 = 'DWI_800.nii.gz'
mod2 = 'GED1.nii.gz'
mod3 = 'GED2.nii.gz'
mod4 = 'GED3.nii.gz'
mod5 = 'GED4.nii.gz'
mod6 = 'T1.nii.gz'
mod7 = 'T2.nii.gz'
mod8 = 'mask_GED4.nii.gz'

mods = {'DWI': mod1, 'GED1': mod2, 'GED2': mod3, 'GED3': mod4, 'GED4': mod5, 'T1': mod6, 'T2': mod7, 'mask_GED4': mod8}

vendor_to_number = {'A': 0, 'B1': 1, 'B2': 2, 'C': 3}


def compute_surface_area(binary_mask):
    """Compute surface area of a 3D binary mask using marching cubes."""
    verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0)
    area = 0.0
    for tri in faces:
        p0, p1, p2 = verts[tri]
        tri_area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        area += tri_area
    return area, verts


def compute_convex_volume(verts):
    """Compute convex hull volume from vertices."""
    try:
        hull = ConvexHull(verts)
        return hull.volume
    except:
        return np.nan


def compute_directional_glcm_features(img_3d, mask_3d, distances=[1], levels=256):
    """Compute simple directional GLCM contrast on three orthogonal slices through ROI."""
    # Take central slices through the ROI
    coords = np.array(np.where(mask_3d))
    zc, yc, xc = [int(np.mean(c)) for c in coords]

    features = {}

    # Helper to compute contrast per angle in 2D
    def glcm_contrast(slice2d):
        # quantize
        img = slice2d.astype(np.uint8)
        glcm = graycomatrix(img, distances=distances, angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=levels, symmetric=True, normed=True)
        contrast = [graycoprops(glcm, 'contrast')[0, i] for i in range(4)]
        return contrast

    # XY slice
    contrast_xy = glcm_contrast(img_3d[zc])
    features['glcm_xy_contrast_mean'] = np.mean(contrast_xy)
    features['glcm_xy_anisotropy'] = (max(contrast_xy) - min(contrast_xy)) / (np.mean(contrast_xy)+1e-8)

    # XZ slice
    contrast_xz = glcm_contrast(img_3d[:, yc, :])
    features['glcm_xz_contrast_mean'] = np.mean(contrast_xz)
    features['glcm_xz_anisotropy'] = (max(contrast_xz) - min(contrast_xz)) / (np.mean(contrast_xz)+1e-8)

    # YZ slice
    contrast_yz = glcm_contrast(img_3d[:, :, xc])
    features['glcm_yz_contrast_mean'] = np.mean(contrast_yz)
    features['glcm_yz_anisotropy'] = (max(contrast_yz) - min(contrast_yz)) / (np.mean(contrast_yz)+1e-8)

    return features


def extract_features_with_intensity_gradient(mask_3d, img_3d):
    """Extract comprehensive features from mask and image data."""
    mask_3d = mask_3d.astype(bool)
    props = measure.regionprops(mask_3d.astype(np.uint8), intensity_image=img_3d)

    # precompute gradients
    gx, gy, gz = np.gradient(img_3d.astype(np.float32))
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # structure tensor
    A_elems = structure_tensor(img_3d, sigma=1)
    l1, l2, l3 = structure_tensor_eigenvalues(A_elems)
    # coherence: (λ1 - λ2) / (λ1 + λ2) for the two largest
    coherence_map = (l1 - l2) / (l1 + l2 + 1e-8)

    # Hessian
    H_elems = hessian_matrix(img_3d, sigma=1, order='rc')
    h_eigs = hessian_matrix_eigvals(H_elems)  # returns sorted eigenvalues
    h_eigs = np.stack(h_eigs, axis=-1)  # shape (Z,Y,X,3)
    # simple anisotropy ratio: |λ1| / (|λ2|+|λ3|)
    h_aniso = np.abs(h_eigs[..., 0]) / (np.abs(h_eigs[..., 1]) + np.abs(h_eigs[..., 2]) + 1e-8)

    # anisotropy maps
    coherence_vals = coherence_map[mask_3d]
    mean_coherence = np.mean(coherence_vals)
    std_coherence = np.std(coherence_vals)

    h_aniso_vals = h_aniso[mask_3d]
    mean_h_aniso = np.mean(h_aniso_vals)
    std_h_aniso = np.std(h_aniso_vals)

    # directional GLCM
    glcm_feats = compute_directional_glcm_features(img_3d, mask_3d)

    feature_list = []

    ### merge regions
    for region in props:
        volume = region.area

        # geometric features
        surf_area, verts = compute_surface_area(mask_3d)
        sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / surf_area if surf_area > 0 else np.nan
        convex_volume = compute_convex_volume(verts)
        solidity = volume / convex_volume if convex_volume and convex_volume > 0 else np.nan
        inertia = region.inertia_tensor
        eigvals, _ = np.linalg.eigh(inertia)
        elongation = np.sqrt(eigvals.min() / eigvals.max()) if eigvals.max() > 0 else np.nan

        # intensity features
        intensities = img_3d[mask_3d]
        mean_int = np.mean(intensities)
        std_int = np.std(intensities)
        min_int = np.min(intensities)
        max_int = np.max(intensities)
        med_int = np.median(intensities)
        skew_int = skew(intensities, bias=False)
        kurt_int = kurtosis(intensities, bias=False)

        # gradient-based features
        grad_vals = grad_mag[mask_3d]
        mean_grad = np.mean(grad_vals)
        std_grad = np.std(grad_vals)
        min_grad = np.min(grad_vals)
        max_grad = np.max(grad_vals)
        med_grad = np.median(grad_vals)
        skew_grad = skew(grad_vals, bias=False)
        kurt_grad = kurtosis(grad_vals, bias=False)

        vol_portion = np.sum(mask_3d) / np.prod(mask_3d.shape)

        feature_list.append({
            "label": region.label,
            # morphology
            "volume_voxels": volume,
            "vol_portion": vol_portion,
            "surface_area": surf_area,
            "sphericity": sphericity,
            "convex_volume": convex_volume,
            "solidity": solidity,
            "elongation": elongation,
            # intensity
            "intensity_mean": mean_int,
            "intensity_std": std_int,
            "intensity_min": min_int,
            "intensity_max": max_int,
            "intensity_median": med_int,
            "intensity_skew": skew_int,
            "intensity_kurtosis": kurt_int,
            # gradient
            "grad_mean": mean_grad,
            "grad_std": std_grad,
            "grad_min": min_grad,
            "grad_max": max_grad,
            "grad_median": med_grad,
            "grad_skew": skew_grad,
            "grad_kurtosis": kurt_grad,
            # anisotropy
            "coherence_mean": mean_coherence,
            "coherence_std": std_coherence,
            "hessian_aniso_mean": mean_h_aniso,
            "hessian_aniso_std": std_h_aniso,
        })
        feature_list[0].update(glcm_feats)
    return pd.DataFrame(feature_list)


def predict_new_data(model_path: str,
                     mask: np.ndarray, 
                     img: np.ndarray,
                     sample_name: str) -> int:
    """Predict the label for a new mask and image using the trained model."""
    mask = mask.astype(np.uint8)
    assert mask.shape == img.shape, "Mask and image must have the same shape"
    df = extract_features_with_intensity_gradient(mask, img)
    vendor = sample_name.split('-')[1]
    vendor_number = vendor_to_number.get(vendor, -1)
    df['vendor'] = vendor_number
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            assert hasattr(model, 'predict'), "Loaded model does not have a predict method"
            assert model.__class__.__name__ == 'RandomForestClassifier', "Loaded model is not a RandomForestClassifier"
            # Ensure the model only uses features it was trained on
            df = df[df.columns.intersection(model.feature_names_in_)]

    except FileNotFoundError:
        print(f"Model file {model_path} not found.")
        return None
    if df.empty:
        print("No features extracted from the mask. Cannot predict.")
        return None
    
    prediction = model.predict_proba(df)

    return prediction[0]


def predict_on_dataset(model_paths: dict[str, str], dataset: list) -> list:
    """Predict labels for a dataset of (mask, img) pairs."""
    predictions = []
    for img, mask, roi, sample_name, mod_name in tqdm(dataset):
        model_path = model_paths.get(mod_name, None)
        if model_path is None:
            raise FileNotFoundError(f"No model found for modality: {mod_name}. Skipping sample: {sample_name}")
        pred = predict_new_data(model_path, mask, img, sample_name)
        if pred is not None:
            predictions.append((sample_name, pred))
    return predictions


def load_and_match_data(search_dir_data, search_dir_mask, target_mods='T1'):
    """Load and match image data with segmentation masks."""
    search_dir_data = Path(search_dir_data)
    
    # Recursively find all files ending with '.nii.gz'
    mod_list = []
    for mod in mods.values():
        file_list = list(search_dir_data.rglob(f'*{os.sep}{mod}'))
        mod_list.append(file_list)
        print(f"Mod {mod} found {len(file_list)} files.")

    mask_list = []
    search_dir_mask = Path(search_dir_mask)
    for mask in search_dir_mask.rglob('*.nii.gz'):
        mask_list.append(mask)
    print(f"Found {len(mask_list)} segmentation masks.")

    # pair the samples with segmentation masks
    mask_str_list = [str(mask.parent).split(os.sep)[-1] + os.sep + mask.name for mask in mask_list]
    mod_roi_list = []

    mod_index = list(mods.keys()).index(target_mods)
    for sample_GED4 in mod_list[list(mods.keys()).index('GED4')]:  # Use GED4 as the base for matching
        sample = str(sample_GED4).replace('GED4.nii.gz', mods[target_mods])  # Replace 'GED4' with the target modality
        #wrap in Path to ensure compatibility
        sample = Path(sample)
        mod_name = sample.name
        sample_name = str(sample.parent).split(os.sep)[-1]  # Get the last part of the path as sample name
        mask_name = sample_name + os.sep + mods[target_mods].replace('.nii.gz', '_pred.nii.gz') 

        if mask_name in mask_str_list:
            idx = mask_str_list.index(mask_name)
            
            mask = nib.load(mask_list[idx]).get_fdata().astype(np.float32)
            img = nib.load(str(sample).replace(mod_name, f'{target_mods}.nii.gz')).get_fdata().astype(np.float32)
            roi = img[mask>0]
            if roi.size == 0:
                print(f"No ROI found for sample: {sample_name} with mod: {mod_name}")
                continue
            mod_roi_list.append((img, mask, roi, sample_name, target_mods))
            print(f"{mask_list[idx]} {str(sample).replace(mod_name, f'{target_mods}.nii.gz')}")
        else:
            print(f"No mask found for sample: {sample_name} with mod: {mod_name}")
            #use the GED4 mask as a fallback
            ged4_mask_name = sample_name + os.sep + mods['GED4'].replace('.nii.gz', '_pred.nii.gz') 
            if ged4_mask_name in mask_str_list:
                idx = mask_str_list.index(ged4_mask_name)
                mask = nib.load(mask_list[idx]).get_fdata().astype(np.float32)
                img = nib.load(str(sample).replace(mod_name, 'GED4.nii.gz')).get_fdata().astype(np.float32)
                roi = img[mask>0]
                if roi.size == 0:
                    print(f"No ROI found for sample: {sample_name} with mod: {mod_name} using GED4 mask")
                    continue
                mod_roi_list.append((img, mask, roi, sample_name, 'GED4'))
            else:
                raise FileNotFoundError(f"No mask found for sample: {sample_name} with mod: {mod_name} and no GED4 mask available.")
    
    return mod_roi_list


def main(input_dir, mask_dir, output_dir):
    """
    Run liver fibrosis stage prediction.
    
    Args:
        input_dir (str): Path to input data directory
        mask_dir (str): Path to mask directory (segmentation results)
        output_dir (str): Path to output directory for results
    """
    # Convert to Path objects
    input_dir = Path(input_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set paths for models (assuming they are in the scripts directory)
    script_dir = Path(__file__).parent
    model_t1 = script_dir / 'RF_0.9722_bal_T1.pkl'
    model_ged4 = script_dir / 'RF_1.0000_bal_GED4.pkl'
    
    print("=" * 60)
    print("CONTRAST PREDICTION")
    print("=" * 60)
    
    output_file_contrast = output_dir / 'LiFS_pred_contrast.csv'
    
    # Load and match data for contrast prediction
    print("Loading and matching data for contrast prediction...")
    mod_roi_list = load_and_match_data(str(input_dir), str(mask_dir), target_mods='GED4')
    
    # Predict on the dataset
    print("Running contrast predictions...")
    pred_all = predict_on_dataset({'GED4': str(model_ged4)}, mod_roi_list)
    
    # Prepare CSV output
    print("Preparing contrast output...")
    rows = []
    for sample_name, pred in pred_all:
        # pred contains probabilities for each class [S1, S2, S3, S4]
        prob_S4 = pred[3]  # Probability of class S4 (Cirrhosis)
        prob_S1 = pred[0]  # Probability of class S1 (Non-fibrotic)
        rows.append([sample_name, 'Contrast', prob_S4, prob_S1])

    # Write contrast results to CSV
    print(f"Saving contrast predictions to {output_file_contrast}")
    with open(output_file_contrast, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Case', 'Setting', 'Subtask1_prob_S4', 'Subtask2_prob_S1'])
        writer.writerows(rows)
    
    print(f"Contrast processing complete! Results saved to {output_file_contrast}")
    print(f"Processed {len(pred_all)} samples for contrast prediction")
    
    print("=" * 60)
    print("NON-CONTRAST PREDICTION")
    print("=" * 60)
    
    output_file_non_contrast = output_dir / 'LiFS_pred_non_contrast.csv'
    
    # Load and match data for non-contrast prediction
    print("Loading and matching data for non-contrast prediction...")
    mod_roi_list = load_and_match_data(str(input_dir), str(mask_dir), target_mods='T1')
    
    # Predict on the dataset
    print("Running non-contrast predictions...")
    pred_all = predict_on_dataset({'T1': str(model_t1), 'GED4': str(model_ged4)}, mod_roi_list)
    
    # Prepare CSV output
    print("Preparing non-contrast output...")
    rows = []
    for sample_name, pred in pred_all:
        # pred contains probabilities for each class [S1, S2, S3, S4]
        prob_S4 = pred[3]  # Probability of class S4 (Cirrhosis)
        prob_S1 = pred[0]  # Probability of class S1 (Non-fibrotic)
        rows.append([sample_name, 'NonContrast', prob_S4, prob_S1])

    # Write non-contrast results to CSV
    print(f"Saving non-contrast predictions to {output_file_non_contrast}")
    with open(output_file_non_contrast, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Case', 'Setting', 'Subtask1_prob_S4', 'Subtask2_prob_S1'])
        writer.writerows(rows)
    
    print(f"Non-contrast processing complete! Results saved to {output_file_non_contrast}")
    print(f"Processed {len(pred_all)} samples for non-contrast prediction")
    
    print("=" * 60)
    print("ALL STAGE PREDICTIONS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Liver Fibrosis Stage Prediction')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Path to input data directory (structured as released validation data)')
    parser.add_argument('--mask_dir', type=str, required=True,
                       help='Path to mask directory containing segmentation results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory for saving prediction results')
    
    # Parse arguments and run main function
    args = parser.parse_args()
    main(args.input_dir, args.mask_dir, args.output_dir) 