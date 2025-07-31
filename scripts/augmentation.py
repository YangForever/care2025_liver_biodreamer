import numpy as np
import random
import torch
from scipy.ndimage import rotate

def ins_paste(img_src,
            label_src,
            img_tgt,
            label_tgt,
            ):
    '''
    img_src/img_tgt: (1, z, psize, psize)
    label_src/label_tgt: (z, psize, psize)
    return img_tgt, label_tgt
    '''

    _, z, y, x = img_src.shape
    _, z_tgt, y_tgt, x_tgt = img_tgt.shape


    src_z_min, src_y_min, src_x_min = torch.min(label_src.nonzero(), dim=0)[0]
    src_z_max, src_y_max, src_x_max = torch.max(label_src.nonzero(), dim=0)[0]
    src_z, src_y, src_x = src_z_max-src_z_min + 1, src_y_max-src_y_min+1, src_x_max-src_x_min+1

    tgt_z_min, tgt_y_min, tgt_x_min = torch.min(label_tgt.nonzero(), dim=0)[0]
    tgt_z_max, tgt_y_max, tgt_x_max = torch.max(label_tgt.nonzero(), dim=0)[0]
    tgt_z, tgt_y, tgt_x = tgt_z_max-tgt_z_min + 1, tgt_y_max-tgt_y_min+1, tgt_x_max-tgt_x_min+1

    src_center = torch.tensor([(src_z_min + src_z_max) / 2,
                              (src_y_min + src_y_max) / 2,
                              (src_x_min + src_x_max) / 2])
    
    tgt_center = torch.tensor([(tgt_z_min + tgt_z_max) / 2,
                                (tgt_y_min + tgt_y_max) / 2,
                                (tgt_x_min + tgt_x_max) / 2])

    

    src_mask = (label_src>0)
    tgt_mask = (label_tgt>0)

    scale_factor = ((tgt_z/src_z).item(), (tgt_y/src_y).item(), (tgt_x/src_x).item())

    ### introduce mild rotation to the src mask and img and paste it to the tgt mask and img
    img_src, label_src = RandomRotate(p=0.5, angle_range=(-30, 30)).__call__((img_src, label_src))

    # rescale the src mask to match the tgt mask size
    src_mask = torch.nn.functional.interpolate(src_mask.float().unsqueeze(0).unsqueeze(0), 
                                                scale_factor=scale_factor, 
                                                mode='trilinear', 
                                                align_corners=False).squeeze() # (z, y, x)
    src_mask = (src_mask > 0.5)
    img_src = torch.nn.functional.interpolate(img_src.unsqueeze(0),
                                                scale_factor=scale_factor, 
                                                mode='trilinear', 
                                                align_corners=False).squeeze() # (z, y, x)


    img_tgt[:, tgt_mask] = img_tgt.min()

    src_z_min, src_y_min, src_x_min = torch.min(src_mask.nonzero(), dim=0)[0]
    src_z_max, src_y_max, src_x_max = torch.max(src_mask.nonzero(), dim=0)[0]
    src_z, src_y, src_x = src_z_max-src_z_min + 1, src_y_max-src_y_min+1, src_x_max-src_x_min+1
    cent_src_z = (src_z_max + src_z_min) // 2
    cent_src_y = (src_y_max + src_y_min) // 2
    cent_src_x = (src_x_max + src_x_min) // 2

    ### place the src mask and img to the tgt mask and img at the tgt center
    # offset_z = int(tgt_center[0] - cent_src_z)
    # offset_y = int(tgt_center[1] - cent_src_y)
    # offset_x = int(tgt_center[2] - cent_src_x)
    # src_mask = torch.roll(src_mask, shifts=(offset_z, offset_y, offset_x), dims=(0, 1, 2))
    # img_src = torch.roll(img_src, shifts=(offset_z, offset_y, offset_x), dims=(0, 1, 2))

    new_z_start = int(tgt_center[0] - src_z // 2)
    new_y_start = int(tgt_center[1] - src_y // 2)
    new_x_start = int(tgt_center[2] - src_x // 2)
    new_z_end = int(new_z_start + src_z)
    new_y_end = int(new_y_start + src_y)
    new_x_end = int(new_x_start + src_x)

    # if new_z_start > 0 and new_y_start > 0 and new_x_start > 0 and \
    #     new_z_end < z_tgt and new_y_end < y_tgt and new_x_end < x_tgt:
    #     shift_margin = min(10, new_z_start, new_y_start, new_x_start, z_tgt-new_z_end, y_tgt-new_y_end, x_tgt-new_x_end)
    #     ### introduce mild translation to the src mask and img
    #     img_src, src_mask = RandomTranslate(p=0.5, shift=shift_margin).__call__((img_src, src_mask))
    #     new_z_start += shift_margin
    #     new_y_start += shift_margin
    #     new_x_start += shift_margin
    #     new_z_end += shift_margin
    #     new_y_end += shift_margin
    #     new_x_end += shift_margin
    # else:
    #     ###shift to fit the tgt bounds
    #     if new_z_start < 0:
    #         offset_z = -new_z_start
    #         new_z_end += offset_z
    #         new_z_start = 0
    #     if new_y_start < 0:
    #         offset_y = -new_y_start
    #         new_y_end += offset_y
    #         new_y_start = 0
    #     if new_x_start < 0:
    #         offset_x = -new_x_start
    #         new_x_end += offset_x
    #         new_x_start = 0
    #     if new_z_end > z_tgt or new_y_end > y_tgt or new_x_end > x_tgt:
    #         raise ValueError('src mask and img are too large to fit in the tgt mask and img')
        
    # src_mask = torch.roll(src_mask, shifts=(offset_z, offset_y, offset_x), dims=(0, 1, 2))
    # img_src = torch.roll(img_src, shifts=(offset_z, offset_y, offset_x), dims=(0, 1, 2))

    src_z_min, src_y_min, src_x_min = torch.min(src_mask.nonzero(), dim=0)[0]
    src_z_max, src_y_max, src_x_max = torch.max(src_mask.nonzero(), dim=0)[0]

    
    # print(f'Pasting src mask and img to tgt at z: {new_z_start}-{new_z_end}, y: {new_y_start}-{new_y_end}, x: {new_x_start}-{new_x_end}')
    # print(f'Src mask and img shape: {src_mask[src_z_min:src_z_max+1, src_y_min:src_y_max+1, src_x_min:src_x_max+1].shape}')
    # print(f'Tgt img shape: {img_tgt[:, new_z_start:new_z_end, new_y_start:new_y_end, new_x_start:new_x_end].shape}')

    # import pdb; pdb.set_trace()
    tgt_img_mask = (
        tgt_mask[new_z_start:new_z_end, new_y_start:new_y_end, new_x_start:new_x_end].bool() &
        src_mask[src_z_min:src_z_max+1, src_y_min:src_y_max+1, src_x_min:src_x_max+1]
    )
    label_tgt[tgt_mask] = 0 # clear the tgt mask
    label_tgt[new_z_start:new_z_end, new_y_start:new_y_end, new_x_start:new_x_end][tgt_img_mask] = 1
    img_tgt[:, new_z_start:new_z_end, new_y_start:new_y_end, new_x_start:new_x_end][:, tgt_img_mask] = img_src[src_z_min:src_z_max+1, src_y_min:src_y_max+1, src_x_min:src_x_max+1][tgt_img_mask]

    return img_tgt.unsqueeze(0), label_tgt

        
def ins_aug(imgs,
            labels
            ):
    '''
    imgs: dual-channel images (bs, 1, z, psize, psize)
    labels: (bs, z, psize, psize)

    return augmented batched data: imgs, labels
    '''
    if len(imgs) == 1: # batch size = 1 (e.g. inference), no need to augment
        return imgs, labels

    for bs in range(len(imgs)):
        img_tgt = imgs[bs].clone()
        label_tgt = labels[bs].clone()
        assert img_tgt.shape[1:] == label_tgt.shape, \
            f'Image shape {img_tgt.shape} and label shape {label_tgt.shape} do not match.'

        remain_seq = list(set(list(range(len(imgs)))) - {bs})
        src_bs = np.random.choice(remain_seq)
        img_src = imgs[src_bs].clone()
        label_src = labels[src_bs].clone()

        assert img_src.shape[1:] == label_src.shape, \
            f'Source image shape {img_src.shape} and label shape {label_src.shape} do not match.'

        img_tgt, label_tgt = ins_paste(img_src,
                                        label_src,
                                        img_tgt,
                                        label_tgt
                                        )

        imgs[bs] = img_tgt.clone()
        labels[bs] = label_tgt.clone()

    return imgs, labels


def random_crop_3d(image, 
                   label, 
                   patchsize=96, 
                   n=4, 
                   min_fg_voxel=5**3):
    '''
    given an 3d image and its segmentation label of shape (z, x, y), randomly crop them into n patches of shape (z, patchsize, patchsize),
        with priority to the patches that contain more segmentation labels
    image: dual-channel (puncta and dendrite) 3d numpy array of shape (2, z, x, y)
    label: 3d numpy array of shape (z, x, y)
    patchsize: int, size of the patch
    n: int, number of patches to crop
    min_fg_voxel: int, minimum number of foreground voxels in a patch
    return: img_pathes of shape (n, 2, z, patchsize, patchsize), label_patches of shape (n, z, patchsize, patchsize)
    '''
    _, z, x, y = image.shape
    patches = []
    
    if n == 1:
        x_start = np.random.randint(0, x-patchsize+1)
        y_start = np.random.randint(0, y-patchsize+1)
        z_start = np.random.randint(0, z-patchsize+1)
        patch = image[:, z_start:z_start+patchsize, x_start:x_start+patchsize, y_start:y_start+patchsize]
        patch_label = label[z_start:z_start+patchsize, x_start:x_start+patchsize, y_start:y_start+patchsize]
        patch, patch_label = pad_3d_image(patch, patch_label, patchsize=patchsize)
        patch_fg = torch.sum(patch_label > 0)
        while patch_fg < min_fg_voxel:
            x_start = np.random.randint(0, x-patchsize+1)
            y_start = np.random.randint(0, y-patchsize+1)
            z_start = np.random.randint(0, z-patchsize+1)
            patch = image[:, z_start:z_start+patchsize, x_start:x_start+patchsize, y_start:y_start+patchsize]
            patch_label = label[z_start:z_start+patchsize, x_start:x_start+patchsize, y_start:y_start+patchsize]
            patch, patch_label = pad_3d_image(patch, patch_label, patchsize=patchsize)
            patch_fg = torch.sum(patch_label > 0)
        patches.append((patch, patch_label, patch_fg))

    else:
        left_patches = n
        while len(patches) < n:
            num_patches_to_try = max(int(left_patches * 1.5), 1)  # Ensure at least 1
            for i in range(num_patches_to_try): # randomly select 1.5n patches
                x_start = np.random.randint(0, x-patchsize+1)
                y_start = np.random.randint(0, y-patchsize+1)
                z_start = np.random.randint(0, z-patchsize+1)
                patch = image[:, z_start:z_start+patchsize, x_start:x_start+patchsize, y_start:y_start+patchsize]
                patch_label = label[z_start:z_start+patchsize, x_start:x_start+patchsize, y_start:y_start+patchsize]
                patch, patch_label = pad_3d_image(patch, patch_label, patchsize=patchsize)
                patch_fg = np.sum(patch_label > 0)
                if patch_fg < min_fg_voxel:
                    continue
                patches.append((patch, patch_label, patch_fg))

                # Break if we've reached the desired number of patches
                if len(patches) >= n:
                    break

            probs = np.array([p[2] for p in patches]) #oversample the patches with more foreground voxels
            probs = (probs + 1e-10) / (np.sum(probs) + 1e-10)
            patches = random.choices(patches, weights=probs, k=left_patches)
            left_patches = n - len(patches)

    # voxel_size = max(voxel_size)
    img_patches = [p[0] for p in patches]
    label_patches = [p[1] for p in patches]
    img_patches = np.stack(img_patches, axis=0) # (n, 2, z, patchsize, patchsize)
    label_patches = np.stack(label_patches, axis=0) # (n, z, patchsize, patchsize)
    return img_patches, label_patches


def pad_3d_image(image, 
                 label, 
                 patchsize=96):
    '''
    image: dual-channel 3d numpy array of shape (2, z, x, y)
    label: 3d numpy array of shape (z, x, y)
    patchsize: int, size of the patch
    return: img_pathes of shape (n, z, patchsize, patchsize), label_patches of shape (n, patchsize, patchsize, patchsize)
    '''
    _, z, x, y = image.shape
    if x == patchsize and y == patchsize:
        return image, label
    x_pad = patchsize - x 
    y_pad = patchsize - y
    z_pad = patchsize - z
    if x_pad < 0 or y_pad < 0 or z_pad < 0:
        raise ValueError('Patch size should be larger than the image size')
    image = np.pad(image, ((0, 0), (0, z_pad), (0, x_pad), (0, y_pad)), mode='constant', constant_values=0)
    label = np.pad(label, ((0, z_pad), (0, x_pad), (0, y_pad)), mode='constant', constant_values=0)
    return image, label


class AddGuassianNoise(object):
    '''
    add guassian noise to the 3d image
    '''
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image, label = data
        if self.mean == 0 and self.std == 0:
            return (image, label)
        if image.dim() == 3: # single image (z, x, y)
            noise = torch.normal(self.mean, self.std, size=image.shape)
            image += noise
        elif image.dim() == 4: # image batches (n, z, x, y)
            for i in range(image.shape[0]):
                noise = torch.normal(self.mean, self.std, size=image[i].shape)
                image[i] += noise
        return (image, label)
    
class RandomFlip(object):
    '''
    randomly flip an axis of the 3d image with its segmentation label
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data
        if image.dim() == 4: # single image (2, z, x, y)
            if random.random() < self.p:
                axis = random.randint(0, 2)
                image = torch.flip(image, dims=[axis+1])
                label = torch.flip(label, dims=[axis])
        elif image.dim() == 5: # image batches (n, 2, z, x, y)
            for i in range(image.shape[0]):
                if random.random() < self.p:
                    axis = random.randint(0, 2)
                    image[i] = torch.flip(image[i], dims=[axis+1])
                    label[i] = torch.flip(label[i], dims=[axis])
        return (image, label)
    
class RandomRotate(object):
    '''
    Randomly rotates a 3D image and its corresponding segmentation label by an
    arbitrary degree.

    This implementation uses scipy.ndimage.rotate, which handles the complex
    interpolation required for non-90-degree rotations.

    Key Features:
    - Image is rotated with high-quality bicubic interpolation (order=3).
    - Label is rotated with nearest-neighbor interpolation (order=0) to preserve
      integer class values.
    - The rotation is performed in a plane defined by two randomly chosen axes.
    - The output dimensions are kept the same as the input (`reshape=False`).
    '''
    def __init__(self, p=0.5, angle_range=(-30, 30)):
        """
        Args:
            p (float): The probability of applying the rotation.
            angle_range (tuple of int): The range (min, max) from which to sample
                                        a random rotation angle in degrees.
        """
        self.p = p
        self.angle_range = angle_range

    def __call__(self, data):
        image, label = data

        # Apply rotation with a probability p
        if random.random() >= self.p:
            return image, label
            
        # Sample a random angle and rotation axes
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        # Axes for the 3D volume (z, x, y) are 1, 2
        axes = [1, 2]

        # --- Convert torch tensors to numpy for SciPy processing ---
        # Note: This requires a CPU -> NumPy -> CPU roundtrip
        image_np = image.cpu().numpy()
        label_np = label.cpu().numpy()

        # --- Perform Rotation using SciPy ---
        
        # For the image, use high-order interpolation (e.g., bicubic)
        # We must rotate each channel independently

        rotated_image = rotate(
            image_np, 
            angle=angle, 
            axes=axes, 
            reshape=False, 
            order=3, # Bicubic interpolation
            mode='constant', 
            cval=0
        )

        

        # For the label, MUST use nearest-neighbor interpolation (order=0)
        rotated_label = rotate(
            label_np, 
            angle=angle, 
            axes=axes, 
            reshape=False, 
            order=0, # Nearest-neighbor interpolation
            mode='constant', 
            cval=0 # Background class is usually 0
        )
        
        # --- Convert back to torch tensors ---
        rotated_image = torch.from_numpy(rotated_image).to(image.device)
        rotated_label = torch.from_numpy(rotated_label).to(label.device)

        return (rotated_image, rotated_label)
    
class RandomTranslate(object):
    '''
    randomly translate the 3d image with its segmentation label along random axis
    '''
    def __init__(self, p=0.5, shift=20):
        self.p = p
        self.shift = shift

    def readjust_shift(self, mask, ax, shift):
        inds = mask.nonzero()
        z_min, y_min, x_min = torch.min(inds[:, 0]), torch.min(inds[:, 1]), torch.min(inds[:, 2])
        z_max, y_max, x_max = torch.max(inds[:, 0]), torch.max(inds[:, 1]), torch.max(inds[:, 2])
        if ax == 0:
            shift = max(shift, z_min) if shift < 0 else min(shift, mask.shape[0] - z_max)
        elif ax == 1:
            shift = max(shift, y_min) if shift < 0 else min(shift, mask.shape[1] - y_max)
        elif ax == 2:
            shift = max(shift, x_min) if shift < 0 else min(shift, mask.shape[2] - x_max)
        return shift if isinstance(shift, int) else shift.item()

    def __call__(self, data):
        image, label = data
        if label.sum() == 0: # no foreground voxels, return the original image and label
            return (image, label)
        if image.dim() == 3: # single image (z, x, y)
            if random.random() < self.p:
                ax = random.randint(0, 2)
                shift = random.randint(-self.shift, self.shift)
                shift = self.readjust_shift(label, ax, shift)
                image = torch.roll(image, shifts=shift, dims=ax)
                label = torch.roll(label, shifts=shift, dims=ax)
                #if not enough foreground voxels, discard the translation
                if torch.sum(label>0) < 5**3:
                    return (image, label)

        elif image.dim() == 4: # image batches (n, z, x, y)
            raise NotImplementedError("RandomTranslate is not implemented for batches of images yet.")
        return (image, label)
    