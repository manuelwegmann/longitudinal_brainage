import ants

target_path = "/mimer/NOBACKUP/groups/brainage/thesis_brainage/playground/data/mni_icbm152_nlin_sym_09a/mni_raw.nii"     # The reference image
moving_path = "/mimer/NOBACKUP/groups/brainage/thesis_brainage/playground/data/mni_icbm152_nlin_sym_09a/moving_raw.nii.gz"     # The image you want to align

target = ants.image_read(target_path)
moving = ants.image_read(moving_path)

def preprocess(img):
    # N4 bias field correction
    n4 = ants.n4_bias_field_correction(img)
    
    # Normalise intensities (robust z-score)
    norm = ants.iMath(n4, "Normalize")
    
    return norm

target_prep = preprocess(target)
moving_prep = preprocess(moving)

import numpy as np

t = target_prep.numpy()
m = moving_prep.numpy()

sx = t.shape[0] // 2   # sagittal
sy = t.shape[1] // 2   # coronal
sz = t.shape[2] // 2   # axial

import matplotlib.pyplot as plt

def compare_slices(t, m, idx, axis_name):
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    
    if axis_name == "sagittal":
        axs[0].imshow(np.rot90(t[idx, :, :]), cmap='gray')
        axs[1].imshow(np.rot90(m[idx, :, :]), cmap='gray')
    elif axis_name == "coronal":
        axs[0].imshow(np.rot90(t[:, idx, :]), cmap='gray')
        axs[1].imshow(np.rot90(m[:, idx, :]), cmap='gray')
    elif axis_name == "axial":
        axs[0].imshow(np.rot90(t[:, :, idx]), cmap='gray')
        axs[1].imshow(np.rot90(m[:, :, idx]), cmap='gray')

    axs[0].set_title(f"Target ({axis_name})")
    axs[1].set_title(f"Moving ({axis_name})")

    for a in axs:
        a.axis("off")

    plt.tight_layout()
    plt.savefig("compare_axial.png", dpi=150)
    plt.show()


