import torchio as tio
import torch
import matplotlib.pyplot as plt

# Load your image
subject = tio.Subject(
    t1=tio.ScalarImage('/mimer/NOBACKUP/groups/brainage/data/oasis3/derivatives/mriprep/sub-OAS30001/ses-d0129/sub-OAS30001_ses-d0129_space-MNI152NLin2009cAsym_desc-brain_T1w.nii.gz')
)

# Access the image tensor
data = subject.t1.data  # shape: (1, X, Y, Z)
print("Shape:", data.shape)

# Convert to numpy and remove channel dimension
array = data.squeeze(0).numpy()

# Pick a slice index (axial view)
slice_index = array.shape[2] // 2

plt.imshow(array[:, :, slice_index].T, cmap='gray', origin='lower')
plt.axis('off')
plt.gca().set_position([0, 0, 1, 1])
plt.show()
plt.savefig('slice1.png')

# Load your image
subject = tio.Subject(
    t1=tio.ScalarImage('/mimer/NOBACKUP/groups/brainage/data/oasis3/derivatives/mriprep/sub-OAS30001/ses-d4467/sub-OAS30001_ses-d4467_space-MNI152NLin2009cAsym_desc-brain_T1w.nii.gz')
)

# Access the image tensor
data = subject.t1.data  # shape: (1, X, Y, Z)
print("Shape:", data.shape)

# Convert to numpy and remove channel dimension
array = data.squeeze(0).numpy()

# Pick a slice index (axial view)
slice_index = array.shape[2] // 2

plt.imshow(array[:, :, slice_index].T, cmap='gray', origin='lower')
plt.axis('off')
plt.gca().set_position([0, 0, 1, 1])
plt.show()
plt.savefig('slice2.png')

# Load the tensor
volume = torch.load("/mimer/NOBACKUP/groups/brainage/thesis_brainage/data/sub-OAS30001/ses-d0129/sub-OAS30001_ses-d0129_latent_4.pt")  # shape could be (1, X, Y, Z) or (X, Y, Z)
print("Volume shape:", volume.shape)

# Remove channel dimension if it exists
if volume.ndim == 4 and volume.shape[0] == 1:
    volume = volume.squeeze(0)  # now (X, Y, Z)

# Pick a slice index (axial)
slice_index = volume.shape[2] // 2
slice_img = volume[:, :, slice_index].T  # transpose for correct orientation

# Plot without white box
plt.imshow(slice_img, cmap="gray", origin="lower")
plt.axis("off")
plt.gca().set_position([0, 0, 1, 1])
plt.show()
plt.savefig('slice1_latent.png')

# Load the tensor
volume = torch.load("/mimer/NOBACKUP/groups/brainage/thesis_brainage/data/sub-OAS30001/ses-d4467/sub-OAS30001_ses-d4467_latent_4.pt")  # shape could be (1, X, Y, Z) or (X, Y, Z)
print("Volume shape:", volume.shape)

# Remove channel dimension if it exists
if volume.ndim == 4 and volume.shape[0] == 1:
    volume = volume.squeeze(0)  # now (X, Y, Z)

# Pick a slice index (axial)
slice_index = volume.shape[2] // 2
slice_img = volume[:, :, slice_index].T  # transpose for correct orientation

# Plot without white box
plt.imshow(slice_img, cmap="gray", origin="lower")
plt.axis("off")
plt.gca().set_position([0, 0, 1, 1])
plt.show()
plt.savefig('slice2_latent.png')