from soma import aims
from label_resample import other_resample
import anatomist.api as anatomist
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt


imgf = "/var/tmp/several_voxel_image.nii"
rs_imgf = "/var/tmp/resampled_several_voxel_image.nii"

dt = np.zeros((100, 100, 100))
dt[48:52, 45:55, 47:53] = 1
dt[40:60, 55:56, 52:55] = 2
dt[55:58, 40:60, 52:55] = 3
aff = np.eye(4)
aff[:, 3] = 1
img = nb.Nifti1Image(dt, affine=aff)
nb.save(img, imgf)

vol = aims.read(imgf)
print(vol.header()['volume_dimension'], vol.header()['voxel_size'])

rvol = other_resample(imgf, None, (2, 2, 2), 0)
aims.write(rvol, rs_imgf)

rdt = np.asarray(rvol)
plt.figure()
plt.imshow(rdt[53, :, :], inteprolation="nearest", aspect="auto")
plt.show()