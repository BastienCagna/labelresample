from soma import aims
from label_resample import resample
import anatomist.api as anatomist
import nibabel as nb
import numpy as np


imgf = "/var/tmp/one_voxel_image.nii"
rs_imgf = "/var/tmp/resampled_one_voxel_image.nii"

dt = np.zeros((100, 100, 100))
dt[50, 50, 50] = 1
aff = np.eye(4)
aff[:, 3] = 1
img = nb.Nifti1Image(dt, affine=aff)
nb.save(img, imgf)

vol = aims.read(imgf)
print(vol.header()['volume_dimension'], vol.header()['voxel_size'])

rvol = resample(imgf, None, (2, 2, 2), 0)
aims.write(rvol, rs_imgf)

# a = anatomist.Anatomist()
# avol = a.loadObject(resampled_skeleton)
# win = a.createWindow('Axial')
# a.addObjects([avol], [win])

print(rvol.header()['volume_dimension'], rvol.header()['voxel_size'])

print(np.sum(np.asarray(vol) != 0))
print(np.sum(np.asarray(rvol) != 0))
