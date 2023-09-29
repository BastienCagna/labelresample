import numpy as np
from soma import aims


def coordinates_array(shape, offset=[0, 0, 0], dtype=int):
    x, y, z = shape
    return np.array([
        np.repeat(np.arange(x), y * z) + offset[0],
        np.tile(np.repeat(np.arange(y), z), x) + offset[1],
        np.tile(np.arange(z), x * y) + offset[2]
    ], dtype=dtype)


def transform_3d_array(arr, trm):
    arr = np.array(arr)
    if len(arr.shape) != 3:
        raise ValueError("Invalid input array.")

    #     X, Y, Z = img.shape
    #     coords = np.array([
    #         # Variation allong axis 0
    #         np.repeat(np.arange(X), Y * Z),
    #         # Variation allong axis 1
    #         np.tile(np.repeat(np.arange(Y), Z), X),
    #         # Variation along axis 2
    #         np.tile(np.arange(Z), X * Y)
    #     ], dtype=int)
    coords = coordinates_array(img.shape)
    tcoords = np.dot(trm[:3, :3],
                     coords + np.tile(trm[:3, 3], (coords.shape[1], 1)).T)

    valids = np.prod(tcoords >= 0, axis=0) * np.prod(
        tcoords < np.tile(img.shape, (tcoords.shape[1], 1)).T, axis=0)
    tcoords = np.array(tcoords.astype(int))
    timg = -np.ones((img.shape))
    vimg = np.zeros((img.shape))

    valids = valids.astype(bool)
    timg[coords[0, valids], coords[1, valids], coords[2, valids]] = img[
        tcoords[0, valids], tcoords[1, valids], tcoords[2, valids]]
    return timg


def resample_3d_arr(original_dt, trm=None, ratio=0, patch=None):
    """
        Parameters
        ----------
        ori: MRI image file (.nii, .nii.gz, .ima)
            Original image.
        trm: 4x4 matrix (optional)
            Transformation matrix. Default is None which skips the transformation step
        ratio: float or tuple or array of 3 elements (optional)
            Resampling ratio. Default is 0 which skips the resampling step
    """
    if not isinstance(ratio, (int, float)):
        raise ValueError("ratio should be either an integer or a float.")

    # Sampling patch
    if len(patch.shape) == 1 and patch.shape[0] == 1:
        patch_coords = np.array([0, 0, 0], dtype=int)
    else:
        x, y, z = np.array((np.array(patch.shape) - 1) / 2).astype(int)
        patch_coords = coordinates_array(patch.shape, offset=(-x, -y, -z))
    patch = patch.flatten()

    if trm is not None:
        # Apply the transformation to the original image
        original_dt = transform_3d_array(original_dt, trm)

    if isinstance(ratio, int) and ratio == 0:
        return original_dt

    # Init the resampled image
    new_dim = np.round(np.asarray(original_dt.shape[:3]) / ratio).astype(int)

    # Coordinates in the resampled space
    # output is of shape (3, number of voxel in resampled space)
    tgt_coords = coordinates_array(new_dim)
    n_vx = tgt_coords.shape[1]

    # Then resample

    # Create a patch per voxel
    # output is of shape (3, number of voxels in the patch, number of voxel in the target)
    patch_coords = np.tile(patch_coords.T, (n_vx, 1, 1)).T
    # Center each patch on the closest voxel of the original space - output is of shape
    # (3, number of voxels in the patch, number of voxel in the target)
    tgt_coords = np.round(tgt_coords * ratio).astype(int)
    patch_coords += np.swapaxes(
        np.tile(tgt_coords, (patch_coords.shape[1], 1, 1)), 0, 1)
    # Do not take into account coordinates that are outside the original image
    maxs = np.repeat(original_dt.shape[:3],
                     patch_coords.shape[1] * patch_coords.shape[2]).reshape(
        (patch_coords.shape))
    valid_coords = np.prod((patch_coords >= 0) * (patch_coords < maxs), axis=0)
    valid_coords = valid_coords.flatten().astype(bool)
    # Read values for each valid patch coordinates
    # Output is ()
    flt_coords = patch_coords.reshape(
        (3, patch_coords.shape[1] * patch_coords.shape[2]))
    values = np.zeros((flt_coords.shape[1],))
    print(flt_coords[:, valid_coords == True][:, :16])

    values[valid_coords] = original_dt[
        flt_coords[0, valid_coords], flt_coords[1, valid_coords], flt_coords[
            2, valid_coords]]
    # Get the maximal value in each patch
    values = np.max(
        values.reshape((patch_coords.shape[1], patch_coords.shape[2])), axis=0)

    return values.reshape(new_dim)


def resample_image(img_f, output_vs, transfo_file=None, patch_size=3):
    vol = aims.read(img_f)
    transfo = aims.read(transfo_file) if transfo_file else None
    ratio = np.array(vol.header()['voxel_size'][:3]) / output_vs
    patch = np.ones((patch_size, patch_size, patch_size))
    return resample_3d_arr(np.array(vol), transfo, ratio, patch)


def demo():
    import matplotlib.pyplot as plt
    from time import time

    img = -np.ones((50, 50, 50))
    img[0, :img.shape[1]//2, :] = 1
    img[0, img.shape[1]//2:, :] = 2
    img[0, -1, :] = 3
    img[:, 30, :] = 4

    # Transformation
    angle = np.deg2rad(0)
    tx, ty, tz = 0, 0, 0
    trm = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, tx],
            [np.sin(angle), np.cos(angle), 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])

    r = 4
    tic = time()
    timg = resample_3d_arr(img, ratio=r, trm=trm, patch=np.ones((3, 3, 3)))
    print(time() - tic)
    print(timg.shape)

    fig = plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(img[:, :, 20], interpolation="nearest", aspect='auto',
               origin="lower")
    plt.title("Orignal image")
    plt.subplot(122)
    plt.imshow(timg[:, :, 20 // r], interpolation="nearest", aspect='auto',
               origin="lower")
    plt.title("Transformed image")


if __name__ == "__main__":
    demo()
