import numpy as np
from soma import aims, aimsalgo


def resample(input_image, transformation, output_vs=None, background=0):
    """
        Apply a tranformation and resample a labelled volume.

        Parameters:
            input_image: Path to the input volume (.nii or .nii.gz file)
            transformation: Linear transformation file (.trm file)
            output_vs: Output voxel size (default: None, no resampling)
            background: Background value (default: 0)

        Return:
            resampled_vol: Transformed and resampled volume
    """
    # Read inputs
    vol = aims.read(input_image)

    if transformation:
        trm = aims.read(transformation)
    else:
        trm = aims.AffineTransformation3d(np.eye(4))
    inv_trm = trm.inverse()

    if output_vs:
        output_vs = np.array(output_vs)
        
        # New volume dimensions
        resampling_ratio = np.array(vol.header()['voxel_size'][:3]) / output_vs
        orig_dim = vol.header()['volume_dimension'][:3]
        new_dim = list((resampling_ratio * orig_dim).astype(int))
    else:
        output_vs = vol.header()['voxel_size'][:3]
        new_dim = vol.header()['volume_dimension'][:3]

    # Transform the background
    # Using the inverse is more straightforward and supports non-linear
    # transforms
    resampled = aims.Volume(new_dim, dtype=vol.__array__().dtype)
    resampled.header()['voxel_size'] = output_vs
    # 0 order (nearest neightbours) resampling
    resampler = aimsalgo.ResamplerFactory(vol).getResampler(0)
    resampler.setDefaultValue(background)
    resampler.setRef(vol)
    resampler.resample_inv(vol, inv_trm, 0, resampled)

    # Create buckets
    bck = aims.BucketMap_VOID()
    bck.setSizeXYZT(*vol.header()['voxel_size'][:3], 1.)
    # build a single bucket from the volume values where voxel are non
    # equal to background value
    bk0 = bck[0]
    # TODO: This takes lot of times ==> parrallelize?
    for p in np.vstack(np.where(vol.__array__() != background)[:3]).T:
        bk0[list(p)] = 1

    # Transform buckets
    # /!\ this function has a bug for aims <= 5.0.1
    bck2 = aimsalgo.resampleBucket(bck, trm, inv_trm, output_vs)
    # bck2 = aimsalgo.transformBucketDirect(bck, tr, output_vs)

    # Rebuild image from buckets
    conv = aims.Converter(intype=bck2, outtype=aims.AimsData(vol))
    conv.convert(bck2, resampled)

    # Merge images

    return resampled

