import torch
import torch.nn as nn


def get_conv(feature_dim=2, **kwargs):
    assert feature_dim in [2, 3], "only supports 2D and 3D data"
    if feature_dim == 2:
        return nn.Conv2d(**kwargs)
    else:
        return nn.Conv3d(**kwargs)


def get_pool(feature_dim=2, **kwargs):
    assert feature_dim in [2, 3], "only supports 2D and 3D data"
    if feature_dim == 2:
        return nn.MaxPool2d(**kwargs)
    else:
        return nn.MaxPool3d(**kwargs)


def get_bn(feature_dim=2, **kwargs):
    assert feature_dim in [2, 3], "only supports 2D and 3D data"
    if feature_dim == 2:
        return nn.BatchNorm2d(**kwargs)
    else:
        return nn.BatchNorm3d(**kwargs)


def get_conv_transpose(feature_dim=2, **kwargs):
    assert feature_dim in [2, 3], "only supports 2D and 3D data"
    if feature_dim == 2:
        return nn.ConvTranspose2d(**kwargs)
    else:
        return nn.ConvTranspose3d(**kwargs)


def auto_crop(enc_feature, dec_feature):
    """
    Center crop "enc_feature" so that it can be concatenated to "dec_feature". Note we assume "enc_feature" has larger
    spatial dimension because of rounding.

    Parameters
    ----------
    enc_feature: torch.Tensor
        Tensor on the encoding pathway of shape (B, C, ...) (2D or 3D)
    dec_feature: torch.Tensor
        Tensor on the decoding pathway of shape (B, C, ...) (2D or 3D)

    Returns
    -------
    torch.Tensor, torch.Tensor
        center-cropped "enc_feature", "dec_feature"
    """
    assert enc_feature.ndim == dec_feature.ndim, "enc and dec features should have same number of dimensions"
    if enc_feature.shape == dec_feature.shape:
        return enc_feature, dec_feature

    enc_feature_shape, dec_feature_shape = enc_feature.shape[2:], dec_feature.shape[2:]
    for enc_shape, dec_shape in zip(enc_feature_shape, dec_feature_shape):
        assert enc_shape >= dec_shape, "encoded feature should have larger spatial dimension"

    enc_feature_cropped = torch.empty_like(dec_feature)
    starting_pos = [(enc_feature_shape[i] - dec_feature_shape[i]) // 2 for i in range(len(enc_feature_shape))]
    if len(enc_feature_shape) == 2:
        enc_feature_cropped = enc_feature[...,
                              starting_pos[0] : starting_pos[0] + dec_feature_shape[0],
                              starting_pos[1] : starting_pos[1] + dec_feature_shape[1]]
    else:
        enc_feature_cropped = enc_feature[...,
                              starting_pos[0]: starting_pos[0] + dec_feature_shape[0],
                              starting_pos[1]: starting_pos[1] + dec_feature_shape[1],
                              starting_pos[2]: starting_pos[2] + dec_feature_shape[2]]

    return enc_feature_cropped, dec_feature


def concat_enc_dec(enc, dec):
    assert enc.shape == dec.shape, "'enc' and 'dec' should have the same shape"

    # concatenate along the feature dimension
    return torch.cat([enc, dec], dim=1)
