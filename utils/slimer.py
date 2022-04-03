from typing import Union

import torch


def union_dim(tensor: torch.Tensor, dim1: int, dim2: int) -> Union[torch.Tensor, None]:
    dim_num = tensor.dim()
    if dim1 < 0:
        dim1 += dim_num
    if dim2 < 0:
        dim2 += dim_num
    assert dim1 < dim_num and dim2 < dim_num and dim_num >= 2, "Invalid Parameter!"
    assert abs(dim1 - dim2) == 1, "Invalid dimension!"
    if dim1 > dim2:
        dim1, dim2 = (dim2, dim1)
    original_shape = tensor.size()
    target_shape = (list(original_shape[:dim1]) +
                    [original_shape[dim1] * original_shape[dim2]] +
                    list(original_shape[dim2 + 1:]))
    return tensor.contiguous().view(target_shape)


def split_dim(tensor: torch.Tensor, dim: int, len1: int, len2: int) -> torch.Tensor:
    dim_num = tensor.dim()
    original_shape = tensor.size()
    if dim < 0:
        dim += dim_num
    assert dim < dim_num and len1 * len2 == original_shape[dim], "Invalid Parameter!"
    target_shape = (list(original_shape[:dim]) +
                    [len1, len2] +
                    list(original_shape[dim + 1:]))
    return tensor.view(target_shape)


def expand_and_repeat(tensor: torch.Tensor, dim: int, times: int) -> torch.Tensor:
    new_tensor = tensor.unsqueeze(dim)
    if dim < 0:
        dim += tensor.dim() + 1
    repeats = [1 if dim != i else times for i in range(new_tensor.dim())]
    return new_tensor.repeat(repeats)