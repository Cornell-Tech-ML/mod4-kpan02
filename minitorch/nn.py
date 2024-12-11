from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw

    tiled = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    tiled = tiled.permute(0, 1, 2, 4, 3, 5)

    tiled = tiled.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    tiled, new_height, new_width = tile(input, kernel)
    pooled = tiled.sum(dim=4) / tiled.shape[-1]
    return pooled.view(tiled.shape[0], tiled.shape[1], new_height, new_width)


# TODO: Implement for Task 4.4.
max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Returns a tensor of the same shape as input with 1s in positions where the maximum value
    occurs along the given dimension and 0s elsewhere.

    Args:
    ----
        input: Input tensor
        dim: Dimension to find argmax over

    Returns:
    -------
        Binary tensor with 1s at max positions

    """
    out = max_reduce(input, dim)
    return input == out


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Computes maximum values along specified dimension."""
        dim_number = int(dim.item())
        ctx.save_for_backward(input, dim)
        return max_reduce(input, dim_number)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes gradient of max operation with respect to input."""
        input, dim = ctx.saved_values
        dim_num = int(dim.item())
        return grad_output * argmax(input, dim_num), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute max reduction along a dimension.

    Args:
    ----
        input: input tensor
        dim: dimension to reduce along

    """
    return Max.apply(input, tensor([dim]))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
    ----
        input: input tensor
        dim: dimension to apply softmax over

    """
    input = input.exp()
    return input / input.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input: input tensor
        dim: dimension to apply log-softmax over

    """
    return softmax(input, dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D
    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling
    Returns:
        Pooled tensor
    """
    tiled, new_height, new_width = tile(input, kernel)
    pooled = max_reduce(tiled, 4)
    return pooled.view(tiled.shape[0], tiled.shape[1], new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input: input tensor
        rate: probability of dropping a position (0 = no dropout)
        ignore: if True, disable dropout

    """
    if ignore or rate == 0.0:
        return input

    if rate >= 1.0:
        return input * 0.0

    # Generate random mask
    mask = rand(input.shape) > rate
    scale = 1.0 / (1.0 - rate)
    return mask * input * scale
