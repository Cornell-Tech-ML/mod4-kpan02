"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the Neg function.

        This function applies the negation operation to the input tensor `t1`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            t1 (Tensor): The input tensor to be negated.

        Returns:
        -------
            Tensor: The negated tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the Neg function.

        This function computes the gradient of the negation operation with respect to the input tensor `t1`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tensor: The gradient of the negation operation with respect to the input tensor `t1`.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the Inv function.

        This function applies the inverse operation to the input tensor `t1`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            t1 (Tensor): The input tensor to be inverted.

        Returns:
        -------
            Tensor: The inverted tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the Inv function.

        This function computes the gradient of the inverse operation with respect to the input tensor `t1`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tensor: The gradient of the inverse operation with respect to the input tensor `t1`.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for the Add function.

        This function adds two input tensors `t1` and `t2`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The sum of the two input tensors.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the Add function.

        This function computes the gradients of the sum operation with respect to the input tensors `t1` and `t2`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients of the sum operation with respect to the input tensors `t1` and `t2`.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for the All function.

        This function returns 1 if all elements of the input tensor `a` are true, 0 otherwise.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            a (Tensor): The input tensor.
            dim (Tensor): The dimension along which the reduction operation is performed.

        Returns:
        -------
            Tensor: The result of the reduction operation.

        """
        if int(dim.item()) != -1:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for the Mul function.

        This function computes the element-wise multiplication of two tensors `t1` and `t2`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The element-wise product of the two input tensors.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the Mul function.

        This function computes the gradient of the output with respect to the input tensors `t1` and `t2`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients of the Mul operation with respect to the input tensors `t1` and `t2`.

        """
        (t1, t2) = ctx.saved_values
        return grad_output.f.mul_zip(t2, grad_output), grad_output.f.mul_zip(
            t1, grad_output
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the Sigmoid function.

        This function computes the element-wise sigmoid of the input tensor `t1`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The element-wise sigmoid of the input tensor.

        """
        temp = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(temp)
        return temp

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the Sigmoid function.

        This function computes the gradient of the output with respect to the input tensor `t1`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tensor: The gradient of the Sigmoid operation with respect to the input tensor `t1`.

        """
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (-sigma + 1.0) * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the ReLU function.

        This function computes the element-wise ReLU of the input tensor `t1`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The element-wise ReLU of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the ReLU function.

        This function computes the gradient of the output with respect to the input tensor `t1`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tensor: The gradient of the ReLU operation with respect to the input tensor `t1`.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the Log function.

        This function computes the element-wise natural logarithm of the input tensor `t1`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The element-wise natural logarithm of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the Log function.

        This function computes the gradient of the output with respect to the input tensor `t1`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tensor: The gradient of the Log operation with respect to the input tensor `t1`.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the Exp function.

        This function computes the element-wise exponential of the input tensor `t1`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The element-wise exponential of the input tensor.

        """
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the Exp function.

        This function computes the gradient of the output with respect to the input tensor `t1`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tensor: The gradient of the Exp operation with respect to the input tensor `t1`.

        """
        (a,) = ctx.saved_values
        return grad_output.f.mul_zip(a, grad_output)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for the Sum function.

        This function computes the sum of the input tensor `t1` along the dimension `dim`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            t1 (Tensor): The input tensor.
            dim (Tensor): The dimension along which the reduction operation is performed.

        Returns:
        -------
            Tensor: The result of the reduction operation.

        """
        ctx.save_for_backward(t1._tensor.shape, dim.shape)
        if int(dim.item()) == -1:
            return t1.f.add_reduce(
                t1.contiguous().view(int(operators.prod(t1.shape))), 0
            )
        else:
            return t1.f.add_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the Sum function.

        This function computes the gradient of the output with respect to the input tensor `t1` and the dimension `dim`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients of the Sum operation with respect to the input tensor `t1` and the dimension `dim`.

        """
        (shape, dimshape) = ctx.saved_values
        return grad_output.f.add_zip(zeros(shape), grad_output), 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for the LT function.

        This function computes the element-wise less than operation between two tensors `t1` and `t2`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The result of the element-wise less than operation.

        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the LT function.

        This function computes the gradient of the output with respect to the input tensors `t1` and `t2`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients of the LT operation with respect to the input tensors `t1` and `t2`.

        """
        (t1, t2) = ctx.saved_values
        return zeros(t1), zeros(t2)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for the EQ function.

        This function computes the element-wise equality operation between two tensors `t1` and `t2`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The result of the element-wise equality operation.

        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the EQ function.

        This function computes the gradient of the output with respect to the input tensors `t1` and `t2`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients of the EQ operation with respect to the input tensors `t1` and `t2`.

        """
        (t1, t2) = ctx.saved_values
        return zeros(t1), zeros(t2)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for the IsClose function.

        This function computes the element-wise closeness operation between two tensors `t1` and `t2` within a specified tolerance.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The result of the element-wise closeness operation.

        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.is_close_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the IsClose function.

        This function computes the gradient of the output with respect to the input tensors `t1` and `t2`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients of the IsClose operation with respect to the input tensors `t1` and `t2`.

        """
        (t1, t2) = ctx.saved_values
        return zeros(t1), zeros(t2)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Forward pass for the Permute function.

        This function permutes the dimensions of the input tensor `a` according to the specified order `order`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            a (Tensor): The input tensor to be permuted.
            order (Tensor): The tensor specifying the new order of dimensions.

        Returns:
        -------
            Tensor: The permuted tensor.

        """
        # print("what",t1.shape)
        ctx.save_for_backward(order)
        return a._new(a._tensor.permute(*[int(order[i]) for i in range(order.size)]))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the Permute function.

        This function computes the gradient of the output with respect to the input tensor `t1`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tensor: The gradient of the Permute operation with respect to the input tensor `t1`.

        """
        order: Tensor = ctx.saved_tensors[0]
        order2: List[int] = [
            a[0]
            for a in sorted(
                enumerate([order[i] for i in range(order.size)]), key=lambda a: a[1]
            )
        ]
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Forward pass for the View function.

        This function reshapes the input tensor `a` to match the specified shape `shape`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            a (Tensor): The input tensor to be reshaped.
            shape (Tensor): The tensor specifying the new shape.

        Returns:
        -------
            Tensor: The reshaped tensor.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the View function.

        This function computes the gradient of the output with respect to the input tensor `a`.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output with respect to the input.

        Returns:
        -------
            Tuple[Tensor, float]: The gradient of the View operation with respect to the input tensor `a`, and a scalar value indicating the gradient of the operation with respect to the shape tensor.

        """
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Computes the central difference approximation of the derivative of a function `f` with respect to the `arg`-th argument at the specified `ind` index.

    This function is used to approximate the derivative of a function `f` with respect to one of its arguments using the central difference method. It calculates the difference quotient between the function values at `x + epsilon` and `x - epsilon` for the specified argument and index, divided by `2 * epsilon`.

    Args:
    ----
        f: The function for which to compute the derivative.
        *vals: The values of the arguments to the function.
        arg: The index of the argument with respect to which the derivative is computed.
        epsilon: A small value used to compute the difference quotient.
        ind: The index within the `arg`-th argument at which the derivative is computed.

    Returns:
    -------
        float: The central difference approximation of the derivative of `f` with respect to the `arg`-th argument at the specified `ind` index.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
