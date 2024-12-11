"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None

        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets the requires_grad attribute of the tensor.

        This method sets the requires_grad attribute of the tensor to the specified value. If requires_grad is True, the tensor will track its gradients during the forward pass, and gradients will be accumulated during the backward pass. If requires_grad is False, the tensor will not track gradients.

        Args:
        ----
            x (bool): The value to set requires_grad to.

        Returns:
        -------
            None: This method modifies the tensor in-place.

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Checks if the tensor requires gradients.

        This method checks if the tensor has gradients enabled. If the tensor requires gradients, it means that the tensor is part of the computation graph and its gradients will be computed during the backward pass.

        Returns
        -------
            bool: True if the tensor requires gradients, False otherwise.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        """Sets the backend of the tensor to the specified backend.

        This method updates the backend of the tensor to the provided backend. If the new backend is CUDA, it also moves the tensor data to the CUDA device.

        Args:
        ----
            backend (TensorBackend): The new backend to set for the tensor.

        """
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        """Creates a new tensor from the given tensor data.

        This method initializes a new tensor with the provided tensor data, including the storage, shape, and strides. It also sets the backend of the new tensor to the backend of the current tensor.

        Args:
        ----
            tensor_data (TensorData): The tensor data to use for the new tensor.

        Returns:
        -------
            Tensor: A new tensor created from the given tensor data.

        """
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor filled with zeros of the specified shape or the shape of the current tensor if no shape is provided.

        Args:
        ----
            shape (Optional[UserShape], optional): The shape of the tensor to be created. Defaults to None.

        Returns:
        -------
            Tensor: A tensor filled with zeros of the specified shape or the shape of the current tensor.

        """

        def zero(shape: UserShape) -> Tensor:
            """Creates a tensor filled with zeros of the specified shape.

            Args:
            ----
                shape (UserShape): The shape of the tensor to be created.

            Returns:
            -------
                Tensor: A tensor filled with zeros of the specified shape.

            """
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the tensor is a constant.

        This method checks if the tensor is a constant by verifying if it has a history of operations. If the tensor does not have a history, it is considered a constant.

        Returns
        -------
            bool: True if the tensor is a constant, False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns an iterable of the parent variables of this tensor.

        This method returns an iterable of the parent variables that contributed to the creation of this tensor. It is used to traverse the computation graph and compute gradients during backpropagation.

        Returns
        -------
            Iterable[Variable]: An iterable of the parent variables of this tensor.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the gradients of the output with respect to the input using the chain rule.

        This method applies the chain rule to compute the gradients of the output with respect to the input. It is a key component of backpropagation, which is essential for training neural networks.

        Args:
        ----
            d_output (Any): The derivative of the output with respect to the input.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple contains a variable and its corresponding derivative.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)

        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Computes the gradients of the output with respect to the input using backpropagation.

        This method initiates the backpropagation process to compute the gradients of the output with respect to the input. It is a key component of automatic differentiation, which is essential for training neural networks.

        Args:
        ----
            grad_output (Optional[Tensor], optional): The gradient of the output with respect to the input. Defaults to None.

        Raises:
        ------
            ValueError: If `grad_output` is not provided and the tensor is not scalar.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        """Element-wise division operation.

        This method performs element-wise division of the current tensor by another tensor or a scalar value. It supports broadcasting for tensors with different shapes.

        Args:
        ----
            b (TensorLike): The tensor or scalar value by which to divide the current tensor.

        Returns:
        -------
            Tensor: The result of the element-wise division.

        """
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        """Right division operation.

        This method performs element-wise division of a scalar or tensor `b` by the current tensor. It supports broadcasting for tensors with different shapes.

        Args:
        ----
            b (TensorLike): The scalar or tensor value by which to divide the current tensor.

        Returns:
        -------
            Tensor: The result of the element-wise division.

        """
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    @property
    def dims(self) -> int:
        """Returns the number of dimensions in the tensor.

        This method returns the number of dimensions in the tensor, which is the length of the shape tuple.

        Returns
        -------
            int: The number of dimensions in the tensor.

        """
        return self._tensor.dims

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor.

        This method returns the total number of elements in the tensor, which is the product of all the dimensions.

        Returns
        -------
            int: The total number of elements in the tensor.

        """
        return self._tensor.size

    # Functions
    # TODO: Implement for Task 2.3.

    def __add__(self, a: TensorLike) -> Tensor:
        """Element-wise addition of two tensors.

        This method performs element-wise addition of the current tensor with another tensor or a scalar value. It supports broadcasting for tensors with different shapes.

        Args:
        ----
            a (TensorLike): The tensor or scalar value to be added to the current tensor.

        Returns:
        -------
            Tensor: The result of the element-wise addition.

        """
        return Add.apply(self, self._ensure_tensor(a))

    def __sub__(self, a: TensorLike) -> Tensor:
        """Element-wise subtraction of two tensors.

        This method performs element-wise subtraction of the current tensor with another tensor or a scalar value. It supports broadcasting for tensors with different shapes.

        Args:
        ----
            a (TensorLike): The tensor or scalar value to be subtracted from the current tensor.

        Returns:
        -------
            Tensor: The result of the element-wise subtraction.

        """
        return Add.apply(self, Neg.apply(self._ensure_tensor(a)))

    def __mul__(self, a: TensorLike) -> Tensor:
        """Element-wise multiplication of two tensors.

        This method performs element-wise multiplication of the current tensor with another tensor or a scalar value. It supports broadcasting for tensors with different shapes.

        Args:
        ----
            a (TensorLike): The tensor or scalar value to be multiplied with the current tensor.

        Returns:
        -------
            Tensor: The result of the element-wise multiplication.

        """
        return Mul.apply(self, self._ensure_tensor(a))

    def __lt__(self, a: TensorLike) -> Tensor:
        """Element-wise less than comparison of two tensors.

        This method performs element-wise comparison of the current tensor with another tensor or a scalar value to determine if the elements of the current tensor are less than the corresponding elements of the other tensor or scalar value. It supports broadcasting for tensors with different shapes.

        Args:
        ----
            a (TensorLike): The tensor or scalar value to compare with the current tensor.

        Returns:
        -------
            Tensor: The result of the element-wise comparison, where each element is True if the corresponding element in the current tensor is less than the corresponding element in `a`, and False otherwise.

        """
        return LT.apply(self, self._ensure_tensor(a))

    def __eq__(self, a: TensorLike) -> Tensor:
        """Element-wise equality comparison of two tensors.

        This method performs element-wise comparison of the current tensor with another tensor or a scalar value to determine if the elements of the current tensor are equal to the corresponding elements of the other tensor or scalar value. It supports broadcasting for tensors with different shapes.

        Args:
        ----
            a (TensorLike): The tensor or scalar value to compare with the current tensor.

        Returns:
        -------
            Tensor: The result of the element-wise comparison, where each element is True if the corresponding element in the current tensor is equal to the corresponding element in `a`, and False otherwise.

        """
        return EQ.apply(self, self._ensure_tensor(a))

    def __gt__(self, a: TensorLike) -> Tensor:
        """Element-wise greater than comparison of two tensors.

        This method performs element-wise comparison of the current tensor with another tensor or a scalar value to determine if the elements of the current tensor are greater than the corresponding elements of the other tensor or scalar value. It supports broadcasting for tensors with different shapes.

        Args:
        ----
            a (TensorLike): The tensor or scalar value to compare with the current tensor.

        Returns:
        -------
            Tensor: The result of the element-wise comparison, where each element is True if the corresponding element in the current tensor is greater than the corresponding element in `a`, and False otherwise.

        """
        return LT.apply(self._ensure_tensor(a), self)

    def __neg__(self) -> Tensor:
        """Element-wise negation of a tensor.

        This method returns a new tensor with each element negated. It supports broadcasting for tensors with different shapes.

        Returns
        -------
            Tensor: The result of the element-wise negation, where each element is the negated value of the corresponding element in the current tensor.

        """
        return Neg.apply(self)

    def __radd__(self, a: TensorLike) -> Tensor:
        """Element-wise addition of two tensors.

        This method performs element-wise addition of the current tensor with another tensor or a scalar value. It supports broadcasting for tensors with different shapes.

        Args:
        ----
            a (TensorLike): The tensor or scalar value to add to the current tensor.

        Returns:
        -------
            Tensor: The result of the element-wise addition, where each element is the sum of the corresponding elements in the current tensor and `a`.

        """
        return Add.apply(self, self._ensure_tensor(a))

    def __rmul__(self, a: TensorLike) -> Tensor:
        """Element-wise multiplication of two tensors.

        This method performs element-wise multiplication of the current tensor with another tensor or a scalar value. It supports broadcasting for tensors with different shapes.

        Args:
        ----
            a (TensorLike): The tensor or scalar value to multiply with the current tensor.

        Returns:
        -------
            Tensor: The result of the element-wise multiplication, where each element is the product of the corresponding elements in the current tensor and `a`.

        """
        return Mul.apply(self, self._ensure_tensor(a))

    def all(self, a: Optional[TensorLike] = None) -> Tensor:
        """Element-wise logical AND operation.

        This method performs element-wise logical AND operation of the current tensor with another tensor or a scalar value. It supports broadcasting for tensors with different shapes.

        Args:
        ----
            a (TensorLike, optional): The tensor or scalar value to perform the logical AND operation with the current tensor. Defaults to None.

        Returns:
        -------
            Tensor: The result of the element-wise logical AND operation, where each element is True if the corresponding elements in the current tensor and `a` are both True, and False otherwise.

        """
        if a:
            return All.apply(self, self._ensure_tensor(a))
        else:
            return All.apply(self, Tensor.make([-1], (1,), backend=self.backend))

    def is_close(self, a: TensorLike) -> Tensor:
        """Element-wise comparison for closeness.

        This method performs element-wise comparison of the current tensor with another tensor or a scalar value to check if they are close within a certain tolerance. It supports broadcasting for tensors with different shapes.

        Args:
        ----
            a (TensorLike): The tensor or scalar value to compare with the current tensor for closeness.

        Returns:
        -------
            Tensor: The result of the element-wise comparison, where each element is True if the corresponding elements in the current tensor and `a` are close within the tolerance, and False otherwise.

        """
        return IsClose.apply(self, self._ensure_tensor(a))

    def sigmoid(self) -> Tensor:
        """Element-wise sigmoid function.

        This method computes the element-wise sigmoid of the current tensor.

        Returns
        -------
            Tensor: The result of the element-wise sigmoid operation.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Element-wise ReLU (Rectified Linear Unit) function.

        This method computes the element-wise ReLU of the current tensor.

        Returns
        -------
            Tensor: The result of the element-wise ReLU operation.

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Element-wise natural logarithm function.

        This method computes the element-wise natural logarithm of the current tensor.

        Returns
        -------
            Tensor: The result of the element-wise natural logarithm operation.

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Element-wise exponential function.

        This method computes the element-wise exponential of the current tensor.

        Returns
        -------
            Tensor: The result of the element-wise exponential operation.

        """
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Element-wise sum function.

        This method computes the element-wise sum of the current tensor.

        Args:
        ----
            dim (Optional[int]): The dimension along which the reduction operation is performed. If None, the operation is performed over all elements.

        Returns:
        -------
            Tensor: The result of the element-wise sum operation.

        """
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Computes the mean of the tensor along the specified dimension(s).

        This method calculates the mean of the elements in the current tensor along the dimension(s) specified by `dim`. If `dim` is None, the mean is computed over all elements in the tensor.

        Args:
        ----
            dim (Optional[int] or Optional[Tensor]): The dimension(s) along which the mean is computed. If None, the mean is computed over all elements.

        Returns:
        -------
            Tensor: The result of the mean operation.

        """
        if dim:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum() / self.size

    def permute(self, *order: int) -> Tensor:
        """Permutes the dimensions of the tensor.

        This method rearranges the dimensions of the current tensor according to the specified order. The order is specified as a sequence of integers, where each integer represents the new position of the corresponding dimension.

        Args:
        ----
            *order (int): The new order of the dimensions.

        Returns:
        -------
            Tensor: The tensor with permuted dimensions.

        """
        return Permute.apply(self, tensor(list(order)))

    def view(self, *shape: int) -> Tensor:
        """Reshapes the tensor to the specified shape.

        This method reshapes the current tensor to a new shape specified by the `shape` argument. The new shape must have the same total number of elements as the original tensor.

        Args:
        ----
            *shape (int): The new shape of the tensor.

        Returns:
        -------
            Tensor: The reshaped tensor.

        """
        return View.apply(self, tensor(list(shape)))

    def zero_grad_(self) -> None:
        """Clears the gradients of the tensor.

        This method sets the gradient of the tensor to None, effectively clearing any previously accumulated gradients.
        """
        self.grad = None
