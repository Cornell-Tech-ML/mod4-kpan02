from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Returns True if this variable is a constant (i.e., has no history).

        A constant variable is one that is not part of the computation graph
        and does not require gradient computation.

        Returns
        -------
            bool: True if the variable is constant, False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns an iterable of the parent variables of this variable.

        This property is used to traverse the computation graph during backpropagation.
        It returns the input variables that were used to compute this variable.

        Returns:
        -------
            Iterable[Variable]: An iterable of the parent variables.

        Raises:
        ------
            AssertionError: If the variable has no history (i.e., is a constant or leaf node).

        Note:
        ----
            This property assumes that the variable has a history. It should only be
            called on non-leaf, non-constant variables.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Implements the chain rule for backpropagation.

        This method computes the gradients with respect to the input variables
        using the chain rule of calculus. It combines the gradient of the output
        with respect to this variable (d_output) with the local gradients computed
        by the backward function of the last operation.

        Args:
        ----
            d_output (Any): The gradient of the final output with respect to this variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple contains:
                - A parent Variable
                - The gradient of the output with respect to that parent Variable

        Note:
        ----
            This method should only be called on non-leaf, non-constant variables.
            It assumes that the variable has a history with a last function and context.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        result = []
        i = h.last_fn._backward(h.ctx, d_output)
        parents = self.parents
        for j, parent in zip(i, parents):
            if not parent.is_constant():
                result.append((parent, j))
        return result

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Less than comparison function.

        Args:
        ----
            b (ScalarLike): The value to compare against.

        Returns:
        -------
            Scalar: A new Scalar with value 1.0 if self < b, else 0.0.

        This method applies the LT (Less Than) function to compare
        the current Scalar with the given value b. It returns a new
        Scalar representing the result of the comparison.

        """
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Greater than comparison function.

        Args:
        ----
            b (ScalarLike): The value to compare against.

        Returns:
        -------
            Scalar: A new Scalar with value 1.0 if self > b, else 0.0.

        This method applies the LT (Less Than) function to b and self
        to implement the greater than comparison. It returns a new
        Scalar representing the result of the comparison.

        """
        return LT.apply(b, self)

    def __add__(self, b: ScalarLike) -> Scalar:
        """Addition operation.

        Args:
        ----
            b (ScalarLike): The value to add to this Scalar.

        Returns:
        -------
            Scalar: A new Scalar representing the sum of this Scalar and b.

        This method applies the Add function to combine the current Scalar
        with the given value b. It returns a new Scalar representing the
        result of the addition.

        """
        return Add.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtraction operation.

        Args:
        ----
            b (ScalarLike): The value to subtract from this Scalar.

        Returns:
        -------
            Scalar: A new Scalar representing the difference between this Scalar and b.

        This method implements subtraction by adding the negation of b to self.
        It uses the Add and Neg functions to perform the operation.

        """
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        """Negation operation.

        Returns
        -------
            Scalar: A new Scalar representing the negation of this Scalar.

        This method applies the Neg function to the current Scalar,
        returning a new Scalar that represents its negation.

        """
        return Neg.apply(self)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Equality comparison operation.

        Args:
        ----
            b (ScalarLike): The value to compare with this Scalar.

        Returns:
        -------
            Scalar: A new Scalar representing the result of the equality comparison.

        This method applies the EQ (Equality) function to compare the current Scalar
        with the given value b. It returns a new Scalar that is 1.0 if the values are equal,
        and 0.0 otherwise.

        """
        return EQ.apply(self, b)

    def relu(self) -> Scalar:
        """Rectified Linear Unit (ReLU) operation.

        Returns
        -------
            Scalar: A new Scalar representing the result of applying ReLU to this Scalar.

        This method applies the ReLU function to the current Scalar,
        returning a new Scalar that represents max(0, x) where x is the value of this Scalar.

        """
        return ReLU.apply(self)

    def log(self) -> Scalar:
        """Natural logarithm operation.

        Returns
        -------
            Scalar: A new Scalar representing the natural logarithm of this Scalar.

        This method applies the natural logarithm function to the current Scalar,
        returning a new Scalar that represents the natural logarithm of the value of this Scalar.

        """
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Exponential operation.

        Returns
        -------
            Scalar: A new Scalar representing the exponential of this Scalar.

        This method applies the Exp function to the current Scalar,
        returning a new Scalar that represents the exponential of the value of this Scalar.

        """
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Sigmoid operation.

        Returns
        -------
            Scalar: A new Scalar representing the result of applying the sigmoid function to this Scalar.

        This method applies the sigmoid function to the current Scalar,
        returning a new Scalar that represents 1 / (1 + exp(-x)) where x is the value of this Scalar.

        """
        return Sigmoid.apply(self)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks the derivative of a given function at specific scalar arguments.

    This function verifies that the derivative of a given function `f` at specific scalar arguments `scalars` matches the expected derivative value computed using central difference. It asserts that the derivative computed using autodiff matches the expected derivative within a certain tolerance.

    Args:
    ----
        f (function): The function to check the derivative for.
        scalars (Scalar): The scalar arguments to evaluate the function and its derivative at.

    Raises:
    ------
        AssertionError: If the computed derivative does not match the expected derivative within the specified tolerance.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
