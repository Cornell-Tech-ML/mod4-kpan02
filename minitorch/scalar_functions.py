from __future__ import annotations

from typing import TYPE_CHECKING


import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple
    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the scalar function to the given inputs.

        This method takes in a variable number of arguments, converts them to Scalar objects if necessary, and then applies the scalar function to these inputs. It returns a new Scalar object representing the result of the function application.

        Args:
        ----
            *vals (ScalarLike): A variable number of arguments to apply the scalar function to. These can be either Scalar objects or scalar values that can be converted to Scalar objects.

        Returns:
        -------
            Scalar: A new Scalar object representing the result of applying the scalar function to the given inputs.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition.

        Args:
        ----
            ctx (Context): The context (unused for addition).
            a (float): The first number to add.
            b (float): The second number to add.

        Returns:
        -------
            float: The sum of a and b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition.

        Args:
        ----
            ctx (Context): The context (unused for addition).
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: A tuple containing the gradients with respect to each input.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for the natural logarithm function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The natural logarithm of the input.

        Note:
        ----
            This function saves the input value for use in the backward pass.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for the natural logarithm function.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        Note:
        ----
            This function uses the saved input value from the forward pass
            to compute the gradient.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# mul
class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for the multiplication function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The product of the two inputs.

        Note:
        ----
            This function saves both input values for use in the backward pass.

        """
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for the multiplication function.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to the inputs (a, b).

        Note:
        ----
            This function uses the saved input values from the forward pass
            to compute the gradients.

        """
        (a, b) = ctx.saved_values
        grad_a = b * d_output
        grad_b = a * d_output
        return grad_a, grad_b


# inv
class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for the inverse function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The inverse of the input (1/a).

        Note:
        ----
            This function saves the input value for use in the backward pass.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for the inverse function.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        Note:
        ----
            This function uses the saved input value from the forward pass
            to compute the gradient.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for the negation function.

        Args:
        ----
           ctx (Context): The context (unused in this function).
           a (float): The input value.

        Returns:
        -------
           float: The negation of the input (-a).

        Note:
        ----
           This function does not need to save any values for the backward pass.

        """
        return float(operators.neg(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for the negation function.

        Args:
        ----
            ctx (Context): The context (unused in this function).
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        Note:
        ----
            The gradient for negation is always -1 times the output gradient.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function
    This class implements the sigmoid activation function, which maps any input
    to a value between 0 and 1.
    """

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The sigmoid of the input.

        Note:
        ----
            This function saves the sigmoid result for use in the backward pass.

        """
        result = operators.sigmoid(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        Note:
        ----
            This function uses the saved sigmoid result from the forward pass
            to compute the gradient efficiently.

        """
        (a,) = ctx.saved_values
        return a * (1 - a) * d_output


class ReLU(ScalarFunction):
    """ReLU (Rectified Linear Unit) function $f(x) = max(0, x)$

    This class implements the ReLU activation function, which returns the input
    if it's positive, and 0 otherwise.
    """

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for the ReLU function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The ReLU of the input.

        Note:
        ----
            This function saves the input value for use in the backward pass.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for the ReLU function.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        Note:
        ----
            This function uses the saved input value from the forward pass
            to compute the gradient. The gradient is d_output if the input was positive,
            and 0 otherwise.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$

    This class implements the exponential function, which returns e raised to the power of the input.
    """

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The exponential of the input (e^a).

        Note:
        ----
            This function saves the input value for use in the backward pass.

        """
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        Note:
        ----
            This function uses the saved input value from the forward pass
            to compute the gradient. The gradient is d_output * e^a, where
            a is the input value saved during the forward pass.

        """
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class LT(ScalarFunction):
    """Less than function $f(x, y) = 1 if x < y else 0$

    This class implements the less than comparison function, which returns 1 if the first input
    is less than the second input, and 0 otherwise.
    """

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for the less than function.

        Args:
        ----
            ctx (Context): The context (unused for this operation).
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: 1.0 if a < b, else 0.0.

        Note:
        ----
            This function does not save any values for the backward pass
            as the gradient is always zero for both inputs.

        """
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for the less than function.

        Args:
        ----
            ctx (Context): The context (unused for this operation).
            d_output (float): The derivative of the output (unused for this operation).

        Returns:
        -------
            Tuple[float, float]: A tuple of zeros representing the gradients with respect to both inputs.

        Note:
        ----
            The gradient for the less than function is always zero for both inputs
            because the function is not differentiable at the point where a = b,
            and has a constant value (either 0 or 1) everywhere else.

        """
        return 0, 0


class EQ(ScalarFunction):
    """Equality function $f(x, y) = 1 if x == y else 0$

    This class implements the equality comparison function, which returns 1 if the two inputs
    are equal, and 0 otherwise.
    """

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for the equality function.

        Args:
        ----
            ctx (Context): The context (unused for this operation).
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: 1.0 if a == b, else 0.0.

        Note:
        ----
            This function does not save any values for the backward pass
            as the gradient is always zero for both inputs.

        """
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for the equality function.

        Args:
        ----
            ctx (Context): The context (unused for this operation).
            d_output (float): The derivative of the output (unused for this operation).

        Returns:
        -------
            Tuple[float, float]: A tuple of zeros representing the gradients with respect to both inputs.

        Note:
        ----
            The gradient for the equality function is always zero for both inputs
            because the function is not differentiable. It has a constant value (either 0 or 1)
            everywhere except at the point where a = b, where it is undefined.

        """
        return 0, 0
