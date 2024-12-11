"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The product of the two numbers.

    """
    return a * b


def id(a: float) -> float:
    """Return the input unchanged.

    Args:
    ----
        a (float): The number to return.

    Returns:
    -------
        float: The number itself.

    """
    return a


def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
    ----
        a (float): The first number to add.
        b (float): The second number to add.

    Returns:
    -------
        float: The sum of the two numbers.

    """
    return a + b


def neg(a: float) -> float:
    """Negate a number.

    Args:
    ----
        a (float): The number to negate.

    Returns:
    -------
        float: The negated number.

    """
    return -a


def lt(a: float, b: float) -> bool:
    """Check if a is less than b.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if a is less than b, False otherwise.

    """
    return a < b


def eq(a: float, b: float) -> bool:
    """Check if two numbers are equal.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if a is equal to b, False otherwise.

    """
    return a == b


def max(a: float, b: float) -> float:
    """Returns the larger of two numbers

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The maximum of the two numbers.

    """
    if a > b:
        return a
    else:
        return b


def is_close(a: float, b: float) -> bool:
    """Check if two numbers are close.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if a is close to b, False otherwise.

    """
    return abs(a - b) < 1e-2


def sigmoid(a: float) -> float:
    """Calculate the sigmoid of a number.

    Args:
    ----
        a (float): The number to calculate the sigmoid of.

    Returns:
    -------
        float: The sigmoid of the number.

    """
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        return math.exp(a) / (1.0 + math.exp(a))


def relu(a: float) -> float:
    """Calculate the ReLU of a number. ReLU is activation function which returns the number if it is greater than 0, otherwise 0.

    Args:
    ----
        a (float): The number to calculate the ReLU of.

    Returns:
    -------
        float: The ReLU of the number.

    """
    return a if a > 0 else 0.0


def log(a: float) -> float:
    """Calculate the natural logarithm of a number.

    Args:
    ----
        a (float): The number to calculate the natural logarithm of.

    Returns:
    -------
        float: The natural logarithm of the number.

    """
    return math.log(a)


def exp(a: float) -> float:
    """Calculate the exponential of a number.

    Args:
    ----
        a (float): The number to calculate the exponential of.

    Returns:
    -------
        float: The exponential of the number.

    """
    return math.exp(a)


def inv(a: float) -> float:
    """Calculate the inverse of a number.

    Args:
    ----
        a (float): The number to calculate the inverse of.

    Returns:
    -------
        float: The inverse of the number.

    """
    return 1 / a


def log_back(a: float, b: float) -> float:
    """Calculate the derivative of log of first argument times a second argument

    Args:
    ----
        a (float): The number to calculate the derivative of log of.
        b (float): The second number.

    Returns:
    -------
        float: The derivative of log of a times b.

    """
    return b / a


def inv_back(a: float, b: float) -> float:
    """Calculate the derivative of the reciprocal of the first argument times a second argument.

    Args:
    ----
        a (float): The number to calculate the derivative of the reciprocal of.
        b (float): The second number.

    Returns:
    -------
        float: The derivative of the inverse of a times b.

    """
    return -b / a**2


def relu_back(a: float, b: float) -> float:
    """Calculate the derivative of the ReLU of the first argument times a second argument.

    Args:
    ----
        a (float): The number to calculate the derivative of the ReLU of.
        b (float): The second number.

    Returns:
    -------
        float: The derivative of the ReLU of a times b.

    """
    if a > 0:
        return b
    else:
        return 0.0


def map(fn: Callable[[float], float], iter: Iterable[float]) -> Iterable[float]:
    """Apply a function to each element of an iterable.

    Args:
    ----
        fn (Callable[[float], float]): function to apply to each element of the iterable
        iter (Iterable[float]): iterable to apply the function to

    Returns:
    -------
        Iterable[float]: iterable with the function applied to each element

    """
    return [fn(x) for x in iter]


def zipWith(
    fn: Callable[[float, float], float], iter1: Iterable[float], iter2: Iterable[float]
) -> Iterable[float]:
    """Apply a function to each pair of elements from two iterables.

    Args:
    ----
        fn (Callable[[float, float], float]): function to apply to each pair of elements
        iter1 (Iterable[float]): first iterable to apply the function to
        iter2 (Iterable[float]): second iterable to apply the function to

    Returns:
    -------
        Iterable[float]: iterable with the function applied to each pair of elements

    """
    result = []
    list1 = list(iter1)
    list2 = list(iter2)
    n = len(list1)
    for i in range(n):
        result.append(fn(list1[i], list2[i]))
    return result


def reduce(
    fn: Callable[[float, float], float], iter: Iterable[float], init: float
) -> float:
    """Apply a function to each element of an iterable and reduce the result to a single value.

    Args:
    ----
        fn (Callable[[float, float], float]): function to apply to each pair of elements
        iter (Iterable[float]): iterable to apply the function to
        init (float): initial value to apply the function to

    Returns:
    -------
        float: value with the function applied to each element of the iterable

    """
    for x in iter:
        init = fn(init, x)
    return init


def negList(iter: Iterable[float]) -> Iterable[float]:
    """Negate a list.

    Args:
    ----
        iter (Iterable[float]): list to negate

    Returns:
    -------
        Iterable[float]: negated list

    """
    return [neg(x) for x in iter]


def addLists(iter1: Iterable[float], iter2: Iterable[float]) -> Iterable[float]:
    """Add two lists.

    Args:
    ----
        iter1 (Iterable[float]): first list to add
        iter2 (Iterable[float]): second list to add

    Returns:
    -------
        Iterable[float]: list with the two lists added

    """
    return zipWith(add, iter1, iter2)


def sum(iter: Iterable[float]) -> float:
    """Sum a list.

    Args:
    ----
        iter (Iterable[float]): list to sum

    Returns:
    -------
        float: sum of the list

    """
    return reduce(add, iter, 0)


def prod(iter: Iterable[float]) -> float:
    """Calculate the product of all elements in a list.

    Args:
    ----
        iter (Iterable[float]): list to take the product of

    Returns:
    -------
        float: product of the list

    """
    return reduce(mul, iter, 1)
