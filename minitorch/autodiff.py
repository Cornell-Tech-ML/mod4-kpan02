from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


## Task 1.1
# Central Difference calculation
def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f: arbitrary function from n-scalar args to one value
        *vals: n-float values $x_0 \ldots x_{n-1}$
        arg: the number $i$ of the arg to compute the derivative
        epsilon: a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # ASSIGN1.1
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)
    # END ASSIGN1.1


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the output with respect to this variable.

        Args:
        ----
            x (Any): The derivative to accumulate.

        Returns:
        -------
            None

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for this variable.

        This property is used to identify each variable uniquely within the computation graph.

        Returns
        -------
            int: A unique identifier for this variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Check if it's leaf or not

        Returns
        -------
            bool: returns if it's leaf

        """
        ...

    def is_constant(self) -> bool:
        """Returns True if this variable is a constant (i.e., has no history).

        A constant variable is one that is not part of the computation graph
        and does not require gradient computation.

        Returns
        -------
            bool: True if the variable is constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns parents

        Returns
        -------
            Iterable["Variable"]: return all the parents

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Returns chain rules

        Args:
        ----
            d_output (Any): last derivative

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: Returns chain rules

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # ASSIGN1.4
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for p in var.parents:
                if not p.is_constant():
                    visit(p)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order
    # END ASSIGN1.4


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Performs backpropagation on the computation graph starting from the given variable.

    Args:
    ----
        variable (Variable): The variable to start backpropagation from.
        deriv (Any): The derivative to start backpropagation with.

    Returns:
    -------
        None: This function modifies the variables in the computation graph in-place.

    """
    # ASSIGN1.4
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d
    # END ASSIGN1.4


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors for backward pass."""
        return self.saved_values
