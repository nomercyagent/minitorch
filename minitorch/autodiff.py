from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from copy import deepcopy

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    args_eps = list(vals)
    args_eps[arg] += epsilon
    return (f(*args_eps) - f(*vals)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    visited = set()
    order = []

    def dfs(v: Variable):
        if v.is_constant():
            return
        visited.add(v.unique_id)
        if v.is_leaf():
            order.append(v)
            return
        its = v.history.inputs
        for i in its:
            if i.unique_id in visited:
                continue
            dfs(i)
        order.append(v)
    dfs(variable)
    order = order[::-1]
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    order = topological_sort(variable)
    d = dict()
    d[variable.unique_id] = deriv
    for v in order:
        # если лист, накапливаем производную
        if v.is_leaf():
            v.accumulate_derivative(d[v.unique_id])
        else:
            # иначе проходим по цепному правилу
            props = v.chain_rule(d[v.unique_id])
            # накапливаем производные
            for i, to_add in props:
                if i.unique_id in d.keys():
                    d[i.unique_id] += to_add
                else:
                    d[i.unique_id] = to_add


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
