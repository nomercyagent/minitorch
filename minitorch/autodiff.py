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

    def dfs(var: Variable):
        if var.unique_id in visited:
            return
        visited.add(var.unique_id)
        if var.history is not None:
            for input_var in var.history.inputs:
                dfs(input_var)
        order.append(var)

    dfs(variable)
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
    topo = topological_sort(variable)
    node2deriv = {variable.unique_id: deriv}

    for v in reversed(topo):
        if v.unique_id not in node2deriv:
            continue

        curr_deriv = node2deriv[v.unique_id]
        deriv = v.chain_rule(curr_deriv)
        check_deriv = deepcopy(deriv)
        print(list(check_deriv))

        for child, d in deriv:
            if child.is_leaf():
                child.accumulate_derivative(d)
            else:
                if child.unique_id in node2deriv:
                    node2deriv[child.unique_id] += d
                else:
                    node2deriv[child.unique_id] = d
    # topologically_sorted_variables : Iterable[Variable] = topological_sort(variable)
    # variables_with_derivs : Dict[int, Any] = {variable.unique_id: deriv}

    # for var in topologically_sorted_variables:
    #     if var.is_leaf():
    #         var.accumulate_derivative(variables_with_derivs[var.unique_id])
    #     else:
    #         derivs_back = var.chain_rule(variables_with_derivs[var.unique_id])
    #         for scalar, der in derivs_back:
    #             if (scalar.unique_id in variables_with_derivs.keys()):
    #                 variables_with_derivs[scalar.unique_id] += der
    #             else:
    #                 variables_with_derivs[scalar.unique_id] = der


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
