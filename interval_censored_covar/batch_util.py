"""Utility for forming batches of data."""

from typing import Sequence, Callable

import abc
import dataclasses
import functools

import numpy as np

import jax
import chex


class Reducer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *pytrees: chex.ArrayTree) -> chex.ArrayTree:
        pass


class ArrayReducer(Reducer):
    """Reducer for arrays."""
    @abc.abstractmethod
    def __call__(self, *arrays: chex.Array) -> chex.Array:
        pass


def pad_to_length(x: chex.Array, length: int, fill_value=-1) -> chex.Array:
    """Pad to length with fill_value."""
    assert x.shape[0] <= length
    return np.pad(x, ((0, length - x.shape[0]), ) + ((0, 0), ) * (x.ndim - 1),
                  constant_values=fill_value)


@dataclasses.dataclass
class PadMultiple(ArrayReducer):
    """Pad to multiple of length with fill_value."""
    pad_multiple: int = 1
    fill_value: chex.Scalar = -1

    def __call__(self, *arrays: chex.Array) -> chex.Array:
        """Pad to length with fill_value."""
        pad_multiple = self.pad_multiple
        if pad_multiple > 0:
            max_len = max(a.shape[0] for a in arrays)
            max_len = (
                (max_len + pad_multiple - 1) // pad_multiple) * pad_multiple
            padded_arrays: Sequence[chex.Array] = [
                pad_to_length(a, max_len, fill_value=self.fill_value)
                for a in arrays
            ]
        else:
            padded_arrays = arrays
        return np.stack(padded_arrays)


@dataclasses.dataclass
class ReduceScalar(ArrayReducer):
    reduce_op: Callable[[chex.Array], chex.Scalar] = np.max

    def __call__(self, *arrays: chex.Array) -> chex.Scalar:
        return self.reduce_op(np.stack(arrays))


MaxScalar = functools.partial(ReduceScalar, reduce_op=np.max)
MinScalar = functools.partial(ReduceScalar, reduce_op=np.min)
MeanScalar = functools.partial(ReduceScalar, reduce_op=np.mean)


def tree_batch_reduce(op_tree: chex.ArrayTree,
                      input_data: Sequence[chex.ArrayTree]) -> chex.ArrayTree:
    """Merge multiple pytrees into a single batched pytree.

  Args:
    op_tree: A pytree of Reducer objects at the leaves.
    input_data: A sequence of pytrees with of the same structure. The structure
      must have op_tree as a prefix.

  Returns:
    A pytree with the same structure as input_data[*], but with each leaf
    reduced by the corresponding Reducer in `op_tree`.
  """
    batch_size = len(input_data)

    def process_subtree(reduce_op: Reducer, *subtrees):
        assert len(subtrees) == batch_size
        if isinstance(reduce_op, ArrayReducer):
            return jax.tree_util.tree_map(reduce_op, *subtrees)
        else:
            return reduce_op(*subtrees)

    ret = jax.tree_util.tree_map(process_subtree,
                                 op_tree,
                                 *input_data,
                                 is_leaf=lambda x: isinstance(x, Reducer))

    return ret
