"""Interval-censored data for independent setting."""

from typing import (Optional, NamedTuple, Sequence, Tuple, Callable, Dict,
                    Union, List)

import collections
import math
import functools
import dataclasses

import numpy as np
import numba

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.ops
import jax.scipy.special
import jax.nn
import jax.lax

import jax.flatten_util
import jax.tree_util
import jax.debug

import jaxopt.linear_solve

import jax_newton_raphson as jnr

import simpleeval

from interval_censored_covar import batch_util

T1Predictor = Callable[[chex.Array], chex.Array]

# Define time1 predictors (identity, log, binary)
time1_identity_predictor: T1Predictor = lambda t: t
time1_log_predictor: T1Predictor = jnp.log
time1_binary_predictor: T1Predictor = lambda t: jnp.where(t > 10, 1, 0)


# Define generated data class
@chex.dataclass
class IntervalCensoredData:
    """Interval-censored data."""
    sample_size: int
    max_cluster_size: int

    # These fields all have shape (episode_size,)
    feature_t2: chex.Array
    feature_t1: chex.Array
    lb1: chex.Array
    rb1: chex.Array
    lb2: chex.Array
    rb2: chex.Array
    cluster_id: chex.Array
    observed: chex.Array

    @property
    def cluster_mask(self) -> chex.Array:
        """Returns a mask for all valid clusters."""
        return self.cluster_id >= 0


# Define data generation function
def dat_gen(key: chex.PRNGKey,
            sample_size: int,
            gamma1: Sequence[float],
            gamma2: Sequence[float],
            beta1: float,
            a1: int,
            b1: int,
            a2: int,
            b2: int,
            common_feature_bernoulli: bool,
            time1_frequency: str,
            time2_frequency: str,
            t1_predictor: T1Predictor = time1_identity_predictor,
            cluster_size_prob: Optional[chex.Array] = None,
            corr_value_times1: Optional[float] = 0.5,
            corr_value_times2: Optional[float] = 0.5,
            obs_last_t2_prob: Optional[float] = 0.3,
            return_debug_data: bool = False,
            num_common_feature: int = 0) -> IntervalCensoredData:
    """generate data"""

    # if time1_frequency == "low":
    #     constant_increment1 = 19 / 90
    #     random_increment1 = 3
    #     num_scheulding1 = 10
    # elif time1_frequency == "medium":
    #     constant_increment1 = 1 / 10
    #     random_increment1 = 1.5
    #     num_scheulding1 = 20
    # elif time1_frequency == "high":
    #     constant_increment1 = 19 / 590
    #     random_increment1 = 0.5
    #     num_scheulding1 = 60

    # if time2_frequency == "low":
    #     constant_increment2 = 19 / 90
    #     random_increment2 = 3
    #     num_scheulding2 = 10
    # elif time2_frequency == "medium":
    #     constant_increment2 = 1 / 10
    #     random_increment2 = 1.5
    #     num_scheulding2 = 20
    # elif time2_frequency == "high":
    #     constant_increment2 = 19 / 590
    #     random_increment2 = 0.5
    #     num_scheulding2 = 60

    if time1_frequency == "low":
        constant_increment1 = 19 / 90
        random_increment1 = 1.5
        num_scheulding1 = 15
    elif time1_frequency == "medium":
        constant_increment1 = 1 / 15
        random_increment1 = 1.0
        num_scheulding1 = 30
    elif time1_frequency == "high":
        constant_increment1 = 19 / 590
        random_increment1 = 0.5
        num_scheulding1 = 60

    if time2_frequency == "low":
        constant_increment2 = 19 / 90
        random_increment2 = 1.5
        num_scheulding2 = 15
    elif time2_frequency == "medium":
        constant_increment2 = 1 / 15
        random_increment2 = 1.0
        num_scheulding2 = 30
    elif time2_frequency == "high":
        constant_increment2 = 19 / 590
        random_increment2 = 0.5
        num_scheulding2 = 60

    # Generate random keys
    (key_cluser_size, tt1_key, common_feature_key, us_key,
     scheduling_times1_key, scheduling_times2_key, tt2_key,
     key_is_obs_last_t2) = jrandom.split(key, num=8)

    gamma1 = jnp.array(list(gamma1))
    gamma2 = jnp.array(list(gamma2))

    # Generate cluster sizes
    gen_clustered = cluster_size_prob is not None

    max_cluster_size = cluster_size_prob.shape[0] if gen_clustered else 1

    cluster_sizes = jrandom.choice(key_cluser_size,
                                   max_cluster_size,
                                   p=cluster_size_prob,
                                   shape=(sample_size, )) + 1
    # Generate cluster ids
    cluster_id = jnp.broadcast_to(
        jnp.arange(sample_size)[..., None], (sample_size, max_cluster_size))
    ttt = jnp.meshgrid(
        jnp.arange(max_cluster_size),
        jnp.arange(sample_size),
    )[0]
    cluster_mask = ttt < cluster_sizes[..., None]

    cluster_id = jnp.where(cluster_mask, cluster_id, -1)

    # Generate common features
    if common_feature_bernoulli:
        common_feature_original = jrandom.bernoulli(common_feature_key,
                                                    p=0.5,
                                                    shape=(sample_size,
                                                           num_common_feature))
    else:
        common_feature_original = jrandom.normal(common_feature_key,
                                                 shape=(sample_size,
                                                        num_common_feature))
    common_feature = jnp.broadcast_to(
        common_feature_original[..., None, :],
        (sample_size, max_cluster_size, num_common_feature))

    # Generate Gaussian marginals from a Gaussian copula for clustered data
    @functools.partial(jax.vmap, in_axes=(0, 0, None))
    def gen_marg_from_copula(key, mask, corr_value):
        """Return Gaussian marginals from a Gaussian copula."""
        corr_mask = jnp.outer(mask, mask)
        corr_mat = jnp.where(
            corr_mask,
            jnp.full(
                (max_cluster_size, max_cluster_size),
                corr_value).at[jnp.diag_indices(max_cluster_size)].set(1.),
            jnp.eye(max_cluster_size))
        mv_norm = jrandom.multivariate_normal(key=key,
                                              mean=jnp.zeros(max_cluster_size),
                                              cov=corr_mat)
        mv_marg = 1. - jax.scipy.stats.norm.cdf(mv_norm)
        return mv_marg

    # Generate true times
    if gen_clustered:
        tt1_raw = gen_marg_from_copula(jrandom.split(tt1_key, sample_size),
                                       cluster_mask, corr_value_times1)
        tt2_raw = gen_marg_from_copula(jrandom.split(tt2_key, sample_size),
                                       cluster_mask, corr_value_times2)
    else:
        tt1_raw = jrandom.uniform(tt1_key, (sample_size, max_cluster_size))
        tt2_raw = jrandom.uniform(tt2_key, (sample_size, max_cluster_size))

    # Generate t1 features, the covariates for true_times1
    feature_t1 = jrandom.normal(us_key,
                                shape=(sample_size,
                                       gamma1.shape[0] - num_common_feature))
    feature_t1 = jnp.concatenate([common_feature_original, feature_t1], axis=1)
    feature_t1 = jnp.broadcast_to(
        feature_t1[...,
                   None, :], (sample_size, max_cluster_size, gamma1.shape[0]))

    # Generate t2 features, the covariates for true_times2
    feature_t2 = jrandom.normal(us_key,
                                shape=(sample_size,
                                       gamma2.shape[0] - num_common_feature))
    feature_t2 = jnp.concatenate([common_feature_original, feature_t2], axis=1)
    feature_t2 = jnp.broadcast_to(
        feature_t2[...,
                   None, :], (sample_size, max_cluster_size, gamma2.shape[0]))

    true_times1 = jnp.where(
        cluster_mask,
        ((-jnp.log(tt1_raw) / jnp.exp(feature_t1 @ gamma1)))**(1 / b1) * a1,
        -jnp.inf)
    true_times2 = jnp.where(
        cluster_mask,
        ((-jnp.log(tt2_raw) /
          jnp.exp(feature_t2 @ gamma2 + beta1 * t1_predictor(true_times1))))
        **(1 / b2) * a2, -jnp.inf)

    # Generate scheduling times
    def scheduling_time(key, constant_increment, random_increment,
                        num_scheulding):
        rand_increment = jrandom.uniform(key, (num_scheulding, ),
                                         minval=0,
                                         maxval=random_increment)
        increment = rand_increment.at[1:].add(constant_increment)
        return jnp.concatenate(
            [jnp.array([0]),
             jnp.cumsum(increment),
             jnp.array([jnp.inf])])

    scheduling_times1 = scheduling_time(scheduling_times1_key,
                                        constant_increment1, random_increment1,
                                        num_scheulding1)
    scheduling_times2 = scheduling_time(scheduling_times2_key,
                                        constant_increment2, random_increment2,
                                        num_scheulding2)

    # Generate left and right bounds
    interval_index1 = jnp.searchsorted(scheduling_times1,
                                       true_times1,
                                       side="left")
    lb1 = jnp.where(cluster_mask, scheduling_times1[interval_index1 - 1],
                    -jnp.inf)
    rb1 = jnp.where(cluster_mask, scheduling_times1[interval_index1], -jnp.inf)

    interval_index2 = jnp.searchsorted(scheduling_times2,
                                       true_times2,
                                       side="left")
    lb2 = jnp.where(cluster_mask, scheduling_times2[interval_index2 - 1],
                    -jnp.inf)
    rb2 = jnp.where(cluster_mask, scheduling_times2[interval_index2], -jnp.inf)

    last_t2_obs = jrandom.bernoulli(
        key_is_obs_last_t2, p=obs_last_t2_prob,
        shape=(sample_size, )) | (cluster_sizes <= 1)
    observed = jnp.ones_like(cluster_mask, dtype=bool)
    observed = observed.at[jnp.arange(sample_size),
                           cluster_sizes].set(last_t2_obs)

    data = IntervalCensoredData(
        sample_size=sample_size,
        max_cluster_size=max_cluster_size,
        feature_t2=feature_t2.reshape(-1, feature_t2.shape[-1]),
        feature_t1=feature_t1.reshape(-1, feature_t1.shape[-1]),
        lb1=lb1.ravel(),
        rb1=rb1.ravel(),
        lb2=lb2.ravel(),
        rb2=rb2.ravel(),
        cluster_id=cluster_id.ravel(),
        observed=observed.ravel())
    if return_debug_data:
        return (data, true_times1, true_times2, tt1_raw, tt2_raw)
    else:
        return data


def unique_times(
        input_data: IntervalCensoredData) -> Tuple[chex.Array, chex.Array]:
    """obtain unique times"""

    def compute_uts(lb, rb):
        right_censored_idxs = np.isinf(rb)
        left_censored_idxs = lb == 0
        not_censored_idxs = ~(right_censored_idxs | left_censored_idxs)
        return np.unique(
            np.concatenate([lb[not_censored_idxs], rb[not_censored_idxs]]))

    # def compute_uts(lb, rb):
    #     return np.unique(np.concatenate([lb, rb]))

    uts1 = compute_uts(input_data.lb1, input_data.rb1)
    uts2 = compute_uts(input_data.lb2, input_data.rb2)
    return uts1, uts2


# Define index pair class: i, l such that uts[l] is in the i-th episode bracket
class IndexPair2L(NamedTuple):
    """index pair for 2-level nested data"""
    time: chex.Array
    episode_bracket: chex.Array


# Define cluster index class
class ClusterIndex(NamedTuple):
    """Given an array, how to index into it to get the cluster data."""
    start_idx: chex.Array  # length: num_clusters + 1
    cluster_idxs: chex.Array  # length: roundup(array_length, max_cluster_length)
    max_cluster_length: int


def make_cluster_index(c: List, sample_size: int) -> ClusterIndex:
    c_ = np.array(c, dtype=int)
    cluster_counts = np.bincount(c_, minlength=sample_size)
    max_cluster_size = cluster_counts.max()
    cluster_start_idx = np.concatenate(
        (np.array([0]), np.cumsum(cluster_counts)))
    cluster_idxs = np.concatenate([
        np.argsort(c_, kind="stable"),
        np.full(max_cluster_size - cluster_counts[-1], -1)
    ])
    return ClusterIndex(
        cluster_start_idx,
        cluster_idxs,
        max_cluster_size,
    )


@numba.njit
def _ut_in_which_brackets2l(uts: chex.ArrayNumpy, cluster_ids: chex.ArrayNumpy,
                            cluster_mask: chex.ArrayNumpy,
                            lb: Optional[chex.ArrayNumpy],
                            rb: Optional[chex.ArrayNumpy]):
    """Obtain unique time where brackets return the data index for
  each unique time."""

    blen = cluster_mask.size

    a, b = [], []
    c = [0 for _ in range(0)]
    for ti, t in enumerate(uts):
        for ni in range(blen):
            if not cluster_mask[ni]:
                continue
            l = lb[ni] if lb is not None else -jnp.inf
            r = rb[ni] if rb is not None else jnp.inf
            if l < t <= r:
                a.append(ti)
                b.append(ni)
                if cluster_ids is not None:
                    c.append(cluster_ids[ni])

    return IndexPair2L(np.array(a), np.array(b)), c


def ut_in_which_brackets2l(sample_size, uts: chex.ArrayNumpy,
                           cluster_ids: chex.ArrayNumpy,
                           cluster_mask: chex.ArrayNumpy,
                           lb: Optional[chex.ArrayNumpy],
                           rb: Optional[chex.ArrayNumpy]):
    ip2l, c = _ut_in_which_brackets2l(uts, cluster_ids, cluster_mask, lb, rb)
    if cluster_ids is not None:
        return ip2l, make_cluster_index(c, sample_size)
    else:
        return ip2l


@numba.njit
def filter_unpaired_times(uts: chex.ArrayNumpy, cluster_mask: chex.ArrayNumpy,
                          lb: Optional[chex.ArrayNumpy],
                          rb: chex.ArrayNumpy) -> chex.ArrayNumpy:
    blen = len(cluster_mask)
    ret = []
    for t in uts:
        for bi in range(blen):
            m = cluster_mask[bi]
            (l, r) = (lb[bi] if lb is not None else -jnp.inf), rb[bi]
            if m and l < t <= r:
                ret.append(t)
                break

    return np.array(ret)


#  Define index pair class
class IndexPair3L(NamedTuple):
    """index pair for 3-level nested data"""
    idx2l: chex.Array
    time: chex.Array


@numba.njit
def _ut_in_which_brackets3l(
    uts1: chex.ArrayNumpy,
    uts2: chex.ArrayNumpy,
    cluster_ids: chex.ArrayNumpy,
    cluster_mask: chex.ArrayNumpy,
    lb1: chex.ArrayNumpy,
    rb1: chex.ArrayNumpy,
    lb2: Optional[chex.ArrayNumpy],
    rb2: chex.ArrayNumpy,
) -> IndexPair3L:
    """Obtain unique time in which brackets return the data index for
  each unique time."""

    assert len(lb1) == len(rb1) == len(rb2)
    blen = len(cluster_mask)

    r2s_iter = []
    for t1 in uts1:
        for bi in range(blen):
            if not cluster_mask[bi]:
                continue
            l1, r1, r2s = lb1[bi], rb1[bi], rb2[bi]
            l2 = lb2[bi] if lb2 is not None else -jnp.inf
            if l1 < t1 <= r1:
                ci = cluster_ids[bi] if cluster_ids is not None else None
                r2s_iter.append((l2, r2s, ci))

    a, b = [], []
    c = [0 for _ in range(0)]

    for t1ni, (l2, r2s, ci) in enumerate(r2s_iter):
        for t2i, t2 in enumerate(uts2):
            if l2 < t2 <= r2s:
                a.append(t1ni)
                b.append(t2i)
                if cluster_ids is not None:
                    c.append(ci)

    return IndexPair3L(np.array(a), np.array(b)), c


def ut_in_which_brackets3l(
    sample_size: int,
    uts1: chex.ArrayNumpy,
    uts2: chex.ArrayNumpy,
    cluster_ids: chex.ArrayNumpy,
    cluster_mask: chex.ArrayNumpy,
    lb1: chex.ArrayNumpy,
    rb1: chex.ArrayNumpy,
    lb2: Optional[chex.ArrayNumpy],
    rb2: chex.ArrayNumpy,
):
    ip3l, c = _ut_in_which_brackets3l(uts1, uts2, cluster_ids, cluster_mask,
                                      lb1, rb1, lb2, rb2)
    if cluster_ids is not None:
        return ip3l, make_cluster_index(c, sample_size)
    else:
        return ip3l


# Define preprocessed data class
@chex.dataclass
class PreprocessedData(IntervalCensoredData):
    """preprocessed data:
  - uts1: unique times 1

  - uts2: unique times 2

  - ut_index1: tuple of indices of:
    (1) unique times 1 (.time) *sorted by*
    (2) their enclosing episode brackets 1 (.episode_bracket) 

  - ut_index2: tuple of indices of :
    (1) unique times 2 (.time) *sorted by*
    (2) their enclosing episode brackets 2 (.episode_bracket)

  - ut_index2_lt_lb: tuple of indices of:
    (1) unique times 2 (.time) *sorted by*
    (2) matching episode brackets 2 (.episode_bracket) where 
        the value of unique time 2 is less than or equal to 
        episode bracket 2 left bound
  
  - ut_index2_lt_rb: tuple of indices of:
    (1) unique times 2 (.time) *sorted by*
    (2) matching brackets 2 (.episode_bracket) where 
        the value of unique time 2 is less than or equal to 
        episode bracket 2 right bound
  
  - ut_index2_lt_rb_star: tuple of indices of 
    (1) unique times 2 (.time) *sorted by*
    (2) matching brackets 2 (.episode_bracket) where
        the value of unique time 2 is less than or equal to 
        episode nbracket 2 right bound star
  
  - ut_index2_lt_rb_star3l: tuple of indices of:
    (1) combination of unique times 1 and their enclosing 
        episode brackets 1 (.idx2l) *sorted by*
    (2) matching unique times 2 (.time) whose value is less than 
        or equal to the corresponding episode's bracket 2 right bound star

  - ut_index2_bwb3l: tuple of indices of:
    (1) combination of unique times 1 and their enclosing
        episode brackets 1 (.idx2l) *sorted by*
    (2) matching unique times 2 (.time) whose value is between
        the corresponding episode's bracket 2 left bound and right bound
  
  """
    uts1: chex.Array
    uts2: chex.Array
    ut_index1: IndexPair2L
    ut_index1_cluster_index: ClusterIndex

    ut_index2: IndexPair2L

    ut_index2_lt_lb: IndexPair2L
    ut_index2_lt_rb: IndexPair2L

    ut_index2_lt_rb_star: IndexPair2L

    ut_index2_lt_lb3l: IndexPair3L
    ut_index2_lt_lb3l_cluster_index: ClusterIndex

    ut_index2_lt_rb3l: IndexPair3L
    ut_index2_lt_rb3l_cluster_index: ClusterIndex

    ut_index2_lt_rb_star3l: IndexPair3L
    ut_index2_bwb3l: IndexPair3L

    def lambda1_mask(self) -> chex.Array:
        """Returns a mask for which indices of xi are actual parameters."""
        return self.uts1 >= 0

    def lambda2_mask(self):
        """Returns a mask for which indices of lambda2 are actual parameters."""
        return self.uts2 >= 0


def preprocess_data(input_data: IntervalCensoredData) -> PreprocessedData:
    """preprocess data"""

    episode_size = input_data.feature_t2.shape[0]

    input_data = jax.tree_util.tree_map(
        lambda x: np.asarray(x)[input_data.cluster_mask]
        if np.ndim(x) > 0 and x.shape[0] == episode_size else x, input_data)

    uts1, uts2 = unique_times(input_data)
    uts1 = filter_unpaired_times(uts1, input_data.cluster_mask, input_data.lb1,
                                 input_data.rb1)
    ut_index1, ut_index1_cluster_index = ut_in_which_brackets2l(
        input_data.sample_size, uts1, input_data.cluster_id,
        input_data.cluster_mask, input_data.lb1, input_data.rb1)

    uts2 = filter_unpaired_times(uts2, input_data.cluster_mask, input_data.lb2,
                                 input_data.rb2)
    rb2_star = np.where(np.isinf(input_data.rb2), input_data.lb2,
                        input_data.rb2)
    ut_index2_lt_rb_star3l = ut_in_which_brackets3l(
        input_data.sample_size, uts1, uts2, None, input_data.cluster_mask,
        input_data.lb1, input_data.rb1, None, rb2_star)

    uts2 = np.unique(uts2[ut_index2_lt_rb_star3l.time])

    ut_index2 = ut_in_which_brackets2l(input_data.sample_size, uts2, None,
                                       input_data.cluster_mask, input_data.lb2,
                                       input_data.rb2)
    ut_index2_lt_lb = ut_in_which_brackets2l(input_data.sample_size, uts2,
                                             None, input_data.cluster_mask,
                                             None, input_data.lb2)
    ut_index2_lt_rb = ut_in_which_brackets2l(input_data.sample_size, uts2,
                                             None, input_data.cluster_mask,
                                             None, input_data.rb2)

    ut_index2_lt_rb_star = ut_in_which_brackets2l(input_data.sample_size, uts2,
                                                  None,
                                                  input_data.cluster_mask,
                                                  None, rb2_star)

    ut_index2_lt_lb3l, ut_index2_lt_lb3l_cluster_index = ut_in_which_brackets3l(
        input_data.sample_size, uts1, uts2, input_data.cluster_id,
        input_data.cluster_mask, input_data.lb1, input_data.rb1, None,
        input_data.lb2)

    ut_index2_lt_rb3l, ut_index32_lt_rb3l_cluster_index = ut_in_which_brackets3l(
        input_data.sample_size, uts1, uts2, input_data.cluster_id,
        input_data.cluster_mask, input_data.lb1, input_data.rb1, None,
        input_data.rb2)

    ut_index2_bwb3l = ut_in_which_brackets3l(input_data.sample_size, uts1,
                                             uts2, None,
                                             input_data.cluster_mask,
                                             input_data.lb1, input_data.rb1,
                                             input_data.lb2, rb2_star)
    return PreprocessedData(
        **input_data,
        uts1=uts1,
        uts2=uts2,
        ut_index1=ut_index1,
        ut_index1_cluster_index=ut_index1_cluster_index,
        ut_index2=ut_index2,
        ut_index2_lt_lb=ut_index2_lt_lb,
        ut_index2_lt_rb=ut_index2_lt_rb,
        ut_index2_lt_rb_star=ut_index2_lt_rb_star,
        ut_index2_lt_lb3l=ut_index2_lt_lb3l,
        ut_index2_lt_lb3l_cluster_index=ut_index2_lt_lb3l_cluster_index,
        ut_index2_lt_rb3l=ut_index2_lt_rb3l,
        ut_index2_lt_rb3l_cluster_index=ut_index32_lt_rb3l_cluster_index,
        ut_index2_lt_rb_star3l=ut_index2_lt_rb_star3l,
        ut_index2_bwb3l=ut_index2_bwb3l,
    )


@dataclasses.dataclass
class ReduceClusterIndex(batch_util.Reducer):
    """Combine ClusterIndexs into a single batch of ClusterIndex."""
    pad_multiple: int

    def __call__(self, *cluster_indexs: ClusterIndex) -> ClusterIndex:
        cluster_indexs = list(cluster_indexs)
        max_cluster_length = max(ci.max_cluster_length
                                 for ci in cluster_indexs)
        for i in range(len(cluster_indexs)):
            ci = cluster_indexs[i]
            cluster_indexs[i] = ci._replace(
                cluster_idxs=np.pad(ci.cluster_idxs, (0, max_cluster_length -
                                                      ci.max_cluster_length),
                                    'constant',
                                    constant_values=-1))
        Pad, MaxScalar = batch_util.PadMultiple, batch_util.MaxScalar  # pylint: disable=invalid-name
        return batch_util.tree_batch_reduce(
            ClusterIndex(
                Pad(self.pad_multiple),
                Pad(self.pad_multiple),
                MaxScalar(),
            ),
            cluster_indexs,
        )


def batch_preprocess_data(batch_input_data: Sequence[IntervalCensoredData],
                          pad_multiple: int = 32,
                          param_pad_multiple: int = 1) -> PreprocessedData:
    """Preprocess data and batch them by padding to the same length."""

    Pad, MaxScalar = batch_util.PadMultiple, batch_util.MaxScalar  # pylint: disable=invalid-name

    op_tree = PreprocessedData(
        sample_size=MaxScalar(),
        max_cluster_size=MaxScalar(),
        feature_t2=Pad(pad_multiple, fill_value=0),
        feature_t1=Pad(pad_multiple, fill_value=0),
        lb1=Pad(pad_multiple),
        rb1=Pad(pad_multiple),
        lb2=Pad(pad_multiple),
        rb2=Pad(pad_multiple),
        cluster_id=Pad(pad_multiple),
        observed=Pad(pad_multiple, False),
        uts1=Pad(param_pad_multiple),
        uts2=Pad(param_pad_multiple),
        ut_index1=Pad(pad_multiple),
        ut_index1_cluster_index=ReduceClusterIndex(pad_multiple),
        ut_index2=Pad(pad_multiple),
        ut_index2_lt_lb=Pad(pad_multiple),
        ut_index2_lt_rb=Pad(pad_multiple),
        ut_index2_lt_rb_star=Pad(pad_multiple),
        ut_index2_lt_lb3l=Pad(pad_multiple),
        ut_index2_lt_lb3l_cluster_index=ReduceClusterIndex(pad_multiple),
        ut_index2_lt_rb3l=Pad(pad_multiple),
        ut_index2_lt_rb3l_cluster_index=ReduceClusterIndex(pad_multiple),
        ut_index2_lt_rb_star3l=Pad(pad_multiple),
        ut_index2_bwb3l=Pad(pad_multiple),
    )

    batch_preprocessed_data = [preprocess_data(d) for d in batch_input_data]
    return batch_util.tree_batch_reduce(op_tree, batch_preprocessed_data)


class Parameters(NamedTuple):
    """Parameters for the model."""
    gamma1: chex.Array
    gamma2: chex.Scalar
    beta1: chex.Scalar
    log_lambda1: chex.Array
    log_lambda2: chex.Array

    @property
    def lambda1(self):
        return jnp.exp(self.log_lambda1)

    @property
    def lambda2(self):
        return jnp.exp(self.log_lambda2)


def safe_mul(a, b):
    """safe multiplication. If b == 0, return 0. Otherwise, return a * b."""
    return jnp.where(b == 0, 0, a * b)


def left_cumsum(x):
    return jnp.cumsum(jnp.insert(x, 0, 0))


def expected_i_given_current_estimate_pairs(
    pd: PreprocessedData,
    current_theta: Parameters,
    t1_predictor: T1Predictor = time1_identity_predictor,
):
    """compute the expected I's for all il pairs given current estimate"""

    ut1_in_predictor = t1_predictor(pd.uts1)
    import jax.numpy as jnp
    import jax.ops

    lambda1_cumsum = left_cumsum(current_theta.lambda1)
    expbz = jnp.exp(
        pd.feature_t1[pd.ut_index1.episode_bracket] @ current_theta.gamma1)
    part1_2_for_pairs1 = (
        jnp.exp(-(lambda1_cumsum[:-1][pd.ut_index1.time]) * expbz) -
        jnp.exp(-(lambda1_cumsum[1:][pd.ut_index1.time]) * expbz))

    part3_for_pairs1 = jnp.exp(
        safe_mul(
            -jnp.exp(pd.feature_t2[pd.ut_index1.episode_bracket]
                     @ current_theta.gamma2 + current_theta.beta1 *
                     ut1_in_predictor[pd.ut_index1.time]),
            jax.ops.segment_sum(
                current_theta.lambda2[pd.ut_index2_lt_lb3l.time],
                pd.ut_index2_lt_lb3l.idx2l,
                num_segments=pd.ut_index1.episode_bracket.shape[0])))

    part4_for_pairs1 = jnp.exp(
        safe_mul(
            -jnp.exp(pd.feature_t2[pd.ut_index1.episode_bracket]
                     @ current_theta.gamma2 + current_theta.beta1 *
                     ut1_in_predictor[pd.ut_index1.time]),
            jax.ops.segment_sum(
                current_theta.lambda2[pd.ut_index2_lt_rb3l.time],
                pd.ut_index2_lt_rb3l.idx2l,
                num_segments=pd.ut_index1.episode_bracket.shape[0])))

    numerator_pairs1 = (part1_2_for_pairs1 * jnp.where(
        pd.observed[pd.ut_index1.episode_bracket],
        part3_for_pairs1 - part4_for_pairs1,
        1.,
    ))

    denominator_pairs = jax.ops.segment_sum(
        numerator_pairs1,
        pd.ut_index1.episode_bracket,
        num_segments=pd.feature_t2.shape[0])[pd.ut_index1.episode_bracket]

    expected_I = jnp.where(denominator_pairs == 0, 0,
                           (numerator_pairs1 / denominator_pairs))

    return jnp.where(pd.ut_index1.time >= 0, expected_I, 0)


def expected_wi_given_current_estimate_triplets_bwb(
    pd: PreprocessedData,
    current_theta: Parameters,
    t1_predictor: T1Predictor = time1_identity_predictor,
):
    """compute the expected WI's for all ilk triplets"""

    uts1_in_predictor = t1_predictor(pd.uts1)

    unique_t1_idx_triplets_bwb = pd.ut_index1.time[pd.ut_index2_bwb3l.idx2l]
    unique_i_idx_triplets_bwb = pd.ut_index1.episode_bracket[
        pd.ut_index2_bwb3l.idx2l]

    expected_I_for_triplets_bwb = expected_i_given_current_estimate_pairs(
        pd,
        current_theta,
        t1_predictor=t1_predictor,
    )[pd.ut_index2_bwb3l.idx2l]

    part5_for_triplets_bwb = jnp.exp(
        current_theta.log_lambda2[pd.ut_index2_bwb3l.time] +
        pd.feature_t2[unique_i_idx_triplets_bwb] @ current_theta.gamma2 +
        current_theta.beta1 * uts1_in_predictor[unique_t1_idx_triplets_bwb])

    sum_lambda2_for_unique_t2_within_brackets2_for_part6 = jax.ops.segment_sum(
        current_theta.lambda2[pd.ut_index2.time],
        pd.ut_index2.episode_bracket,
        num_segments=pd.feature_t2.shape[0])

    part6_for_triplets = -jnp.expm1(-jnp.exp(
        pd.feature_t2[unique_i_idx_triplets_bwb] @ current_theta.gamma2 +
        current_theta.beta1 * uts1_in_predictor[unique_t1_idx_triplets_bwb]
    ) * sum_lambda2_for_unique_t2_within_brackets2_for_part6[
        unique_i_idx_triplets_bwb])

    return jnp.where(
        pd.ut_index2_bwb3l.idx2l >= 0,
        (safe_mul(part5_for_triplets_bwb, expected_I_for_triplets_bwb) /
         part6_for_triplets),
        0,
    )


def safe_index(x, idx, fill_value=0.):
    return jnp.where(idx >= 0, x[idx], fill_value)


def q_for_estep(
    pd: PreprocessedData,
    current_theta: Parameters,
    theta: Parameters,
    t1_predictor: T1Predictor = time1_identity_predictor,
) -> float:
    """compute the Q function for the E step"""
    import jax.numpy as jnp
    uts1_in_predictor = jnp.nan_to_num(t1_predictor(pd.uts1),
                                       nan=0.,
                                       posinf=jnp.inf,
                                       neginf=-jnp.inf)
    expected_I_pairs = expected_i_given_current_estimate_pairs(
        pd, current_theta, t1_predictor=t1_predictor)
    expected_WI_triplets_bwb = expected_wi_given_current_estimate_triplets_bwb(
        pd, current_theta, t1_predictor=t1_predictor)

    unique_t1_idx_triplets_bwb = pd.ut_index1.time[pd.ut_index2_bwb3l.idx2l]
    unique_i_idx_triplets_bwb = pd.ut_index1.episode_bracket[
        pd.ut_index2_bwb3l.idx2l]

    unique_t1_idx_triplets_lt_rb_star = pd.ut_index1.time[
        pd.ut_index2_lt_rb_star3l.idx2l]
    unique_i_idx_triplets_lt_rb_star = pd.ut_index1.episode_bracket[
        pd.ut_index2_lt_rb_star3l.idx2l]

    lambda1_cumsum = left_cumsum(theta.lambda1)
    expbz = jnp.exp(pd.feature_t1[pd.ut_index1.episode_bracket] @ theta.gamma1)
    part1_2_for_pairs1 = jnp.log(
        jnp.exp(-(lambda1_cumsum[:-1][pd.ut_index1.time]) * expbz) -
        jnp.exp(-(lambda1_cumsum[1:][pd.ut_index1.time]) * expbz))

    part1_sum_pairs = jnp.sum(
        jnp.where(
            pd.ut_index1.time >= 0,
            expected_I_pairs * part1_2_for_pairs1,
            0,
        ))

    part2_segment_sum = jax.ops.segment_sum(
        expected_WI_triplets_bwb *
        (theta.log_lambda2[pd.ut_index2_bwb3l.time] +
         pd.feature_t2[unique_i_idx_triplets_bwb] @ theta.gamma2 +
         theta.beta1 * uts1_in_predictor[unique_t1_idx_triplets_bwb]),
        pd.ut_index2_bwb3l.idx2l,
        num_segments=pd.ut_index1.time.shape[0])

    part3_segment_sum = jax.ops.segment_sum(
        expected_I_pairs[pd.ut_index2_lt_rb_star3l.idx2l] *
        jnp.exp(theta.log_lambda2[pd.ut_index2_lt_rb_star3l.time] +
                pd.feature_t2[unique_i_idx_triplets_lt_rb_star] @ theta.gamma2
                + theta.beta1 *
                uts1_in_predictor[unique_t1_idx_triplets_lt_rb_star]),
        pd.ut_index2_lt_rb_star3l.idx2l,
        num_segments=pd.ut_index1.time.shape[0])

    part2_sum_pairs = jnp.sum(part2_segment_sum *
                              pd.observed[pd.ut_index1.episode_bracket])

    part3_sum_pairs = jnp.sum(part3_segment_sum *
                              pd.observed[pd.ut_index1.episode_bracket])

    # jax.debug.breakpoint()
    return (part1_sum_pairs + part2_sum_pairs -
            part3_sum_pairs) / pd.feature_t2.shape[0]


class EMResult(NamedTuple):
    """Result of the EM algorithm."""
    theta: Parameters
    m_step_status: Dict[str, int]
    e_step_status: int


def update_lambda2(
    pd: PreprocessedData,
    current_theta: Parameters,
    theta: Parameters,
    t1_predictor: T1Predictor = time1_identity_predictor,
) -> chex.Array:
    """Update lambda2 using closed form."""

    import jax.numpy as jnp
    uts1_in_predictor = jnp.nan_to_num(t1_predictor(pd.uts1),
                                       nan=0.,
                                       posinf=jnp.inf,
                                       neginf=-jnp.inf)

    expected_I_pairs = expected_i_given_current_estimate_pairs(
        pd,
        current_theta,
        t1_predictor=t1_predictor,
    )
    expected_WI_triplets_bwb = expected_wi_given_current_estimate_triplets_bwb(
        pd,
        current_theta,
        t1_predictor=t1_predictor,
    )

    unique_i_idx_triplets_bwb = pd.ut_index1.episode_bracket[
        pd.ut_index2_bwb3l.idx2l]

    unique_t1_idx_triplets_lt_rb_star = pd.ut_index1.time[
        pd.ut_index2_lt_rb_star3l.idx2l]
    unique_i_idx_triplets_lt_rb_star = pd.ut_index1.episode_bracket[
        pd.ut_index2_lt_rb_star3l.idx2l]

    numerator = jax.ops.segment_sum(expected_WI_triplets_bwb *
                                    pd.observed[unique_i_idx_triplets_bwb],
                                    pd.ut_index2_bwb3l.time,
                                    num_segments=pd.uts2.shape[0])

    denominator = jax.ops.segment_sum(safe_mul(
        jnp.exp(
            pd.feature_t2[unique_i_idx_triplets_lt_rb_star] @ theta.gamma2 +
            theta.beta1 *
            uts1_in_predictor[unique_t1_idx_triplets_lt_rb_star]),
        expected_I_pairs[pd.ut_index2_lt_rb_star3l.idx2l] *
        pd.observed[unique_i_idx_triplets_lt_rb_star]),
                                      pd.ut_index2_lt_rb_star3l.time,
                                      num_segments=pd.uts2.shape[0])

    new_log_lambda2 = jnp.where(pd.lambda2_mask(),
                                jnp.log(numerator) - jnp.log(denominator), 0)
    return jnp.where(~jnp.isfinite(new_log_lambda2), -40., new_log_lambda2)


class StaticSizes(NamedTuple):
    sample_size: int
    ut_index1_cluster_index_max_cluster_length: int
    ut_index2_lt_lb3l_cluster_index_max_cluster_length: int
    ut_index2_lt_rb3l_cluster_index_max_cluster_length: int


def _fix_sizes(pd: PreprocessedData,
               static_sizes: StaticSizes) -> PreprocessedData:
    i1 = pd.ut_index1_cluster_index._replace(
        max_cluster_length=static_sizes.
        ut_index1_cluster_index_max_cluster_length)
    i2 = pd.ut_index2_lt_lb3l_cluster_index._replace(
        max_cluster_length=static_sizes.
        ut_index2_lt_lb3l_cluster_index_max_cluster_length)
    i3 = pd.ut_index2_lt_rb3l_cluster_index._replace(
        max_cluster_length=static_sizes.
        ut_index2_lt_rb3l_cluster_index_max_cluster_length)
    return dataclasses.replace(
        pd,
        sample_size=static_sizes.sample_size,
        ut_index1_cluster_index=i1,
        ut_index2_lt_lb3l_cluster_index=i2,
        ut_index2_lt_rb3l_cluster_index=i3,
    )


LoopState = collections.namedtuple(
    "LoopState", ["theta_prev", "theta_current", "m_step_status", "step"])


def _unpack_theta(update_params, fixed_masks, theta, x_updated):
    for p in update_params:
        if p in fixed_masks:
            xp = getattr(theta, p)
            mask = fixed_masks[p]
            xp = xp.at[~mask].set(x_updated[p])
        else:
            xp = x_updated[p]
        theta = theta._replace(**{p: xp})
    return theta


def solve_theta(
    static_sizes: StaticSizes,
    pd: PreprocessedData,
    initial_guess: Parameters,
    e_step_tol=1e-3,
    m_step_tol=1e-3,
    e_step_maxiter=1000,
    m_step_maxiter=1000,
    m_step_ls_maxiter=10,
    t1_predictor: T1Predictor = time1_identity_predictor,
    update_beta_beta1_xi_include_lambda2_closed_form: bool = False,
    # solve_params: Parameters = None,
    fixed_params: Dict[str, Union[Sequence[int], bool]] = dict()):
    """solve for theta using the EM algorithm"""
    pd = _fix_sizes(
        pd,
        static_sizes,
    )
    import jax.numpy as jnp

    init_state = LoopState(
        theta_prev=jax.tree_util.tree_map(
            functools.partial(jnp.full_like, fill_value=jnp.inf),
            initial_guess),
        theta_current=initial_guess,
        m_step_status=jnp.array(0),
        step=jnp.array(0),
    )

    def cond_fn(state: LoopState):

        def to_v(theta):
            return jax.flatten_util.ravel_pytree(theta)[0]

        return ((state.m_step_status == 0) &
                (~jnp.allclose(to_v(state.theta_current),
                               to_v(state.theta_prev),
                               atol=1e-3,
                               rtol=e_step_tol)) &
                (state.step < e_step_maxiter + 1))

    def body_fn(state: LoopState) -> LoopState:

        state = state._replace(theta_prev=state.theta_current)

        all_params = ["gamma1", "gamma2", "beta1", "log_lambda1"]
        fixed_masks = {}
        update_params = []
        for p in all_params:
            if isinstance(fixed_params[p], (list, tuple)):
                if len(fixed_params[p]):
                    mask = np.zeros(getattr(state.theta_current, p).shape,
                                    dtype=bool)
                    mask[list(fixed_params[p])] = True
                    fixed_masks[p] = mask
                update_params.append(p)
            elif isinstance(fixed_params[p], bool):
                if not fixed_params[p]:
                    update_params.append(p)
            else:
                raise ValueError(f"Invalid fixed_params for {p}")
        x = {
            p: (getattr(state.theta_current, p)[fixed_masks[p]]
                if p in fixed_masks else getattr(state.theta_current, p))
            for p in update_params
        }

        x_flat, x_unraveler = jax.flatten_util.ravel_pytree(x)

        if len(x_flat) > 0:

            def coord_q(x_flat: chex.Array) -> float:
                current_theta = state.theta_current
                x_updated = x_unraveler(x_flat)
                theta = _unpack_theta(update_params, fixed_masks,
                                      current_theta, x_updated)

                if update_beta_beta1_xi_include_lambda2_closed_form:
                    log_lambda2 = update_lambda2(
                        pd,
                        current_theta,
                        theta,
                        t1_predictor=t1_predictor,
                    )
                    theta = theta._replace(log_lambda2=log_lambda2)
                return -q_for_estep(pd, current_theta, theta, t1_predictor)

            opt_result = jnr.minimize(coord_q,
                                      x_flat,
                                      atol=1e-3,
                                      rtol=m_step_tol,
                                      maxiter=m_step_maxiter,
                                      maxls=m_step_ls_maxiter)
            x_updated = x_unraveler(opt_result.guess)
            state = state._replace(m_step_status=opt_result.status)
            new_theta = _unpack_theta(update_params, fixed_masks,
                                      state.theta_current, x_updated)
        else:
            new_theta = state.theta_current

        # Update Lambda2
        new_log_lambda2 = update_lambda2(
            pd,
            state.theta_current,
            new_theta,
            t1_predictor=t1_predictor,
        )

        # jax.debug.breakpoint()

        state = state._replace(
            theta_current=new_theta._replace(log_lambda2=new_log_lambda2),
            step=state.step + 1,
        )

        # jax.debug.print(
        #     "P: {p}\n L2: {lambda2}\n gamma2={gamma2} beta1={beta1} opt_result={opt_result}===",
        #     p=state.theta_current.p(pd.lambda1_mask()),
        #     lambda2=jnp.exp(new_log_lambda2),
        #     gamma2=new_gamma2,
        #     beta1=new_beta1,
        #     opt_result=opt_result,
        # )

        return state

    state = jax.lax.while_loop(cond_fn, body_fn, init_state)

    e_step_status = jnp.where(state.step < e_step_maxiter, 0, 1)
    return EMResult(state.theta_current, state.m_step_status, e_step_status)


def obs_loglik_at_cluster(
    pd: PreprocessedData,
    theta: Parameters,
    cluster_id: int,
    t1_predictor: Callable[[chex.Array],
                           chex.Array] = time1_identity_predictor,
) -> float:
    """Observed likelihood of the data for a given cluster."""

    ut1_in_predictor = t1_predictor(pd.uts1)

    def get_indexer_for_cluster(cluster_index: ClusterIndex):
        start_idx = cluster_index.start_idx[cluster_id]
        end_idx = cluster_index.start_idx[cluster_id + 1]
        cluster_size = end_idx - start_idx
        cluster_mask = jnp.arange(
            cluster_index.max_cluster_length) < cluster_size
        cluster_idx = jax.lax.dynamic_slice_in_dim(
            cluster_index.cluster_idxs, start_idx,
            cluster_index.max_cluster_length)
        return jnp.where(cluster_mask, cluster_idx, -1)

    def compact_index(arr: chex.Array,
                      cluster_idx: chex.Array,
                      fill_value: chex.Scalar = -1,
                      aux_arr=None,
                      aux_cluster_idx=None):
        assert (aux_arr is None) == (aux_cluster_idx is None)
        arr_to_rank = arr[cluster_idx]
        mask = cluster_idx >= 0
        if aux_arr is not None:
            arr_to_rank = jnp.concatenate(
                [arr_to_rank, aux_arr[aux_cluster_idx]])
            mask = jnp.concatenate([mask, aux_cluster_idx >= 0])
        rank = jax.scipy.stats.rankdata(jnp.where(mask, arr_to_rank, jnp.inf),
                                        method="dense") - 1
        if aux_arr is not None:
            rank = rank[:cluster_idx.shape[0]]
        return jnp.where(cluster_idx >= 0, rank, fill_value)

    ut_index1_cluster_idx = get_indexer_for_cluster(pd.ut_index1_cluster_index)
    ut_index1_episode_bracket = pd.ut_index1.episode_bracket[
        ut_index1_cluster_idx]
    ut_index1_time = pd.ut_index1.time[ut_index1_cluster_idx]

    ut_index2_lt_lb3l_cluster_idx = get_indexer_for_cluster(
        pd.ut_index2_lt_lb3l_cluster_index)
    ut_index2_lt_rb3l_cluster_idx = get_indexer_for_cluster(
        pd.ut_index2_lt_rb3l_cluster_index)
    ut_index2_lt_lb3l_idx2l = compact_index(
        pd.ut_index2_lt_lb3l.idx2l,
        ut_index2_lt_lb3l_cluster_idx,
        aux_arr=pd.ut_index2_lt_rb3l.idx2l,
        aux_cluster_idx=ut_index2_lt_rb3l_cluster_idx)
    ut_index2_lt_rb3l_idx2l = compact_index(pd.ut_index2_lt_rb3l.idx2l,
                                            ut_index2_lt_rb3l_cluster_idx)
    ut_index2_lt_lb3l_time = pd.ut_index2_lt_lb3l.time[
        ut_index2_lt_lb3l_cluster_idx]
    ut_index2_lt_rb3l_time = pd.ut_index2_lt_rb3l.time[
        ut_index2_lt_rb3l_cluster_idx]

    part12_power1_ = -jnp.exp(
        pd.feature_t2[ut_index1_episode_bracket] @ theta.gamma2 +
        theta.beta1 * ut1_in_predictor[ut_index1_time])
    part1_power2_ = jax.ops.segment_sum(
        theta.lambda2[ut_index2_lt_lb3l_time],
        ut_index2_lt_lb3l_idx2l,
        num_segments=pd.ut_index1_cluster_index.max_cluster_length)

    part1 = jnp.exp(part12_power1_ * part1_power2_)

    part2_power2_ = jax.ops.segment_sum(
        theta.lambda2[ut_index2_lt_rb3l_time],
        ut_index2_lt_rb3l_idx2l,
        num_segments=pd.ut_index1_cluster_index.max_cluster_length)
    part2 = jnp.exp(part12_power1_ * part2_power2_)

    lambda1_cumsum = left_cumsum(theta.lambda1)
    expbz = jnp.exp(pd.feature_t1[ut_index1_episode_bracket] @ theta.gamma1)
    log_part3 = jnp.log(
        jnp.exp(-(lambda1_cumsum[:-1][ut_index1_time]) * expbz) -
        jnp.exp(-(lambda1_cumsum[1:][ut_index1_time]) * expbz))

    observed = pd.observed[ut_index1_episode_bracket]

    p12 = part1 - part2
    cond = observed & (ut_index1_cluster_idx >= 0) & (p12 > 0)
    loglik_ = (jnp.where(cond, jnp.log(jnp.where(cond, p12, 1)), 0) +
               log_part3)

    loglik = jnp.sum(jnp.where(
        (ut_index1_cluster_idx >= 0),
        loglik_,
        0,
    ))

    # jax.debug.print("{loglik} {part1} {part2} {log_part3} {observed} {p12}",
    #                 loglik=loglik,
    #                 part1=part1,
    #                 part2=part2,
    #                 log_part3=log_part3,
    #                 observed=observed,
    #                 p12=p12)
    return loglik


def obs_loglik(
    pd: PreprocessedData,
    theta: Parameters,
    t1_predictor: Callable[[chex.Array],
                           chex.Array] = time1_identity_predictor,
    minibatch_size: int = 100,
) -> float:
    """Observed likelihood of the data.

  Args:
    pd: Preprocessed data.
    theta: Parameters.
    t1_predictor: A function that takes a time and returns a predictor.
    minibatch_size: The size of the minibatch to use for the vmap inside a scan.
  """

    # ret = jax.vmap(functools.partial(obs_loglik_at_cluster,
    #                                  t1_predictor=t1_predictor),
    #                in_axes=(None, None, 0))(pd, theta,
    #                                         jnp.arange(pd.sample_size))
    # return jnp.sum(ret)

    def accumulate_loglik(loglik, cluster_i):
        cis = jnp.arange(minibatch_size) + cluster_i * minibatch_size
        loglik += jnp.sum(
            jnp.where(
                cis < pd.sample_size,
                jax.vmap(functools.partial(obs_loglik_at_cluster,
                                           t1_predictor=t1_predictor),
                         in_axes=(None, None, 0))(pd, theta, cis),
                0,
            ))
        return loglik, None

    loglik, _ = jax.lax.scan(
        accumulate_loglik, 0.,
        jnp.arange(0, int(math.ceil(pd.sample_size / minibatch_size))))

    return loglik


cov_sandwich_deriv_only_beta_beta1 = Parameters(
    gamma1=2,
    gamma2=2,
    beta1=2,
    log_lambda1=0,
    log_lambda2=0,
)
cov_sandwich_deriv_all = Parameters(
    gamma1=2,
    gamma2=2,
    beta1=2,
    log_lambda1=1,
    log_lambda2=1,
)


def pdify_padded_hessian(hess, pmask):
    """Make hessian diagonal corresponding to non-parameters the average of
  the rest of the diagonal elements."""
    diag_idxs = jnp.diag_indices_from(hess)
    hess_diag = hess[diag_idxs]
    avg_diag = jnp.sum(jnp.abs(hess_diag * pmask)) / jnp.sum(pmask)
    return hess.at[diag_idxs].add(avg_diag * (1 - pmask))


def hvp(f, primals, tangents):
    """Compute the hessian vector product."""
    return jax.jvp(jax.jacfwd(f), primals, tangents)[1]


def hessinvb(f, x, b, *args, hess_meth: Union[Callable, str] = jax.hessian):
    """Compute the inverse of the hessian of `f` times a vector.

  Calculates f''(x) .* b, where f''(x) is the hessian of f at x.
  """
    if hess_meth == "cg":

        def f_single(x):
            return f(x, *args)

        @functools.partial(jax.vmap, in_axes=1, out_axes=1)
        def matvec(b):
            return hvp(f_single, (x, ), (b, ))

        return jaxopt.linear_solve.solve_cg(matvec, b)
    else:
        hess = hess_meth(f)(x, *args)
        ret = jax.scipy.linalg.solve(hess, b, assume_a="sym", lower=True)
        # jax.debug.breakpoint()
        return ret


def compute_cov(
    static_sizes: StaticSizes,
    pd: PreprocessedData,
    theta_est: Parameters,
    return_sandwich: bool = True,
    param_mask: Parameters = cov_sandwich_deriv_only_beta_beta1,
    jacfn=jax.jacobian,
    hessfn=jax.hessian,
) -> chex.Array:
    """Compute the covraiance estimate of the parameters, using the sandwich.

  Args:
    pd: preprocessed data.
    theta_est: current estimate of the parameters.
    use_sandwich: whether to use the sandwich estimator.
    param_mask: an integer mask of the parameters to estimate. 0 means the
      parameter is a fixed constant, >= 1 means it is used to calculate the
      jacobian and hessian, >=2 means we are calculating variance for it.
  """
    pd = _fix_sizes(pd, static_sizes)

    theta_flat, theta_unraveler = jax.flatten_util.ravel_pytree(theta_est)
    # broadcast the mask to the same shape as theta
    param_mask = jax.tree_util.tree_map(
        lambda t, p: np.broadcast_to(p, t.shape), theta_est, param_mask)
    param_mask_flat, _ = jax.tree_util.tree_flatten(param_mask)
    param_mask_flat = np.concatenate(
        [np.atleast_1d(p) for p in param_mask_flat])
    theta_fixed_idxs, = np.where(param_mask_flat == 0)
    theta_deriv_idxs, = np.where(param_mask_flat >= 1)
    theta_cov_idxs, = np.where(param_mask_flat[theta_deriv_idxs] >= 2)

    def decompose_params(theta):
        theta_deriv = theta[theta_deriv_idxs]
        theta_fixed = theta[theta_fixed_idxs]
        return theta_deriv, theta_fixed

    def compose_params(theta_deriv, theta_fixed) -> Parameters:
        theta = jnp.zeros_like(theta_flat)
        theta = theta.at[theta_deriv_idxs].set(theta_deriv)
        theta = theta.at[theta_fixed_idxs].set(theta_fixed)
        return theta_unraveler(theta)

    def obs_loglik_flat_at_cluster(theta_deriv, theta_fixed, pd, cluster_id):
        params = compose_params(theta_deriv, theta_fixed)
        return obs_loglik_at_cluster(pd, params, cluster_id)

    def obs_loglik_flat(theta_deriv, theta_fixed, pd):
        params = compose_params(theta_deriv, theta_fixed)
        # old = obs_loglik_old(pd, params)
        ret = obs_loglik(pd, params)
        return ret

    theta_deriv, theta_fixed = decompose_params(theta_flat)

    x = jnp.zeros((theta_deriv.shape[0], theta_cov_idxs.shape[0]))
    x = x.at[theta_cov_idxs, :].set(jnp.eye(theta_cov_idxs.shape[0]))

    if hessfn == "cg":
        hessfn_padded = hessfn
    else:

        def hessfn_padded(*hess_args, **hess_kwargs):
            """Hessian function that handles padded parameters
                to make the hessian invertible."""

            def f(*args, **kwargs):
                hess = hessfn(*hess_args, **hess_kwargs)(*args, **kwargs)
                pmask = jnp.concatenate([
                    jnp.ones_like(theta_cov_idxs),
                    pd.lambda1_mask(),
                    pd.lambda2_mask()
                ])[theta_deriv_idxs]
                return pdify_padded_hessian(hess, pmask)

            return f

    with jax.named_scope("hessinvb"):
        bread = hessinvb(obs_loglik_flat,
                         theta_deriv,
                         x,
                         theta_fixed,
                         pd,
                         hess_meth=hessfn_padded)

    hinvcov = -bread[theta_cov_idxs]

    if not return_sandwich:
        return hinvcov
    else:

        def accumulate_meat(carry, cluster_id):
            jac = jacfn(obs_loglik_flat_at_cluster)(theta_deriv, theta_fixed,
                                                    pd, cluster_id)
            carry = carry + jnp.outer(jac, jac)
            return carry, None

        with jax.named_scope("sandwich_scan"):
            meat, _ = jax.lax.scan(
                accumulate_meat,
                jnp.zeros(
                    (theta_deriv_idxs.shape[0], theta_deriv_idxs.shape[0])),
                jnp.arange(pd.sample_size))

        with jax.named_scope("sandwich_hessinvb"):
            sandwich = hessinvb(obs_loglik_flat,
                                theta_deriv,
                                meat @ bread,
                                theta_fixed,
                                pd,
                                hess_meth=hessfn_padded)[theta_cov_idxs]

        # jax.debug.breakpoint()
        return (hinvcov, sandwich)


def jacfd(f, eps: float = 1e-6):
    """Compute the Jacobian of a function using finite differences."""

    def wrapped(x, *args, **kwargs):
        chex.assert_rank(x, 1)

        def jacf(vec):
            return (f(x + eps * vec, *args, **kwargs) -
                    f(x, *args, **kwargs)) / eps

        return jax.vmap(jacf, out_axes=-1)(jnp.eye(x.shape[0]))

    return wrapped


def hessfd(f, eps: float = 1e-6):
    """Compute the Hessian of a function using finite differences."""
    return jacfd(jacfd(f, eps), eps)


def make_fd_funs(static_sizes,
                 param_templ: Parameters,
                 eps: float = 1e-6,
                 **solver_kwargs):
    """Create the finite difference jacobian and hessian functions under profiled
  likelihood.

  Currently, this is a bit of a hack, in that it does not handle arbitrarily
  shaped parameters, and asssumes that the caller (`compute_cov`) is using the
  `cov_sandwich_deriv_only_beta_beta1` mode.

  Args:
    param_templ: A template of the parameters to use to determine the shape of
      the parameters.
    eps: The finite difference step size.
    **solver_kwargs: Keyword arguments to pass to the solver

  Returns:
    A tuple of the jacobian and hessian functions.
  """
    _, unraveler = jax.flatten_util.ravel_pytree(param_templ)

    def make_derivfn(deriv_hof: Union[jacfd, hessfd], f):
        """Create a derivative function that uses the profiled likelihood.

    Args:
      deriv_hof: The derivative function to use, either jacfd or hessfd.
      f: The function to differentiate, which is the true likelihood function.
    """

        def derivfun(x, *args, **kwargs):
            """The wrapped derivative function, has the same input signature as f,
      returns the corresponding derivative."""

            def newf(theta_deriv, theta_fixed, pd, *args):
                """An overloaded version of f. It is the same as f, except that it
        used the re-solved version of theta_fixed (lambda2) given
        current theta_deriv (gamma2, beta1 and xi) using the profiled likelihood."""

                if theta_deriv is x:
                    return f(theta_deriv, theta_fixed, pd, *args)

                theta_flat_ = jnp.concatenate([theta_deriv, theta_fixed])
                theta_ = unraveler(theta_flat_)
                result = solve_theta(
                    static_sizes,
                    pd,
                    theta_,
                    **solver_kwargs,
                    fixed_params=dict(gamma1=True,
                                      gamma2=True,
                                      beta1=True,
                                      log_lambda1=True),
                )
                return f(
                    theta_deriv,
                    jnp.concatenate(
                        [result.theta.log_lambda1, result.theta.log_lambda2]),
                    pd, *args)

            return deriv_hof(newf, eps=eps)(x, *args, **kwargs)

        return derivfun

    jac = functools.partial(make_derivfn, jacfd)
    hess = functools.partial(make_derivfn, hessfd)

    return jac, hess


def make_cf_funs(param_templ: Parameters, use_cg: bool = False):
    _, unraveler = jax.flatten_util.ravel_pytree(param_templ)

    def make_derivfn(deriv_hof: Union[jacfd, hessfd], f):
        """Create a derivative function that uses the profiled likelihood.

    Args:
      deriv_hof: The derivative function to use, either jacfd or hessfd.
      f: The function to differentiate, which is the true likelihood function.
    """

        def derivfun(theta_deriv, theta_fixed, pd: PreprocessedData):
            """The wrapped derivative function, has the same input signature as f,
      returns the corresponding derivative."""

            def newf(theta_deriv, theta_fixed, pd: PreprocessedData, *args):
                """An overloaded version of f. It is the same as f, except that it
        used the re-solved version of theta_fixed (lambda2) given
        current theta_deriv (gamma2, beta1 and xi) using the profiled likelihood."""

                def update_lambda_fun(theta_deriv, theta_fixed):
                    theta_flat_ = jnp.concatenate([theta_deriv, theta_fixed])
                    theta_ = unraveler(theta_flat_)
                    new_lambda2 = update_lambda2(
                        pd, theta_, theta_._replace(log_lambda2=None))
                    new_lambda2 = jnp.where(jnp.isneginf(new_lambda2), -40.,
                                            new_lambda2)
                    return new_lambda2

                @jax.custom_jvp
                def h_star(theta_deriv):
                    del theta_deriv
                    return theta_fixed

                def h_star_jvp(primals, tangents):
                    theta_deriv, = primals
                    t, = tangents

                    if use_cg:
                        h10t = jax.jvp(
                            lambda x: update_lambda_fun(x, theta_fixed),
                            (theta_deriv, ), (t, ))[1]

                        def eye_m_h01(b):
                            return b - jax.jvp(
                                lambda x: update_lambda_fun(theta_deriv, x),
                                (theta_fixed, ), (b, ))[1]

                        dh = jaxopt.linear_solve.solve_normal_cg(
                            eye_m_h01, h10t)

                    else:
                        h10 = jax.jacfwd(lambda x: update_lambda_fun(
                            x, theta_fixed))(theta_deriv)
                        h01 = jax.jacfwd(lambda x: update_lambda_fun(
                            theta_deriv, x))(theta_fixed)
                        dh = jax.scipy.linalg.solve(jnp.eye(h01.shape[0]) -
                                                    h01,
                                                    h10 @ t,
                                                    assume_a="gen")
                    return theta_fixed, dh

                h_star.defjvp(h_star_jvp)

                return f(theta_deriv, h_star(theta_deriv), pd, *args)

            return deriv_hof(newf)(theta_deriv, theta_fixed, pd)

        return derivfun

    jac = functools.partial(make_derivfn, jax.jacfwd)
    hess = functools.partial(make_derivfn, jax.hessian)

    return jac, hess


def build_covs_fns_helper(param_tmpl, sample_size, clustered, solver, covs):
    solver = dict(solver)

    def all_cov_fn(*args):
        static_sizes = args[0]
        cov_results = {}

        infered_sandwich = clustered

        def ad(use_cg=False, sandwich: Optional[bool] = None):
            sandwich = infered_sandwich if sandwich is None else sandwich
            covs = compute_cov(
                *args,
                param_mask=cov_sandwich_deriv_all,
                return_sandwich=sandwich,
                jacfn=jax.jacobian,
                hessfn="cg" if use_cg else jax.hessian,
            )
            if sandwich:
                cov_results["hessinv-all-ad"], cov_results[
                    "sandwich-ad"] = covs
            else:
                cov_results["hessinv-all-ad"] = covs

        def cf(use_cg: bool = False, sandwich: Optional[bool] = None):
            sandwich = infered_sandwich if sandwich is None else sandwich
            jacfn, hessfn = make_cf_funs(param_tmpl, use_cg=use_cg)
            covs = compute_cov(
                *args,
                param_mask=cov_sandwich_deriv_only_beta_beta1,
                return_sandwich=sandwich,
                jacfn=jacfn,
                hessfn=hessfn,
            )
            if sandwich:
                cov_results["hessinv-bg-cf"], cov_results[
                    "sandwich-bg-cf"] = covs
            else:
                cov_results["hessinv-bg-cf"] = covs

        def fd(eps: bool, sandwich: Optional[bool] = None):
            sandwich = infered_sandwich if sandwich is None else sandwich
            jacfn, hessfn = make_fd_funs(static_sizes, param_tmpl,
                                         eps / np.sqrt(sample_size), **solver)
            fd_name = f"bg-fd-{eps}"
            covs = compute_cov(
                *args,
                param_mask=cov_sandwich_deriv_only_beta_beta1,
                return_sandwich=sandwich,
                jacfn=jacfn,
                hessfn=hessfn,
            )
            if sandwich:
                cov_results[f"hessinv-{fd_name}"], cov_results[
                    f"sandwich-{fd_name}"] = covs
            else:
                cov_results[f"hessinv-{fd_name}"] = covs

        parser = simpleeval.EvalWithCompoundTypes(functions={
            "ad": ad,
            "cf": cf,
            "fd": fd
        })
        if covs:
            parser.eval(covs)
        return cov_results

    return all_cov_fn
