"""Test for experiments script"""

import argparse
import collections
import copy
import os
import shutil
from typing import Any, List

import hypothesis as hp
import hypothesis.strategies as st

import vaes_ptorch.experiments as exp
import vaes_ptorch.utils as ut

TEST_EXP_PATH = os.path.join(exp.DATA_PATH, "temp_experiments")


def test_init_args_basic():
    none_args = {
        "info_vae": [
            0,
        ],
        "div_scales": None,
        "div_scale": 1,
        "latent_dims": None,
        "latent_dim": 2,
        "lrs": [3.0],
        "lr": 5,
    }
    in_args = argparse.Namespace(**none_args)
    out_args, num_experiments = exp.init_args(in_args)
    assert out_args.info_vae == [0]
    assert out_args.div_scales == [1]
    assert out_args.latent_dims == [2]
    assert out_args.lrs == [3.0]
    assert num_experiments == 1


small_set_strategy = st.sets(st.integers(), min_size=1, max_size=100)


def counter_check(input_list: List[Any], repeated_list: List[Any], num_repeats: int):
    counter = collections.Counter(repeated_list)
    assert len(counter) == len(input_list)
    for count in counter.values():
        assert count == num_repeats


@hp.given(small_set_strategy, small_set_strategy, small_set_strategy)
def test_init_args(div_scales, latent_dims, lrs):
    in_args = argparse.Namespace(
        **{
            "info_vae": [0, 1],
            "div_scales": list(div_scales),
            "div_scale": "a",
            "latent_dims": list(latent_dims),
            "latent_dim": "b",
            "lrs": list(lrs),
            "lr": "c",
        }
    )
    out_args, num_experiments = exp.init_args(copy.deepcopy(in_args))
    assert num_experiments == len(div_scales) * len(latent_dims) * len(lrs) * len(
        in_args.info_vae
    )
    counter_check(
        input_list=div_scales,
        repeated_list=out_args.div_scales,
        num_repeats=num_experiments // len(div_scales),
    )
    counter_check(
        input_list=latent_dims,
        repeated_list=out_args.latent_dims,
        num_repeats=num_experiments // len(latent_dims),
    )
    counter_check(
        input_list=lrs,
        repeated_list=out_args.lrs,
        num_repeats=num_experiments // len(lrs),
    )
    counter_check(
        input_list=in_args.info_vae,
        repeated_list=out_args.info_vae,
        num_repeats=num_experiments // len(in_args.info_vae),
    )


def test_end_to_end():
    # setup temp directory if necessary
    ut.check_exp_dir(TEST_EXP_PATH)
    try:
        args = argparse.Namespace(
            **{
                "info_vae": [0],
                "div_scales": None,
                "div_scale": 1.0,
                "latent_dims": [20, 100],
                "latent_dim": 10,
                "lrs": [1e-5, 1e-3, 1e-1],
                "lr": 1e-3,
                "num_repeats": 2,
                "exp_path": TEST_EXP_PATH,
                "trunc_share": 0.99,
                "batch_size": 128,
                "num_epochs": 2,
            }
        )
        num_expected_experiments = (
            len(args.latent_dims) * len(args.lrs) * args.num_repeats
        )
        exp.mnist_main(args)
        full_data = ut.load_experiments_data(TEST_EXP_PATH)
        assert len(full_data) == 9  # 9 keys to save data for, for each experiment
        assert all(
            len(saved_items) == num_expected_experiments
            for saved_items in full_data.values()
        ), (
            [len(vals) for vals in full_data.values()],
            num_expected_experiments,
        )
    finally:
        # tear down temp directory
        shutil.rmtree(TEST_EXP_PATH)
