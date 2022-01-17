import os
import shutil

import hypothesis as hp
import hypothesis.strategies as st

import vaes_ptorch.experiments as exp
import vaes_ptorch.utils as ut

TEST_EXP_PATH = os.path.join(exp.DATA_PATH, "temp_experiments")


@hp.given(
    st.lists(st.integers(), min_size=0, max_size=100),
    st.integers(min_value=1, max_value=100),
)
def test_repeat_list(input_list, num_repeats):
    out_list = ut.repeat_list(input_list, num_repeats)
    n = len(input_list)
    for ix in range(n * num_repeats):
        circled_ix = ix % n
        assert out_list[ix] == input_list[circled_ix]


data_point_st = st.tuples(
    st.floats(allow_nan=True, allow_infinity=True),
    st.floats(allow_nan=True, allow_infinity=True),
)
experiment_data_st = st.lists(data_point_st, min_size=1, max_size=1000)
experiments_data_st = st.lists(experiment_data_st, min_size=1, max_size=100)


@hp.given(experiments_data_st)
def test_save_load_experiments(exp_datas):
    # setup temp directory if necessary
    ut.check_exp_dir(TEST_EXP_PATH)
    try:
        num_to_save = 0
        for exp_data in exp_datas:
            for (val_err, test_err) in exp_data:
                ut.save_experiment(
                    {"val_err": val_err, "test_err": test_err}, TEST_EXP_PATH
                )
                num_to_save += 1
        full_data = ut.load_experiments_data(TEST_EXP_PATH)
        assert len(full_data) == 2, len(full_data)
        assert len(full_data["val_err"]) == num_to_save, (
            len(full_data["val_err"]),
            num_to_save,
        )
        assert len(full_data["test_err"]) == num_to_save, (
            len(full_data["test_err"]),
            num_to_save,
        )
    finally:
        # tear down temp directory
        shutil.rmtree(TEST_EXP_PATH)
