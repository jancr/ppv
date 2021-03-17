
# core imports
import pathlib
import pickle

# 3rd party imports
import pytest
import pandas as pd

# local imports
import ppv

TEST_FOLDER = pathlib.Path(__file__).parent
TEST_DATA = TEST_FOLDER / 'test_data'


@pytest.fixture()
def df_features():
    return pd.read_pickle(str(TEST_DATA / 'df_features.pickle'))


@pytest.fixture()
def inference_data():
    return pickle.load((TEST_DATA / 'inference_data.pickle').open('rb'))


@pytest.fixture
def true_prior():
    return 207 / (217902 - 207)


@pytest.fixture()
def ppv_model(df_features, inference_data, true_prior):
    df_strong_features = df_features.ppv._drop_weak_features()
    model = ppv.model.PPVModel(df_strong_features, true_prior=true_prior)
    model.add_trace(inference_data)
    return model

    

