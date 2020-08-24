# core imports
import os
import os.path
from os.path import join as pjoin

# 3rd party import
import pytest
import pandas as pd
import numpy as np

# local imports
from sequtils import SequenceRange
import peputils  # noqa

# module being tested
from ppv.protein import ProteinFeatureExtractor

TEST_DATA = pjoin(os.path.dirname(__file__), "test_data")


# fixtures
@pytest.fixture(scope='module')
def glucagon_sequence():
    with open(pjoin(TEST_DATA, "glucagon/glucagon.fasta")) as f:
        return ''.join((line.rstrip() for line in f.readlines()[1:]))


@pytest.fixture(scope='module')
def glucagon_known_peptides():
    with open(pjoin(TEST_DATA, "glucagon/mouse_glucagon.known")) as f:
        peptides = set()
        f.readline()  # skip header
        for line in f.readlines():
            protein_id, start, stop, sequence, *_ = line.rstrip().split('\t')
            peptides.add(SequenceRange(start, stop, seq=sequence))
    return peptides


@pytest.fixture(scope='function')
def df_glucagon():
    base_file = pjoin(TEST_DATA, "glucagon/mouse_brain_glucagon{ext}")
    upf_file = base_file.format(ext='.upf')
    sample_meta_file = base_file.format(ext='.sample.meta')
    df_raw = pd.DataFrame.peptidomics.load_upf_meta(upf_file, sample_meta_file, "mouse brain")
    return df_raw  # test dataset is to "snall" for normalization to work
    #  return df_raw.peptidomics.normalize()


@pytest.fixture(scope='function')
def pfe_glucagon(df_glucagon, glucagon_sequence, glucagon_known_peptides):
    #  median = np.nanmedian(df_glucagon.values.flatten())
    return ProteinFeatureExtractor(df_glucagon, glucagon_sequence, glucagon_known_peptides)


class TestProteinFeatureExtractor:
    #  glucagon_file = "tests/test_data/mouse_brain_glucagon.{}"

    def test_init(self, pfe_glucagon, glucagon_sequence, df_glucagon,
                  glucagon_known_peptides):
        assert pfe_glucagon.campaign_id == "mouse brain"
        assert pfe_glucagon.protein_id == "P55095"
        assert pfe_glucagon.n_samples == df_glucagon.shape[1]

    def test_create_feature_df(self):
        pass  # TODO!!!
