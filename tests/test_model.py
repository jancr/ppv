
# core imports
import pathlib
import tempfile

# 3rd party imports
import pytest

# local imports
from ppv.model import PPVModel



class TestModelIO:
    def equal(self, model1, model2, only_samples=False):
        assert all(model1.df == model2.df)
        assert model1.true_prior == model2.true_prior
        if only_samples:
            assert all(model1.beta0 == model2.beta0)
            assert all(model1.betaj == model2.betaj)
        else:
            assert model1.trace == model2.trace

    def test_io(self, ppv_model):
        with tempfile.NamedTemporaryFile() as ppv_file:
            for fmt in ('large', 'pickle', 'netcdf'):
                ppv_model.save(ppv_file.name, fmt)
                ppv_model_load = PPVModel.load(ppv_file.name)
                self.equal(ppv_model, ppv_model_load)
            for fmt in ('small', 'samples'):
                ppv_model.save(ppv_file.name, fmt)
                ppv_model_load = PPVModel.load(ppv_file.name)
                self.equal(ppv_model, ppv_model_load, True)


