import matplotlib
matplotlib.use('pdf')  # noqa

# core imports
import pickle
import argparse
import tqdm
from os.path import join as pjoin

# 3rd party imports
import pandas as pd

# jcr modules
import peputils  # noqa
from peputils import fasta_to_protein_hash

# local imports
#  from ppv.protein import ProteinFeatureExtractor
import ppv  # noqa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="number of processes", default=16, type=int)
    parser.add_argument("--load", help="load pickle module", action="store_true", default=False)
    return parser.parse_args()


def create_features(load, processes):
    base_folder = "/opt/projects/4478_NN/00009_NHP_Peptidomics/jcr/peptidomics/results/campaign"
    base_folder = "results/campaign"
    base_files = (
        "mouse_brain_combined{ext}", "mouse_brain_large_mq{ext}", "mouse_epifat_combined{ext}",
        "mouse_ileum_mascot{ext}", "mouse_liver_combined{ext}", "mouse_pancreas_combined{ext}",
        "mouse_qmuscle_combined{ext}", "mouse_scfat_combined{ext}")
    campaign_names = ("Mouse Brain", "Mouse Large Brain", "Mouse Epifat", "Mouse Ileum",
                      "Mouse Liver", "Mouse pancreas", "Mouse Qmuscle", "Mouse Scfat")
    base_files = [pjoin(base_folder, f) for f in base_files]

    peptidomics_folder = "/opt/projects/4478_NN/00009_NHP_Peptidomics/jcr/peptidomics-skk/{}"
    mouse_fasta = peptidomics_folder.format("results/proteomes/10090_uniprot.fasta")
    mouse_proteins = fasta_to_protein_hash(mouse_fasta)
    known_file = "tests/test_data/10090_known.tsv"

    base_pickle_file = 'pickle/mouse_peptide_features_{}.pickle'
    if load:
        features = [pickle.load(open(base_pickle_file.format(c), 'rb')) for c in campaign_names]
    else:
        _iter = zip(base_files, campaign_names)
        _iter = tqdm.tqdm(_iter, "Creating Features", total=len(campaign_names))
        features = []
        for base_file, campaign_name in _iter:
            _iter.set_description("Creating Features({}".format(campaign_name), False)

            # upf
            upf_file = base_file.format(ext='.upf')
            sample_meta_file = base_file.format(ext='.sample.meta')
            df_raw = pd.DataFrame.peptidomics.load_upf_meta(upf_file, sample_meta_file,
                                                            campaign_name)
            df = df_raw.peptidomics.normalize()

            # features
            dataset_features = df.ppv.create_feature_df(mouse_proteins, n_cpus=processes,
                                                        known=known_file, peptides='fair')
            features.append(dataset_features)

            # pickle
            pickle_file = 'pickle/mouse_peptide_features_{}.pickle'.format(campaign_name)
            pickle.dump(dataset_features, open(pickle_file.format(campaign_name), 'wb'))

    return pd.concat(features)


#  def plot(df):
#      df.ppv_features.plot("ppv_features_joinplot.pdf")


if __name__ == '__main__':
    args = parse_args()
    create_features(args.load, args.p)
