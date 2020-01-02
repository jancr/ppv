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
from peputils.proteome import fasta_to_protein_hash

# local imports
#  from ppv.protein import ProteinFeatureExtractor
import ppv  # noqa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="number of processes", default=16, type=int)
    parser.add_argument("--load", help="load pickle module", action="store_true", default=False)
    parser.add_argument("--train", help="train model", action="store_true", default=False)
    parser.add_argument("--paper", help="Exclude Big Brain", action="store_true", default=False)
    return parser.parse_args()


def create_features(load, processes, paper=False):
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

    #  base_pickle_file = 'pickle/mouse_peptide_features_{}.pickle'

    if paper:
        mouse_features_file = "pickle/mouse_features_paper.pickle"
    else:
        mouse_features_file = "pickle/mouse_features.pickle"
    if load:
        #  features = [pickle.load(open(base_pickle_file.format(c), 'rb')) for c in campaign_names]
        #  features = [features[0]]  # TODO delete me
        return pickle.load(open(mouse_features_file, 'rb'))
    else:
        _iter = list(zip(base_files, campaign_names))
        if paper:
            _iter = [(_f, _c) for (_f, _c) in _iter if _c != "Mouse Large Brain"]
        _iter = tqdm.tqdm(_iter, "Creating Features")
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
            dataset_features = df.ppv_feature_extractor.create_feature_df(
                mouse_proteins, n_cpus=processes, known=known_file, peptides='valid')
            features.append(dataset_features)

            # pickle
            pickle_file = 'pickle/mouse_peptide_features_{}.pickle'.format(campaign_name)
            pickle.dump(dataset_features, open(pickle_file.format(campaign_name), 'wb'))

    features = pd.concat(features)
    target = features["Annotations", "Known"].astype(bool).astype('category')
    features["Annotations", "Known"] = target
    pickle.dump(features, open(mouse_features_file, 'wb'))
    return features


#  def train_model(features):
#      if isinstance(features, str):
#          features = pickle.loads(open(features, 'rb'))
#
#      model = features.ppv_features.create_model("pickle/ppv.model")
#      print(features.ppv_features.predict(model))
#      #  print(features.ppv_features.predict("pickle/ppv.model"))

    #  # fix features
    #  features_transformed = features.ppv_features.transform()
    #  scaler = features_transformed.ppv_features.get_scaler()
    #  features_scaled = features_transformed.ppv_features.scale_predictors(scaler)
    #
    #  # train
    #  clf = features_scaled.ppv_features.train()
    #  pd.DataFrame.ppv_features.save_model(clf, scaler, "pickle/model.pickle",)
    #  #  pickle.dump((clf, scaler), open("pickle/model.pickle", "wb"))
    #  return clf, scaler


if __name__ == '__main__':
    args = parse_args()
    features = create_features(args.load, args.p, args.paper)
    #  if args.train:
    #      train_model(features)
