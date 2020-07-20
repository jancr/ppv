# core imports
import pickle

# 3rd party imports
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pymc3 as pm


# lodal imports
import ppv  # noqa
from ppv.model import PPVModel
#  from scripts.run import create_features


# setup
mpl.use('agg')
plt.style.use('seaborn-white')
color = '#87ceeb'
f_dict = {'size': 16}


#  def qr_data(data):
#      # center data
#      X = data.ppv._transform().ppv.predictors
#      meanx = X.mean().values
#      scalex = X.std().values
#      zX = ((X - meanx) / scalex).values


#  def create_model_qr(zX, y, graph=False):
#
#      with pm.Model() as model:
#          # qr decompose predictors and normalize
#          Q, R = sp.linalg.qr(zX, mode='economic')
#          Q *= zX.shape[0]
#          R /= zX.shape[0]
#
#          zbeta0 = pm.Normal('zbeta0', mu=0, sd=2)
#          zbetaj = pm.Normal('zbetaj', mu=0, sd=2, shape=(zX.shape[1]))  # noqa
#
#          betaj_tilde = pm.Deterministic('betaj_tilde', pm.math.dot(R, zbetaj))
#          p = pm.invlogit(zbeta0 + pm.math.dot(Q, betaj_tilde))
#
#          likelihood = pm.Bernoulli('likelihood', p, observed=y.values)  # noqa
#
#          pm.model_to_graphviz(model)
#      return model


#  def sample(model, n=3000):
#      with model:
#          trace = pm.sample(n, chains=8, cores=1)
#      return trace


#  def sample_model(zX, y, down_sample=None, model_type='default'):
#      #  positives = df[df["Annotations", "Known"]]
#      #  negatives = df[~df["Annotations", "Known"].astype(bool)]
#      #  data = pd.concat((positives, negatives))
#
#      if model_type == 'default':
#          model = create_model(zX, y)
#      else:
#          model = create_model_qr(zX, y)
#      with model:
#          #  trace = pm.sample(1000, cores=8)
#          trace = pm.sample(3000, cores=4, chains=8)
#      if down_sample is None:
#          pickle.dump(trace, open("pickle/model_trace_obs_data.pickle", "wb"))
#      else:
#          _f = ("pickle/model_trace_obs_data_{}.pickle".format(down_sample)
#          pickle.dump(trace, open(_f, "wb"))
#          pm.traceplot(trace)
#      pm.traceplot(trace)
#      fig = plt.gcf()
#      fig.savefig("plots/trace_{}_{}.pdf".format(down_sample, model_type))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="number of processes", default=16, type=int)
    parser.add_argument("-s", help="class balance after down-sampeling", default=None, type=int)
    parser.add_argument("--draw", help="how manny draws to make", type=int, default=2000)
    parser.add_argument("--model", help="model type", default='default')
    parser.add_argument("--drop-weak-features", action='store_true', default=False)
    parser.add_argument("--build", action='store_true', default=False)
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--devel", action="store_true", default=False)
    parser.add_argument("--paper", help="Exclude Big Brain", action="store_true", default=False)
    parser.add_argument("--null", help="Predict based on intensity only", action="store_true",
                        default=False)
    parser.add_argument("--grey-list", help="Remove negatives with KR during training",
                        action="store_true", default=False)
    parser.add_argument("--ms", help="Only keep MS features", action="store_true", default=False)
    parser.add_argument("--chemical", help="Only keep Chemical features", action="store_true",
                        default=False)
    parser.add_argument("--no-transform", help="Do not transform chemical featurs",
                        action="store_true", default=False)
    parser.add_argument("--core", help="Only keep Intensity/Bool start/stop", action="store_true",
                        default=False)
    #  parser.add_argument("--metropolis", default=False, action='store_true')
    parser.add_argument("--mini-batch", help="Use  batches to speed up", type=int,
                        default=0)
    parser.add_argument('--traceplot', default=False, action='store_true')
    # TODO: make plots with and without observed feature!!
    return parser.parse_args()


if __name__ == '__main__':  # noqa
    args = parse_args()
    base_name = ""
    if args.paper:
        df = pickle.load(open("pickle/mouse_features_paper.pickle", 'rb'))
        base_name += "paper_"
    else:
        df = pickle.load(open("pickle/mouse_features.pickle", 'rb'))

    # data upsampling/down
    if args.build:
        data = df
        base_name += "ppv_data"
    else:
        data = df.ppv.observed
        #  data = df[df["MS Bool", "observed"]]
        #  del data["MS Bool", "observed"]
        base_name += "obs_data"

    # feature engenering
    if args.no_transform:
        base_name += '_no_transform'
    else:
        data = data.ppv._transform()

    # (sub)feature selection
    if args.null:
        data = data.ppv.transform_to_null_features()
        base_name += "_null_model"
    elif args.core:
        data = data.ppv.transform_to_core_features()
        base_name += "_core_model"
    elif args.ms:
        data = data.ppv.drop("Chemical")
        base_name += "_ms_model"
    elif args.chemical:
        data = data.ppv.subset({'Chemical'})
        base_name += "_chem_model"
    else:
        base_name += "_full_model"

    # feature selection
    if args.drop_weak_features:
        base_name += '_strong'
        data = data.ppv._drop_weak_features()

    # reduce stuff
    down_sample = args.s
    true_prior = data.ppv.get_prior()  # calcuate true prior before down sampling!
    if args.s is not None:
        base_name += '_{}'.format(args.s)
        positives = data.ppv.positives
        negatives = data.ppv.negatives
        negative_samples = down_sample * positives.shape[0]
        data = pd.concat((positives, negatives.sample(negative_samples)))

    if args.load:
        ppv_model = PPVModel.load("pickle/model_{}.ppvmodel".format(base_name))
    else:
        #  if args.mini_batch > 0:
        #      base_name += "mini_batch_{}".format(args.mini_batch)
        ppv_model = PPVModel(data, mini_batch=args.mini_batch)
        with ppv_model.model:
            #  if args.metropolis:
            #      base_name += "_metropolis"
            #      trace = pm.sample(args.draw, step=pm.HamiltonianMC(), cores=8, chains=8)
            #  else:
            trace = pm.sample(args.draw, cores=8, chains=8, target_accept=0.9)
            #  trace = pm.sample(args.draw, cores=8, chains=8,
            #                    max_treedepth=15, target_accept=0.90
            #                    step=[ppv_model.nuts, ppv_model.metropolis])
        ppv_model.add_trace(trace, true_prior=true_prior)
        ppv_model.save("pickle/model_{}.ppvmodel".format(base_name))

    if args.traceplot:
        ax = pm.plots.traceplot(ppv_model.trace)
        fig = plt.gcf()
        fig.savefig('figures/traceplots/{}.pdf'.format(base_name))

    # posteori
    ppv_model.plot_posterior("figures/posterior_{}.pdf".format(base_name))
    ppv_model.plot_zposterior("figures/posterior_z_{}.pdf".format(base_name))
