#!/usr/bin/env python

# core imports
import sys
import os
import pathlib

# 3rd party imports
import pandas as pd
import numpy as np

import peputils
# from peputils.proteome import fasta_to_protein_hash
# import ppv

def load_refrence_upf(ref_dir):
    df_mouse = {}
    metas = sorted([f for f in pathlib.Path(ref_dir).iterdir() if f.name.endswith('meta')])
    upfs = sorted([f for f in pathlib.Path(ref_dir).iterdir() if f.name.endswith('upf')])
    for meta_file, upf_file in zip(metas, upfs):
        campaign_name = f"Mouse {upf_file.name.split('_')[1].capitalize()}"
        df_raw = pd.DataFrame.peptidomics.load_upf_meta(upf_file, meta_file, campaign_name)
        
        df_mouse[campaign_name] = df_raw.peptidomics.normalize()
    return df_mouse

def calc_refrence_ms_distribution(df_ref):
    means = []
    variances = [] 
    for name, df in df_ref.items():
        df_log = np.log10(df)
        means.append(df_log.mean())
        variances.append(df_log.var(ddof=1))
    m = pd.concat(means).mean()
    sd = (pd.concat(variances) ** 0.5).mean()
    return (m, sd)


if __name__ == '__main__':
    df_ref = load_refrence_upf(sys.argv[1])
    ref_mean, ref_sd = calc_refrence_ms_distribution(df_ref)
    print(f'Refrence Mean is {ref_mean}')
    print(f'Refrence sd   is {ref_sd}')


