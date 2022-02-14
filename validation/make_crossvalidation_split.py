'''
Splits the data into five folds. Only run this once and keep assignments fixed for all experiments.
'''
import argparse
import ppv
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('file_in', type=str)
    parser.add_argument('file_out', type=str)
    args = parser.parse_args()

    df = pd.read_pickle(args.file_in)

    #df = df.sort_values(('Annotations', 'Known'), ascending=False) # bioactives are processed first.
    #proteins = df.index.get_level_values(1).astype('category').codes
    #tissues = df.index.get_level_values(0).astype('category').codes
    #labels = df['Annotations']['Known'].cat.codes.values # Index and Series objects work differently.
    #partition_assignments = partition_assignment(proteins, tissues, labels, n_partitions=5, n_class=2, n_kingdoms=7)
    #df[('Annotations', 'Partition')] = partition_assignments.astype(int)
    df.ppv.generate_xfolds()

    print('Label balance')
    print(df['Annotations'].groupby('Fold')['Known'].value_counts().unstack())

    print('Tissue balance')
    print(df['Annotations'].groupby(level=0)['Fold'].value_counts().unstack())

    dirname = os.path.dirname(args.file_out)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    df.to_pickle(args.file_out)


if __name__ == '__main__':
    main()
