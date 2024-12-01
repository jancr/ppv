'''
Helper script to run PeptideRanker on our full dataframe.
Cannot use multithreading in notebook, need 64core node.
'''

import tempfile
import subprocess
import os
from pandarallel import pandarallel
import pandas as pd

# PeptideRanker
BIN_DIR = '/novo/users/jref/share/fegt/bin_with_peptideranker'
PEPTIDE_RANKER_CMD = os.path.join(BIN_DIR, 'PeptideRanker')
PEPTIDE_RANKER_MODELS_DIR = os.path.join(BIN_DIR, 'PeptideRankerModels')
PEPTIDE_RANKER_MODELS_SHORT = os.path.join(PEPTIDE_RANKER_MODELS_DIR, 'Models_Short.txt')
PEPTIDE_RANKER_MODELS_LONG = os.path.join(PEPTIDE_RANKER_MODELS_DIR, 'Models_Long.txt')
################################################################################
# PeptideRanker
################################################################################
def get_peptide_rank(seq: str) -> float:
    """returns the peptide rank for a sequence
    Note: this method is thread safe, if we need it speed up"""

    # peptide ranker is very quirly:
    # it always writes output = "{input_file_name}.predictions"
    # - to the same folder
    # it only looks for Models_Long and Models Short in the current folder
    # - so we have to change dir for the process running the NN

    ranker_string = "1\npeptide\n{length}\n{seq}\n1\n"
    with tempfile.NamedTemporaryFile(prefix='PeptideRanker_') as tmp_file:
        input_file = tmp_file.name
        '{}.predictions'.format(input_file)
        length = len(seq)
        file_content = ranker_string.format(seq=seq, length=length)

        # make sure it is closed (and thus flushed)
        with open(input_file, 'w') as tf:
            tf.write(file_content)

        if length <= 20:
            models = 'Models_Short.txt'
        else:
            models = 'Models_Long.txt'
        command_args = ['../PeptideRanker', models, input_file, './']
        subprocess.call(command_args, cwd=PEPTIDE_RANKER_MODELS_DIR)

        prediction_file = '{}.predictions'.format(input_file)

        #  import colored_traceback.auto; import ipdb; ipdb.set_trace()
        score = open(prediction_file, 'r').readlines()[-1].rstrip().split(' ')[-1]
        os.unlink(prediction_file)
        return float(score)

if __name__ == '__main__':
    pandarallel.initialize(nb_workers=64, progress_bar=False)
    df = pd.read_pickle('mouse_features_paper_assembly.pickle')
    df[('Predictions', 'PeptideRanker')] = df['Annotations']['Sequence'].parallel_apply(get_peptide_rank)


    df.to_pickle('mouse_features_paper_assembly_with_peptideranker.pickle')