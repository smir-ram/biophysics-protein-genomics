"""
pssm = amino_acid_residues
sol_acc=2
nc_term =2
feats = pssm+sol_acc+nc_term
"""

import pandas as pd
import numpy as np

sequence_len = 500

nc_sasa = 4
amino_acid_residues = 21
num_classes = 8
# total_features = nc_sasa+amino_acid_residues+num_classes

feats = nc_sasa+ amino_acid_residues

X_base_coln = set(['Nt','Ct','SASA-Abs','SASA-Rel']+list('ACDEFGHIKLMNPQRSTVWXY'))
Y_base_coln= set(['ssL', 'ssB', 'ssE', 'ssG', 'ssI', 'ssH', 'ssS', 'ssT'])

def load_dataset(data_file_path, use_labels=[]):
    num_Y = len(use_labels)
    if  num_Y == 0:
        total_features = nc_sasa+amino_acid_residues+num_classes # all labels
        num_Y = num_classes
    else:
        total_features = nc_sasa+amino_acid_residues+num_Y
    # Load Data CSV
    data = pd.read_csv(data_file_path)

    # filter labels 
    remove_coln = Y_base_coln - set(use_labels)
    re = "|".join([x for x in remove_coln])
    columns_to_drop = data.columns[data.columns.str.contains(re)]
    data.drop(columns=columns_to_drop, inplace=True)
    print (f" removed labels {re}" )
    return np.reshape(data.values, (data.shape[0],sequence_len,total_features)), num_Y

def get_data_labels(d, num_Y,only_pssm=True):
    if only_pssm: start_feat = 4
    else: start_feat =0
    
    X = d[:,:,start_feat:(start_feat+amino_acid_residues)]
    Y = d[:, :, (start_feat+amino_acid_residues):(start_feat+amino_acid_residues + num_Y)]
    
    return X,Y
        
                      
                      
    