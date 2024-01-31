"""
pssm = amino_acid_residues
sol_acc=2
nc_term =2
feats = pssm+sol_acc+nc_term
"""

import pandas as pd
import numpy as np

sequence_len = 500
total_features = 57
amino_acid_residues = 21
num_classes = 8
feats = amino_acid_residues +4

 


def load_dateset(data_file_path):
    data = pd.read_csv(data_file_path)
    return np.reshape(data.values, (data.shape[0],sequence_len,feats))

def get_data_labels(d, only_pssm=True):
    if only_pssm: start_feat = 4
    else: start_feat =0
    
    X = d[:,:,start_feat:(start_feat+amino_acid_residues)]
    Y = D[:, :, (start_feat+amino_acid_residues):(start_feat+amino_acid_residues + num_classes)]
    
    return X,Y
        
                      
                      
    