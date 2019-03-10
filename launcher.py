import os
import sys
import torch
import logging
from data_loader import prepare_data
import numpy as np
import datetime
from argparse import ArgumentParser
#add YW
import pandas as pd
from accuracy_measure import *
import scipy.sparse as sp

# Add ProteinFunctionPrediction to sys.path for importing modules
sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))

# Parse arguments
parser = ArgumentParser()
parser.add_argument('-n', '--num_proteins', default=1000, dest="num_proteins", type=int)

args = parser.parse_args()

num_loaded_proteins = args.num_proteins   # Number of proteins to load

if os.path.isdir('/usr/data/cvpr_shared/biology/function'):
    data_dir = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
               'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/'
    memmap_dir = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                 'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/'
    protein_data_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                            'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/BLAST_KNN/data/proteins.csv'
    gt_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                 'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/test/gt_test_holdout_stage2.npy'
    go_term_lookup_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                 'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/GOtermsLookup_file.csv'
    BLAST_KNN_result_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                            'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/BLAST_KNN/data/Blast_knn_result_parents.npy'
else:
    sys.exit('Data directories not found.')

torch.multiprocessing.set_sharing_strategy('file_system')


proteins = pd.read_csv(protein_data_path)
# num_proteins
protein_test = proteins['name'][0:num_loaded_proteins]
x,y,unique_go_values,idxMF, idxBP, idxCC = prepare_data(memmap_dir,protein_test)

print("Functions in MF: {}".format(len(idxMF)))
print("Functions in BP: {}".format(len(idxBP)))
print("Functions in CC: {}".format(len(idxCC)))
#
from blast_knn_model import BlastKnnModel
print("BLAST KNN MODEL")
# gotermlookup = pd.read_csv(go_term_lookup_path)
# protein = pd.read_csv(protein_data_path)
# gt = sp.load_npz(go_term_protein)
# #protein name for training
# x = protein[protein["Unnamed: 0"].isin(selected_rows)]["name"].tolist()
# all go term labels
# unique_go_values = gotermlookup.iloc[selected_columns]["termID"].tolist()
# #groundtruth matrix
# y = gt[selected_rows].todense()
# y = y[:,selected_columns]
blastKnnModel = BlastKnnModel(idxMF, idxBP, idxCC,x,unique_go_values,y)
BlastKnnModel.prepare_blast_result(blastKnnModel,BLAST_KNN_result_path)
test_prob,gt = blastKnnModel.blast()



