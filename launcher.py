import os
import sys
import torch
import logging
from data_loader import prepare_data,prepare_test_data
import numpy as np
import datetime
from argparse import ArgumentParser
#add YW
import pandas as pd
from accuracy_measure import *

# Add ProteinFunctionPrediction to sys.path for importing modules
sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))

# Parse arguments
parser = ArgumentParser()
parser.add_argument('-n', '--num_proteins', default=1000, dest="num_proteins", type=int)

args = parser.parse_args()

num_loaded_proteins = args.num_proteins   # Number of proteins to load

if os.path.isdir('/usr/data/cvpr_shared/biology/function'):
    test_data_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                 'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/test/proteins_holdout_stage2.csv'
    gt_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                 'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/test/gt_test_holdout_stage2.npy'
    go_term_lookup_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                 'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/GOtermsLookup_file.csv'
    BLAST_KNN_result_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                            'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/BLAST_KNN/Blast_knn_result_parents.npy'
else:
    sys.exit('Data directories not found.')

torch.multiprocessing.set_sharing_strategy('file_system')

x_train,x_val,y_train,y_val,train_data, val_data, class_weights, num_classes, unique_go_values, idxMF, idxBP, idxCC = prepare_data(memmap_dir, train_val_ratio, num_loaded_proteins,max_protein_len, min_protein_len, func_threshold, pt_seq_dir ,verbose=verbose)
#generate fasta file of our training data
proteins = pd.read_csv(test_data_path)
protein_test = proteins['name']
x_test,y_test,train_data_test,unique_go_values_test,idxMF_test, idxBP_test, idxCC_test = prepare_test_data(memmap_dir,protein_test,max_protein_len, min_protein_len,func_threshold,pt_seq_dir)
print("Total functions: {}".format(num_classes))
print("Functions in MF: {}".format(len(idxMF)))
print("Functions in BP: {}".format(len(idxBP)))
print("Functions in CC: {}".format(len(idxCC)))
#
from blast_knn_model import BlastKnnModel
print("BLAST KNN MODEL")

blastKnnModel = BlastKnnModel(idxMF_test, idxBP_test, idxCC_test,x_test,x_test,unique_go_values_test,y_test,y_test)
BlastKnnModel.prepare_blast_result(blastKnnModel,BLAST_KNN_result_path)
# # gotermlookup = pd.read_csv(go_term_lookup_path)
#
test_prob,idxMF_all, idxBP_all, idxCC_all = blastKnnModel.blast()
#np.save("/usr/data/cvpr_shared/biology/function/CAFA3/training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/Blast_test_result_stage2.npy",test_prob)
np.save("Blast_test_result_stage2.npy", test_prob)
# print(np.shape(test_prob))

gotermlookup = pd.read_csv(go_term_lookup_path)
idxMF_all=np.where(gotermlookup["mainOntology"] == "F")[0]
#print(np.shape(idxMF_all[0]))
idxBP_all=np.where(gotermlookup["mainOntology"] == "P")[0]
idxCC_all=np.where(gotermlookup["mainOntology"] == "C")[0]
print("BLAST KNN MODEL ALL")
gt_test = np.load(gt_path)
#print(np.shape(gt_test))
test_prob = np.load("Blast_test_result_stage2.npy")
#print(np.shape(torch.squeeze(torch.FloatTensor(test_prob))))
eval_results_val = evaluate(torch.squeeze(torch.FloatTensor(test_prob)),torch.squeeze( torch.FloatTensor(gt_test)), idxMF_all, idxBP_all, idxCC_all)
print("-----------------------")
print("VALIDATION RESULTS:")
print("MF:\tF-max: {} Precision: {} Recall: {} Best threshold: {}".format(eval_results_val['MF']['f_max'],
                                                                          eval_results_val['MF']['precision'],
                                                                          eval_results_val['MF']['recall'],
                                                                          eval_results_val['MF']['threshold']))
print("BP:\tF-max: {} Precision: {} Recall: {} Best threshold: {}".format(eval_results_val['BP']['f_max'],
                                                                          eval_results_val['BP']['precision'],
                                                                          eval_results_val['BP']['recall'],
                                                                          eval_results_val['BP']['threshold']))
print("CC:\tF-max: {} Precision: {} Recall: {} Best threshold: {}".format(eval_results_val['CC']['f_max'],
                                                                          eval_results_val['CC']['precision'],
                                                                          eval_results_val['CC']['recall'],
                                                                          eval_results_val['CC']['threshold']))
print("-----------------------")


