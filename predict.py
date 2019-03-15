import os
import sys
import pandas as pd
from data_loader import prepare_data
from blast_knn_model import BlastKnnModel

#for combining models, reture probability matrix for test proteins.
def predict(protein_data_path):
    if os.path.isdir('/usr/data/cvpr_shared/biology/function'):
        # protein_data_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
        #                     'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/proteins.csv'
        go_term_lookup_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                              'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/GOtermsLookup_file.csv'
        go_term_protein_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                               'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/GOtermsPerProtein_sparse.npz'
        BLAST_KNN_result_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                                'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/protein_fun_pred_WS19/BLAST_KNN/data/Blast_knn_result_parents.npy'
    else:
        sys.exit('Data directories not found.')
    proteins = pd.read_csv(protein_data_path)
    num_loaded_proteins = len(proteins['name'])
    protein_test_name, gt_test, unique_go_values, idxMF, idxBP, idxCC = prepare_data(protein_data_path,
                                                                                     num_loaded_proteins,
                                                                                     go_term_lookup_path,
                                                                                     go_term_protein_path)

    blastKnnModel = BlastKnnModel(idxMF, idxBP, idxCC, protein_test_name, unique_go_values, gt_test, verbose=False)
    BlastKnnModel.prepare_blast_result(blastKnnModel, BLAST_KNN_result_path)
    test_prob, gt = blastKnnModel.blast()
    return test_prob