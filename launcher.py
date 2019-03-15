import os
import sys
from data_loader import prepare_data
from argparse import ArgumentParser

# Add ProteinFunctionPrediction to sys.path for importing modules
sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))

# Parse arguments
parser = ArgumentParser()
parser.add_argument('-n', '--num_proteins', default=1000, dest="num_proteins", type=int)

args = parser.parse_args()

num_loaded_proteins = args.num_proteins   # Number of proteins to load

# path of datasets
if os.path.isdir('/usr/data/cvpr_shared/biology/function'):
    protein_data_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                            'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/proteins.csv'
    go_term_lookup_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                 'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/GOtermsLookup_file.csv'
    go_term_protein_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                 'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/GOtermsPerProtein_sparse.npz'
    BLAST_KNN_result_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                            'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/protein_fun_pred_WS19/BLAST_KNN/data/Blast_knn_result_parents.npy'
else:
    sys.exit('Data directories not found.')


protein_test_name,gt_test,unique_go_values,idxMF,idxBP,idxCC = prepare_data(protein_data_path,num_loaded_proteins,go_term_lookup_path,go_term_protein_path)

from blast_knn_model import BlastKnnModel
print("BLAST KNN MODEL")

#Initialize BlastKnnModel
blastKnnModel = BlastKnnModel(idxMF, idxBP, idxCC,protein_test_name,unique_go_values,gt_test,verbose = True)
BlastKnnModel.prepare_blast_result(blastKnnModel,BLAST_KNN_result_path)

#Running blast() and calculate f-score
test_prob,gt = blastKnnModel.blast()



