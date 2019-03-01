import os
import sys
import torch
import logging
from data_loader import prepare_data,prepare_test_data
from trainer import Trainer
from model import Model
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
parser.add_argument('-g', '--gpu', default=0, dest="gpu", type=int)
parser.add_argument('-n', '--num_proteins', default=1000, dest="num_proteins", type=int)
parser.add_argument('-mL', '--max_protein_len', default=1000, dest="max_protein_len", type=int)
parser.add_argument('-ml', '--min_protein_len', default=1000, dest="min_protein_len", type=int)
parser.add_argument('-wd', '--weight_decay', default=1e-3, dest="weight_decay", type=float)
parser.add_argument('-f', '--func_threshold', default=5, dest="func_threshold", type=int)
parser.add_argument('-e', '--epochs', default=5, dest="epochs", type=int)
parser.add_argument('-l', '--lr', default=1e-2, dest="learning_rate", type=float)
parser.add_argument('-o', '--owner', default='', dest='owner', type=str)
parser.add_argument('-w', '--workers', default=0, dest='workers', type=int)
args = parser.parse_args()

num_loaded_proteins = args.num_proteins   # Number of proteins to load
max_protein_len = args.max_protein_len
min_protein_len = args.min_protein_len
func_threshold = args.func_threshold          # Function count threshold. Encode only functions that cross this threshold
numWorkers = args.workers              # how many subprocesses to use for data loading
verbose = True
load_from_checkpoint = False
num_epochs = args.epochs
#max_protein_len = 500  # Coev matrices only computed for this length
top_n_samples = 5
train_val_ratio = 0.8
nr_channels = 461   # input data (co-evolution matrices in form of LxLx441 - L: protein's amino acid sequence length)

# Default arguments for adam optimizer
adam_lr = args.learning_rate  # learning rate: was 1e-4 before
adam_betas = (0.9,0.999)   # decay rates for running avg of first and second moments
adam_eps = 1e-8  # to avoid division by 0
adam_weight_decay = args.weight_decay  # weight decay
default_adam_args = {"lr": adam_lr,
                     "betas": adam_betas,
                     "eps": adam_eps,
                     "weight_decay": adam_weight_decay}


load_from_ckpt_path = ''
current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# Join results_path and loss_path_name to log_path
results_path = '/usr/prakt/w0184/Desktop/DLproject/result'
loss_path_name = 'result_' + args.owner + '_%s' % current_time
log_path = os.path.join(results_path, loss_path_name)

train_loss_file = os.path.join(log_path, 'train_loss')
loss_avg_file = os.path.join(log_path, 'loss_avg')
val_loss_file = os.path.join(log_path, 'val_loss')
val_loss_avg_file = os.path.join(log_path, 'val_loss_avg')
train_eval_file = os.path.join(log_path, 'train_evaluation')
val_eval_file = os.path.join(log_path, 'val_evaluation')
train_gt_and_pred_file = os.path.join(log_path, 'train_gt_and_pred')
val_gt_and_pred_file = os.path.join(log_path, 'val_gt_and_pred')
checkpoint_file_path = os.path.join(log_path, 'checkpoint')
variables_file = os.path.join(log_path, 'variables')
log_messages_file = os.path.join(log_path, 'messages.log')

log_file_dict = {
    'log_path': log_path,
    'train_loss_file': train_loss_file,
    'loss_avg_file': loss_avg_file,
    'val_loss_file': val_loss_file,
    'val_loss_avg_file': val_loss_avg_file,
    'train_eval_file': train_eval_file,
    'val_eval_file': val_eval_file,
    'train_gt_and_pred_file': train_gt_and_pred_file,
    'val_gt_and_pred_file': val_gt_and_pred_file,
    'checkpoint_file_path': checkpoint_file_path,
    'variables': variables_file,
    'messages': log_messages_file
}
if not os.path.isdir(log_file_dict['log_path']):
    try:
        os.makedirs(log_file_dict['log_path'])
    except OSError as exc:  # Guard against race condition
        raise

logging.basicConfig(filename=log_messages_file, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print('Log path: {}'.format(log_path))

if not torch.cuda.is_available():
    print("CUDA is not available, exiting..")
    quit()
print('Cuda is available: {}'.format(torch.cuda.is_available()))
print("Number of GPUs: {}".format(torch.cuda.device_count()))
device = "cuda:" + str(args.gpu)
print("Using device '{}'".format(device))

if os.path.isdir('/usr/data/cvpr_shared/biology/function'):
    data_dir = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
               'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/'
    memmap_dir = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                 'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/'
    #added YW
    pt_seq_dir = '/usr/data/cvpr_shared/biology/function/CAFA3/training_data/clustered_70seqid/fa/'
    test_data_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                 'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/test/proteins_holdout_stage2.csv'
    gt_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                 'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/test/gt_test_holdout_stage2.npy'
    # go_term_lookup_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
    #              'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/GOtermsLookup_file.csv'
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
path = r"Blast_knn_result_parents.npy"

blastKnnModel = BlastKnnModel(idxMF_test, idxBP_test, idxCC_test,x_test,x_test,unique_go_values_test,y_test,y_test)
BlastKnnModel.prepare_blast_result(blastKnnModel,path)
# # gotermlookup = pd.read_csv(go_term_lookup_path)
#
test_prob,idxMF_all, idxBP_all, idxCC_all = blastKnnModel.blast()
#np.save("/usr/data/cvpr_shared/biology/function/CAFA3/training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/Blast_test_result_stage2.npy",test_prob)
np.save("Blast_test_result_stage2.npy", test_prob)
# print(np.shape(test_prob))
go_term_lookup_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                              'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/GOtermsLookup_file.csv'
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
# #added YW
# from naive_model import NaiveModel
# print("NAIVE MODEL")
# naiveModel = NaiveModel(idxMF, idxBP, idxCC,class_weights,y_train,y_val)
# naiveModel.naive_model()
# from blast_knn_model import BlastKnnModel
# print("BLAST KNN MODEL")
# path = r"Blast_knn_result_parents.npy"
#
# blastKnnModel = BlastKnnModel(idxMF, idxBP, idxCC,x_train,x_val,unique_go_values,y_train,y_val)
# BlastKnnModel.prepare_blast_result(blastKnnModel,path)
# blastKnnModel.blast()
# #end added
# #added YW
# from trigram_model import TrigramModel
# print("TRIGRAM MODEL")
# trigramModel = TrigramModel(idxMF, idxBP, idxCC,x_train,x_val,unique_go_values,y_train,y_val)
# path = r"protein_trigram.npy"
# TrigramModel.prepare_trigram_result(trigramModel,path)
# trigramModel.trigram()
# path = r"Blast_knn_result.npy"



#end added



# Create DataLoader for training data
train_loader=torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=numWorkers, pin_memory=True)

# Create DataLoader for validation data
val_loader=torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=numWorkers, pin_memory=True)


# Load existing or create new model
if load_from_checkpoint:
    if verbose:
        print('Load model from checkpoint')
    model = torch.load(load_from_ckpt_path)
else:
    if verbose:
        print('Create model')
    model = Model(num_classes)
    model.initialize_weights()

# Create new trainer object
trainer = Trainer(model, device, class_weights, unique_go_values, idxMF, idxBP, idxCC, args_adam=default_adam_args)

# Parameters that do require gradient update
model_parameters = filter(lambda p: p.requires_grad, model.parameters())

# Print number of parameters that are to be optimized?
params = sum([np.prod(p.size()) for p in model_parameters])
if verbose:
    print('Number of parameters to be optimized: ', params)

with open(variables_file, "a") as f:
    f.write('log_path:             ' + str(log_path) + '\n' +
            'checkpoint:           ' + str(load_from_checkpoint) + '\n' +
            'num_workers:          ' + str(numWorkers) + '\n' +
            'adam_lr:              ' + str(adam_lr) + '\n' +
            'adam_betas:           ' + str(adam_betas) + '\n' +
            'adam_eps:             ' + str(adam_eps) + '\n' +
            'adam_weight_decay:    ' + str(adam_weight_decay) + '\n' +
            'num_loaded_proteins:  ' + str(num_loaded_proteins) + '\n' +
            'train_proteins:       ' + str(len(train_loader.dataset)) + '\n' +
            'val_proteins:         ' + str(len(val_loader.dataset)) + '\n' +
            'maxProtLen:           ' + str(max_protein_len) + '\n' +
            'minProtLen:           ' + str(min_protein_len) + '\n' +
            '#epochs:              ' + str(num_epochs) + '\n' +
            'top_n_sampels:        ' + str(top_n_samples) + '\n' +
            'train_val_r:          ' + str(train_val_ratio) + '\n' +
            'nr_channels:          ' + str(nr_channels) + '\n' +
            'func_threshold:       ' + str(func_threshold) + '\n' +
            'num_classes:          ' + str(num_classes) + '\n' +
            'unique_GO_terms:      ' + str(unique_go_values) + '\n' +
            'idxMF                 ' + str(idxMF)+ '\n' +
            'idxBP                 ' + str(idxBP)+ '\n' +
            'idxCC                 ' + str(idxCC)+ '\n'
            )
    f.close()

# Train model
if verbose:
    print('Start training model...')
trainer.train(train_loader, val_loader, max_protein_len, num_epochs, log_file_dict, verbose)


# #added YW
# from naive_model import NaiveModel
# print("NAIVE MODEL")
# naiveModel = NaiveModel(idxMF, idxBP, idxCC,class_weights,y_train,y_val)
# naiveModel.naive_model()
# from blast_knn_model import BlastKnnModel
# print("BLAST KNN MODEL")
# path = r"Blast_knn_result.npy"
#
# blastKnnModel = BlastKnnModel(idxMF, idxBP, idxCC,x_train,x_val,unique_go_values,y_train,y_val)
# BlastKnnModel.prepare_blast_result(blastKnnModel,path)
# blastKnnModel.blast()
# #end added
