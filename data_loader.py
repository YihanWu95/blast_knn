############################
# data_loader.py
#
# Contains
#	Data preparation: Filtering of annotations, compilation of ground truth vectors and class weights
#
# Adapted from previous team and modified heavily by Yihan Wu (yihan.wu@tum.de)
############################

import numpy as np
import pandas as pd
import scipy.sparse as sp


def prepare_data(protein_data_path,num_proteins,go_term_lookup_path,go_term_protein_path):
    print("loading proteins...")
    proteins = pd.read_csv(protein_data_path)
    protein_test_name = proteins['name'][0:num_proteins]
    selected_rows = proteins[proteins["name"].isin(protein_test_name)]["Unnamed: 0"].tolist()
    print("finishing loading...")
    print("preparing ground truth matrix...")
    gotermlookup = pd.read_csv(go_term_lookup_path)
    unique_go_values = gotermlookup["termID"].tolist()
#    selected_columns = gotermlookup["index"].tolist()
    idxMF = np.where(gotermlookup["mainOntology"] == "F")[0]
    idxBP = np.where(gotermlookup["mainOntology"] == "P")[0]
    idxCC = np.where(gotermlookup["mainOntology"] == "C")[0]


    gt = sp.load_npz(go_term_protein_path)
    print(np.shape(gt))
    # groundtruth matrix
    gt_test = gt[selected_rows].todense()
    print(np.shape(gt_test))
    print("ground truth matrix is ready!")
#    gt_test = gt_test[:,selected_columns]


    return (
        protein_test_name,
        gt_test,
        unique_go_values,
        idxMF,
        idxBP,
        idxCC)
