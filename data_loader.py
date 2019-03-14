import numpy as np
import pandas as pd
import scipy.sparse as sp


def prepare_data(protein_data_path,num_proteins,go_term_lookup_path,go_term_protein_path):
    '''

    :param protein_data_path: path to test proteins file
    :param num_proteins: number of test proteins
    :param go_term_lookup_path: path to file GOtermsLookup_file.csv
    :param go_term_protein_path: path to file GOtermsLookup_file.csv
    :return: name of test proteins and ground truth
    '''
    print("loading proteins...")
    # load test protein files
    proteins = pd.read_csv(protein_data_path)

    # take out test sample name
    protein_test_name = proteins['name'][0:num_proteins]

    # get index of selected proteins, will be used in generating ground truth data
    selected_rows = proteins[proteins["name"].isin(protein_test_name)]["Unnamed: 0"].tolist()
    print("finishing loading...")
    print("preparing ground truth matrix...")

    # load gotermlookup file, which include go terms we need as ground truth
    gotermlookup = pd.read_csv(go_term_lookup_path)

    # take out all go term names
    unique_go_values = gotermlookup["termID"].tolist()
    # calculate indices of go terms corresponds to three different mainOntology: MF, BP and CC.
    idxMF = np.where(gotermlookup["mainOntology"] == "F")[0]
    idxBP = np.where(gotermlookup["mainOntology"] == "P")[0]
    idxCC = np.where(gotermlookup["mainOntology"] == "C")[0]

    # load the sparse matrix of ground truth, which include all proteins and all go_terms
    gt = sp.load_npz(go_term_protein_path)
    print(np.shape(gt))
    # selected the ground truth for test proteins
    gt_test = gt[selected_rows].todense()
    # print(np.shape(gt_test))
    print("ground truth matrix is ready!")
#    gt_test = gt_test[:,selected_columns]


    return (
        protein_test_name,
        gt_test,
        unique_go_values,
        idxMF,
        idxBP,
        idxCC)
