############################
# data_loader.py
#
# Contains
#	Data preparation: Filtering of annotations, compilation of ground truth vectors and class weights
#
# Adapted from previous team and Konstantin Weissenow (k.weissenow@tum.de)
# modified heavily by Yihan Wu (yihan.wu@tum.de)
############################

import os
import glob
import numpy as np
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
import torch
import torch.utils.data
from torch.autograd import Variable


def get_protein_names(memmap_dir):
    """
    Get proteins identifiers from file names.
    :param memmap_dir: string. Directory path to read files from.
    :return: protein_names: List of strings. List of protein identifiers.
    """
    goterms_ext = "J"
    protein_names = []
    for filename in os.listdir(memmap_dir):
        if goterms_ext not in filename:
            continue
        name_parts = filename.split('_')
        protein_names.append(name_parts[0])
    print("Find total ",len(protein_names)," proteins")
    return protein_names



def prepare_data(memmap_path,protein_names):
    coev_subdir = 'ccmpred/J_and_h/'
    go_terms_subdir = 'GOterms'
    coev_ext = '_J_'
    goterms_ext = '.GOterms.memmap'
    coev_memmap_dir = os.path.join(memmap_path, coev_subdir)
    go_terms_memmap_dir = os.path.join(memmap_path, go_terms_subdir)

    go_ontology_filename = os.path.join(memmap_path, 'Scripts/hierarchical_classification_GO_terms/go_db/term.txt')
    main_ontologies_per_term_dict = {}
    ontology_labels = {'molecular_function': 'F', 'biological_process': 'P', 'cellular_component': 'C'}
    with open(go_ontology_filename) as f:
        for line in f:
            (termID, description, ontology, GOID) = line.split(',\'')
            if GOID[:-2][:3] == 'GO:' and ontology[:-1] != '':
                main_ontologies_per_term_dict[int(GOID[:-2][3:])] = ontology_labels[ontology[:-1]]

    x_test = []  # List of proteins, where both coevolution and function memmap files exist (previously protein_names_existing)
    funcs_instances_counter = Counter()  # Counter for annotations per function over all loaded proteins
    funcs_dict = defaultdict(list)
    num_proteins = 0
    while num_proteins < len(protein_names):
        protein_name = protein_names[num_proteins]
        func_memmap_path = os.path.join(go_terms_memmap_dir, protein_name + goterms_ext)
        func_memmap_path = glob.glob(func_memmap_path)
        if not (func_memmap_path):  # (coev_memmap_path and func_memmap_path):
            continue
        # Process memmap file of protein annotations, build a counter with number of annotations for each function
        x_test.append(protein_name)
        func_memmap_path = func_memmap_path[0]
        func_memmap = np.memmap(func_memmap_path, dtype=np.int32, mode='r')
        for func in func_memmap:
            # if func in list(main_ontologies_per_term_dict.keys()):
            #    if (main_ontologies_per_term_dict[func] == 'F'):
            funcs_instances_counter[func] += 1
            funcs_dict[protein_name].append(func)
        del func_memmap

        # To be removed. Number of proteins to load, ideally load all ~40K proteins
        num_proteins += 1
        if num_proteins % 100 == 0:
            print("Loaded {} proteins".format(num_proteins))
    print("Finished loading {} proteins total".format(num_proteins))

    # Remove rare functions - we don't try to predict them


    # Sort GO terms in ascending order
    unique_go_values = sorted(funcs_instances_counter.keys())
    num_classes = len(unique_go_values)
    # Compiling ground truth vectors

    y_test = []
    for protein_name in x_test:
        np_ground_truth = np.zeros((num_classes), dtype=np.float32)
        for i, goterm in enumerate(unique_go_values):
            if goterm in funcs_dict[protein_name]:
                np_ground_truth[i] = 1.0
        y_test.append(np_ground_truth)
    # Optional function weighting
    # function_weights = calculate_label_weights(unique_go_values, funcs_instances_counter, num_proteins)
    # Preparing index lists for the three main ontologies
    idxMF = []
    idxBP = []
    idxCC = []
    for i, go_term in enumerate(unique_go_values):
        if go_term not in list(main_ontologies_per_term_dict.keys()):
            print("WARNING: Did not find {} in the ontology dictionary, skipping..".format(go_term))
            continue
        if main_ontologies_per_term_dict[go_term] == 'F':
            idxMF.append(i)
        if main_ontologies_per_term_dict[go_term] == 'P':
            idxBP.append(i)
        if main_ontologies_per_term_dict[go_term] == 'C':
            idxCC.append(i)
    return (
        x_test,
        y_test,
        unique_go_values,
        idxMF,
        idxBP,
        idxCC)
