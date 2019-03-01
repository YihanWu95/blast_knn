############################
# data_loader.py
#
# Contains
#	Data preparation: Filtering of annotations, compilation of ground truth vectors and class weights
#	CoevolutionData: Dataset class used in the Pytorch DataLoader, implementing on demand loading of mememory mapped coevolution files
#
# Adapted from previous team and modified heavily by Konstantin Weissenow (k.weissenow@tum.de)
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


def train_val_split(x, y, train_val_ratio, num_proteins):
    """
    :param x: List of strings. Protein names
    :param y: List of arrays. Encoded ground truth.
    :param train_val_ratio: float. Splitting ratio for train/validation sets.
    :param num_proteins: Integer. Total number of proteins used.
    :return: Tuple of splits
    """

    num_train_data = int(train_val_ratio * num_proteins)

    x_train = x[:num_train_data]
    y_train = y[:num_train_data]

    x_val = x[num_train_data:]
    y_val = y[num_train_data:]

    return (x_train, y_train, x_val, y_val)


def filter_functions(funcs_instances_counts, func_threshold):
    """
    Remove functions with annotations fewer than threshold
    :param funcs_instances_counts: Counter. Has a count of instances for each function.
    :return:Counter. With rare functions removed.
    """
    for func in list(funcs_instances_counts):
        if funcs_instances_counts[func] < func_threshold:
            del funcs_instances_counts[func]
    return funcs_instances_counts


def calculate_label_weights(unique_values, function_count_dict, num_samples):
    """
    Calculate label weights used to deal with imbalanced dataset.
    The more often a function was encountered, the smaller the weight will be.
    weight = count_all - count_fct/count_all
    :param unique_values: list of integers. Unique GO labels that are encoded
    :param function_count_dict: dictionary containing GO labels and num_occurances of GO labels.
    :return: 1-D FloatTensor containing weights for each class
    """
    function_weights = torch.FloatTensor(len(unique_values))
    for i,f in enumerate(unique_values):
        function_weights[i] = float(num_samples - function_count_dict[f]) / num_samples
        # DEBUG
        print("Class: {} Annotations: {} Weight: {}".format(f, function_count_dict[f], function_weights[i]))
    return function_weights


def prepare_data(memmap_path, train_val_ratio, num_loaded_proteins):
    """
    Prepare data
    :param memmap_path: string
    :param train_val_test_ratio: array of float. Train, val and test ratios for splitting data.
    :param num_loaded_proteins:
    :param prot_pack:
    :param func_threshold
    :param verbose: verbose
    :return: train, test and validation data and encoded function information
    """


    coev_subdir = 'ccmpred/J_and_h/'
    go_terms_subdir = 'GOterms'
    coev_ext = '_J_'
    goterms_ext = '.GOterms.memmap'
    coev_memmap_dir = os.path.join(memmap_path, coev_subdir)
    go_terms_memmap_dir = os.path.join(memmap_path, go_terms_subdir)

    # TODO: 'go_ontologies.txt' is currently in the memmap_path, but should logically be in the go_terms_subdir, which is not writeable for me (Konstantin, 7.8.2018)
    go_ontology_filename = os.path.join(memmap_path, 'Scripts/hierarchical_classification_GO_terms/go_db/term.txt')
    main_ontologies_per_term_dict = {}
    ontology_labels = {'molecular_function': 'F', 'biological_process': 'P', 'cellular_component': 'C'}
    with open(go_ontology_filename) as f:
        for line in f:
            (termID, description, ontology, GOID) = line.split(',\'')
            if GOID[:-2][:3] == 'GO:' and ontology[:-1] != '':
                main_ontologies_per_term_dict[int(GOID[:-2][3:])] = ontology_labels[ontology[:-1]]


    if verbose:
        print('Searching in {0} for coev-data.\nSearching in {1} for GO terms.'.format(coev_memmap_dir, go_terms_memmap_dir))
    protein_names = get_protein_names(coev_memmap_dir)  # List of protein identifiers
   # print(protein_names[1000])
    if verbose:
        print('Found {0} proteins to be processed.'.format(len(protein_names)))

    x = [] # List of proteins, where both coevolution and function memmap files exist (previously protein_names_existing)
    funcs_instances_counter = Counter()  # Counter for annotations per function over all loaded proteins
    funcs_dict = defaultdict(list)  # Dictionary of lists with protein identifiers as keys and functions as values (order of functions can be arbitrary!)

    num_proteins = 0
    num_total_loaded_proteins = 0;
    while num_proteins < num_loaded_proteins:
        num_total_loaded_proteins += 1
        # print("num_total_loaded_proteins", num_total_loaded_proteins)
        protein_name = protein_names[num_total_loaded_proteins+1000]
        # print(protein_name)
        # print("num_loaded_proteins",num_loaded_proteins)
        #print("num_proteins", num_proteins)
    # for protein_name in protein_names:  # Go through the list of protein identifiers
#        coev_memmap_path = os.path.join(coev_memmap_dir, protein_name+coev_ext+'*.memmap')
#        coev_memmap_path = glob.glob(coev_memmap_path)
        func_memmap_path = os.path.join(go_terms_memmap_dir, protein_name + goterms_ext)
        func_memmap_path = glob.glob(func_memmap_path)
        if not (func_memmap_path): # (coev_memmap_path and func_memmap_path):
            continue
#        name_parts = coev_memmap_path[0].split('_')
#        prot_len = int(name_parts[-5])
#        if (prot_len <= max_protein_len and prot_len>= min_protein_len ):
#         x.append(protein_name)
#        elif (prot_len > max_protein_len):
#            print('skipping current sample, protlen = %s > %s ' % (prot_len, max_protein_len))
#            continue
#        elif (prot_len < min_protein_len):
#            print('skipping current sample, protlen = %s < %s ' % (prot_len, min_protein_len))
#            continue
        # Process memmap file of protein annotations, build a counter with number of annotations for each function
        x.append(protein_name)
        func_memmap_path = func_memmap_path[0]
        func_memmap = np.memmap(func_memmap_path, dtype=np.int32, mode='r')
        for func in func_memmap:
            #if func in list(main_ontologies_per_term_dict.keys()):
            #    if (main_ontologies_per_term_dict[func] == 'F'):
            funcs_instances_counter[func] += 1
            funcs_dict[protein_name].append(func)
        del func_memmap

        # To be removed. Number of proteins to load, ideally load all ~40K proteins
        num_proteins += 1
        # print("num_loaded_proteins", num_loaded_proteins)
        # if num_proteins >= num_loaded_proteins:
        #    break
        if num_proteins % 100 == 0:
            print("Loaded {} proteins".format(num_proteins))
    print("Finished loading {} proteins total".format(num_proteins))


    # Remove rare functions - we don't try to predict them
    print("Total unique classes: {}".format(len(funcs_instances_counter)))
    funcs_instances_counter = filter_functions(funcs_instances_counter, func_threshold)
    print("Total unique classes after filtering: {}".format(len(funcs_instances_counter)))

    # Sort GO terms in ascending order
    unique_go_values = sorted(funcs_instances_counter.keys())
    num_classes = len(unique_go_values)

    # Compiling ground truth vectors
    y = []
    for protein_name in x:
        np_ground_truth = np.zeros((num_classes), dtype=np.float32)
        for i,goterm in enumerate(unique_go_values):
            if goterm in funcs_dict[protein_name]:
                np_ground_truth[i] = 1.0
        y.append(np_ground_truth)

    # Optional function weighting
    function_weights = calculate_label_weights(unique_go_values, funcs_instances_counter, num_proteins)

    # Splitting datasets
    x_train, y_train, x_val, y_val = train_val_split(x, y, train_val_ratio, num_proteins)

    # Preparing index lists for the three main ontologies
    idxMF = []
    idxBP = []
    idxCC = []
    for i,go_term in enumerate(unique_go_values):
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
        x_train,
        x_val,
        y_train,
        y_val,
        num_classes,
        unique_go_values,
        idxMF,
        idxBP,
        idxCC
    )

