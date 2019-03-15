Protein Function Prediction from Co-evolution Data

Author: Yihan Wu (yihan.wu@tum.de)

This file explains the code structure of the subproject BLAST-KNN model.


########
launcher.py
########

This is the main project file, which sets up and starts training runs. The launcher defines all necessary variables
(log paths etc.) and training parameters in one place. Processes such as data-loading and the start of the training
are started.

How to run? 'python launcher.py [<options>]' with the following possible options:


'-n', '--num_proteins': int: Number of protein samples to be used during training.


########
data_loader.py
########

Contains the central method 'prepare_data' and its utility functions,

'prepare_data' is called by the launcher before the actual training. Its main responsibility is to load,
compile and encode ground truth values of the training dataset.

The go term functions are encoded in ground truth vectors and stored for each sample.

corresponding to the encoded ground truth vectors. Since we are evaluating the three main GO ontologies separately,
lists of ground truth indices for each of the ontologies are computed and returned as well.


########
blast_knn_model.py
########

Compute go term score for each protein in the training set. Since I have generate a numpy file (Blast_knn_result_parents.npy) which contains all proteins and their go term score.
Generating prediction matrix for training proteins is kind of easy -- just extract training proteins and go terms score from 'Blast_knn_result_parents.npy'


########
accuracy_measure.py
########

Computes the relevant evaluation metrics. Common metrics explained in https://arxiv.org/abs/1601.00891 are used for evaluation. Precision, recall and f-max score are calculated for the three main GO categories (Biological Process, Molecular Function, Cellular Component).
This breakdown is common standard and can give valuable insights into the performance for each single category.

For evaluation, prediction probabilities (floats between 0 and 1) and ground truth values (either 0 or 1) together with the links of unique GO terms to the corresponding GO categories are used.

