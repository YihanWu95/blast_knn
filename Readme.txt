Protein Function Prediction from Co-evolution Data

This file explains the code structure of the project.

########
launcher.py
########

This is the main project file, which sets up and starts training runs. The launcher defines all necessary variables
(log paths etc.) and training parameters in one place. Processes such as data-loading and the start of the training
are started.

How to run? 'CUDA_VISIBLE_DEVICES=4 python launcher.py [<options>]' with the following possible options:
Setting the CUDA_VISIBLE_DEVICES environment variable to the device number of the GPU device that should be used for computation.
'-g', '--gpu': int : (deprecated) Identifier of the GPU to be used. Just set it to 0 if using CUDA_VISIBLE_DEVICES for GPU selection
'-n', '--num_proteins': int: Number of protein samples to be used during training
'-f', '--func_threshold': int: Minimum number of annotations for a function to be included in predictions
'-e', '--epochs': Number of epochs
'-l', '--lr': Learning rate for the Adam optimizer
'-o', '--owner': Text description to identify the respective result folder
'-w', '--workers': Number of processes to be used for data-loading

Some other variables which are not modified frequently are defined at the beginning of the file
(e.g. max_protein_len, top_n_samples, train_val_ratio, nr_channels, etc.)

For optimization, adam optimizer is used within this project.

Training results are currently saved in
'/usr/data/cvpr_shared/biology/function/CAFA3/training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/results/'

After each training run, following output-files are stored in the respective results folder:
- variables: Contains an overview of the values that were set for the training run
- checkpoint: Model checkpoint, used for training runs with pre-trained models
- loss_avg: Training and validation loss averaged over each epoch
- train_evaluation: Evaluation scores for the training set
- val_evaluation: Evaluation scores for the validation set
- gt.txt: Ground truth values
- pred_prob.txt: Prediction probabilities after the last completed epoch
- messages.log: should be used to print log messages


########
trainer.py
########

Includes the class Trainer, that stores relevant parts of the training model
(model setup, class_weights, unique_go_values, etc. )

The central class method 'train' starts the network training with the parameters set in the main launcher file.
We use a batch size of 1 because of memory constraints (single sample can be well over 1 GB) and varying input sizes
depending on the protein length.
After each training epoch, the network performance on the validation set is evaluated (see accuracy_measure.py)
and results are logged in the files described in the section above.


########
data_loader.py
########

Contains the central method 'prepare_data' and its utility functions, as well as the class CoevolutionData
(derived from the default 'Dataset' class in Pytorch) used for dynamic loading of samples during training.

'prepare_data' is called by the launcher before the actual training. Its main responsibility is to load,
compile and encode ground truth values of the training dataset. After loading the specified number of samples,
the annotations are filtered according to the function threshold.
The remaining functions are encoded in ground truth vectors and stored for each sample.
The dataset is split into training and validation sets, which are returned to the launcher in addition to class weights
and a list of the function identifiers
corresponding to the encoded ground truth vectors. Since we are evaluating the three main GO ontologies separately,
lists of ground truth indices for each of the ontologies are computed and returned as well.

Since the input data is potentially big (>1GB per sample), it's not possible to load big chunks of data into main memory
(much less the GPU). For this reason, the Dataset class 'CoevolutionData' implements dynamic data loading during the training.
When a specific sample is requested, the class loads it from the memory mapped files and returns it to the training script.


########
model.py
########

Contains the network definition.
So far, we were experimenting with 1-4 convolutional layers with filter sizes between 1 (for the first layer) and 7. Since the input data can have varying sizes depending on the protein length,
it is required to obtain a fixed size latent representation at some point in the network. We use global average pooling after the convolution steps, leading to a latent vector containing the same amount of features as we had feature layers after the last convolution operation.
After global pooling, we use blocks of fully-connected layers, each predicting 1000 classes each. Using multiple blocks instead of one big FC layer is helpful if the latent vector is very big. The idea then would be to split the vector up and use only parts of it
for each FC block. So far, our architectures didn't require this, though.


########
accuracy_measure.py
########

Computes the relevant evaluation metrics. Common metrics explained in https://arxiv.org/abs/1601.00891 are used for evaluation. Precision, recall and f-max score are calculated for the three main GO categories (Biological Process, Molecular Function, Cellular Component).
This breakdown is common standard and can give valuable insights into the performance for each single category.

For evaluation, prediction probabilities (floats between 0 and 1) and ground truth values (either 0 or 1) together with the links of unique GO terms to the corresponding GO categories are used.


########
visualizer.py
########

Contains utility logging and plotting functions used throughout the project.
