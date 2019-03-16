Protein Function Prediction from Co-evolution Data

Author: Yihan Wu (yihan.wu@tum.de)

This file explains the datasets of the subproject BLAST-KNN model.

All the necessary data is stored on the server under '/usr/data/cvpr_shared/biology/function/CAFA3/training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/protein_fun_pred_WS19/BLAST_KNN/data'

But if one wants to run sth locally, you can copy the data from there and store the data here. For example for running the notebooks. Following a short explanation of the important data files:


########
GOtermsPerProtein_sparse.npz
########

sparse matrix that stores the annotated GO terms for all the proteins. Shape: (34349(proteins), 23663(GO terms)) Indices of rows correspond to the indices of proteins dataframe (proteins.csv) Indices of columns correspond to the indices of the GO terms lookup dataframe (GOtermsLookup_file.csv)

########
GOtermsLookup_file.csv
########

lookup for all the GO terms present in our dataset. Holds information about the GO terms' main ontology as well as the index of each GO term in the GOtermsPerProtein_sparse matrix.
!!REMARK!!
The column "index" refers to an old index, when looking at all ~40k GO terms. The column index in the GOtermsPerProtein_sparse matrix corresponds to the row number of this dataframe



########
proteins.csv
########

lookup for all the proteins in our dataset. Contains "cluster names" and uniProt IDs as well as the length of the protein sequence. The column "Unnamed: 0" is the index of the protein in the GOtermsPerProtein_sparse matrix

--> The above files can be created with coevolution_network/src/data/prepare_data.py from the data files on the server.


########
Blast_knn_result_parents.npy
########
Go term score of all the proteins in our dataset, store in numpy dictionary.
key is protein name like 'clust-xxxxx-00001' and value include:
'accession': accession number of proteins
'score' : dictionary of go terms and their scores.

For example, accessing the score of 'GO:000001' w.r.t protein 'clust-xxxxx-00001' one can do
Blast_knn_result_parents = np.load(path).item()
Blast_knn_result_parents['clust-xxxxx-00001']['score']['GO:000001']

calculated by Blast-knn in BLAST_data_prepare






