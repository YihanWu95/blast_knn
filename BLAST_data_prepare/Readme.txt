Protein Function Prediction from Co-evolution Data

Author: Yihan Wu (yihan.wu@tum.de)

This file explains the preparation of dataset for BLAST-KNN.

Important: If one want to run blast locally, please download blast software from 'https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download'
Then download the corresponding database (for me I download the swissprot database)

All the necessary data is stored on the server under '/usr/data/cvpr_shared/biology/function/CAFA3/training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/protein_fun_pred_WS19/BLAST_KNN/BLAST_data_prepare'

But if one wants to run sth locally, you can copy the data from there and store the data here. For example for running the notebooks. Following a short explanation of the important data files and code:

P.S. Running BLAST on all training proteins costs one day or more, be patient and good luck;)

########
ProteinWithGo.xml
########
This file is downloaded with 'curl -O ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz'
and preprocessed with 
'grep -e '^<entry' 'entry>' 'type="GO"' 'accession' > ProteinWithGo.txt'
'sed 's/\([0-9]*"\)\(>\)/\1\/\2/g' ProteinWithGo.txt '
In case some of you are not familiar with GNU tools, we put the file after processing here. This is the input file of "prepare_uniprot_swissprot_data()"

########
SwissprotProteinWithGo.npy
########
Contains 558898 proteins of Swissprot with their go terms. Generated from "prepare_uniprot_swissprot_data(ProteinWithGo.xml)" in Blast-knn.ipynb

This dataset is stored in dictionary with key = "accession number of proteins" and value = "list of go terms w.r.t the protein"
For example key = "Q8BQM9" value = ['GO:000001','GO:000002']

Load the data in python with SwissprotProteinWithGo = np.load(path).item()
Access go term of "Q8BQM9": SwissprotProteinWithGo["Q8BQM9"]

Warning, the accession key here is not the protein name "cluster-11585-00001"

########
SwissprotProteinWithGoParents.npy 
########
Contains 558898 proteins of Swissprot with their go terms and go parents terms. Generated from "prepare_uniprot_swissprot_data_parents()" in Blast-knn.ipynb

########
goterm_pairs.npy
########
Contain all go terms and their parent terms, since our ground truth is go terms plus all of their parents in GO ontology tree, we need also add them into our blast result. Generated from "generate_go_parents_terms()" in go_parent_term.ipynb

This dataset is stored in dictionary with key = "go term" and value = "list of go terms "
For example key = 'GO:000001' value = ['GO:000001','GO:000002']

Load the data in python with SwissprotProteinWithGo = np.load(path).item()
Access go term of "Q8BQM9": SwissprotProteinWithGo["Q8BQM9"]

########
Blast-knn.ipynb
########

Running blast for all proteins and calculating their corresponding go term scores.

Includes 
prepare_uniprot_swissprot_data()
prepare_uniprot_swissprot_data_parents()
This two function is used to generate proteins in Swissprot with their go terms and go parents terms.

blast_knn_output()
This function runs blast on ONE single protein

calculate_Blastknn_score()
This function calculate weighted average score for ONE protein

calculate_Blastknn_score_forall()
Go through all proteins and using former 2 functions to calculate corresponding go term scores.


########
go_parent_term.ipynb
########
Mainly used to find parents for all go terms in GO ontology tree.

