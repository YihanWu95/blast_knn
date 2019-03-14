import numpy as np
import torch
import pandas as pd
from accuracy_measure import *


class BlastKnnModel:

    def __init__(self,idxMF, idxBP, idxCC,x_train,unique_go_values,y_train):
        '''
        :param idxMF: List of indices for terms in ground truth for Molecular Function
        :param idxBP: List of indices for terms in ground truth for Biological Process
        :param idxCC: List of indices for terms in ground truth for Cellular Component
        :param x_train: name of rotein for training
        :param unique_go_values: go terms names used in ground truth
        :param y_train: List of arrays. Encoded ground truth.
        '''
        self.idxMF = idxMF
        self.idxBP = idxBP
        self.idxCC = idxCC
        self.y_train = torch.from_numpy(np.array(y_train)).float()
        self.x_train = x_train
        self.unique_go_values = unique_go_values
        # self.blast_knn_result = blast_knn_result

    def blast(self):
        '''
        loaded Blast_knn_result_parents.npy and generate prediction probabilities of selected proteins based on it and
        calculate corresponding f scores
        :return: prediction probabilities and ground truth after embedding into groundtruth spaces
        '''
        #initial prediction matrix
        y_train_blast = []
        #loop over all train proteins to find their go term scores in Blast_knn_result_parents.npy
        for protein_name in self.x_train:
            np_predict = np.zeros((len(self.unique_go_values)), dtype=np.float32)
            for i, goterm in enumerate(self.unique_go_values):
                if protein_name not in self.blast_knn_result:
                    print("not found",protein_name)
                    np_predict[i] = 0
                elif self.blast_knn_result[protein_name].get(goterm):
                    np_predict[i] = self.blast_knn_result[protein_name][goterm]
            y_train_blast.append(np_predict)
        naive_probabilities_train = torch.FloatTensor(y_train_blast)
        eval_results_train = evaluate(naive_probabilities_train, self.y_train, self.idxMF, self.idxBP, self.idxCC)
        print("TRAIN RESULTS:")
        print("MF:\tF-max: {} Precision: {} Recall: {} Best threshold: {}".format(eval_results_train['MF']['f_max'],
                                                                                  eval_results_train['MF']['precision'],
                                                                                  eval_results_train['MF']['recall'],
                                                                                  eval_results_train['MF'][
                                                                                      'threshold']))
        print("BP:\tF-max: {} Precision: {} Recall: {} Best threshold: {}".format(eval_results_train['BP']['f_max'],
                                                                                  eval_results_train['BP']['precision'],
                                                                                  eval_results_train['BP']['recall'],
                                                                                  eval_results_train['BP'][
                                                                                      'threshold']))
        print("CC:\tF-max: {} Precision: {} Recall: {} Best threshold: {}".format(eval_results_train['CC']['f_max'],
                                                                                  eval_results_train['CC']['precision'],
                                                                                  eval_results_train['CC']['recall'],
                                                                                  eval_results_train['CC'][
                                                                                      'threshold']))
        print("-----------------------")

        # embedding test probability matrix into (num.proteins,num.totalgoterms)
        go_term_lookup_path = '/usr/data/cvpr_shared/biology/function/CAFA3/' \
                              'training_data/clustered_70seqid/hhblits_n5_uniclust30_2016_03/data_protein_pred/GOtermsLookup_file.csv'
        gotermlookup = pd.read_csv(go_term_lookup_path)
        pred_prob = np.zeros((len(self.x_train), len(gotermlookup["termID"])))
        gt = np.zeros((len(self.x_train), len(gotermlookup["termID"])))
        mid1=np.reshape(y_train_blast,np.shape(y_train_blast))
        mid2 = np.reshape(self.y_train, np.shape(self.y_train))
        for i, goterm in enumerate(self.unique_go_values):
            m = np.where(gotermlookup["termID"] == goterm)[0]
            if m:
                a = int(m)
                pred_prob[:, a] = mid1[:,i]
                gt[:, a] = mid2[:, i]

        #end add



        return pred_prob,gt


    def prepare_blast_result(self,path):
        blast_knn_result = np.load(path).item()
        result = {}
        # result_final = {}
        for key in blast_knn_result:
            result[key] = {}
            # result_final[key] = {}
            mid = blast_knn_result[key]['score']
            # print('max',key,mid[max(mid, key=mid.get)])
            for go_name in blast_knn_result[key]['score']:
                new_go_name = int(go_name.split(':')[1])
                #print('max',go_name,mid[max(mid, key=mid.get)])
                result[key][new_go_name]  = mid[go_name]  #blast_knn_result[key]['score'][go_name]/
        self.blast_knn_result = result

        # for w in sorted(d, key=d.get, reverse=True):
