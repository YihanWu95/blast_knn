import numpy as np
import torch
import pandas as pd
from accuracy_measure import *


class BlastKnnModel:

    def __init__(self,idxMF, idxBP, idxCC,x_test,unique_go_values,y_test):
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
        self.y_test = torch.from_numpy(np.array(y_test)).float()
        self.x_test = x_test
        self.unique_go_values = unique_go_values
        # self.blast_knn_result = blast_knn_result

    def blast(self):
        '''
        loaded Blast_knn_result_parents.npy and generate prediction probabilities of selected proteins based on it and
        calculate corresponding f scores
        :return: prediction probabilities and ground truth after embedding into ground truth spaces
        '''
        #initial prediction matrix
        y_test_blast = []
        #loop over all train proteins to find their go term scores in Blast_knn_result_parents.npy
        for protein_name in self.x_test:
            np_predict = np.zeros((len(self.unique_go_values)), dtype=np.float32)
            for i, goterm in enumerate(self.unique_go_values):
                if protein_name not in self.blast_knn_result:
                    print("not found",protein_name)
                    np_predict[i] = 0
                elif self.blast_knn_result[protein_name].get(goterm):
                    np_predict[i] = self.blast_knn_result[protein_name][goterm]
            y_test_blast.append(np_predict)
        naive_probabilities_train = torch.FloatTensor(y_test_blast)
        eval_results_test = evaluate(naive_probabilities_train, self.y_test, self.idxMF, self.idxBP, self.idxCC)
        #print test results
        print("TEST RESULTS:")
        print("MF:\tF-max: {} Precision: {} Recall: {} Best threshold: {}".format(eval_results_test['MF']['f_max'],
                                                                                  eval_results_test['MF']['precision'],
                                                                                  eval_results_test['MF']['recall'],
                                                                                  eval_results_test['MF'][
                                                                                      'threshold']))
        print("BP:\tF-max: {} Precision: {} Recall: {} Best threshold: {}".format(eval_results_test['BP']['f_max'],
                                                                                  eval_results_test['BP']['precision'],
                                                                                  eval_results_test['BP']['recall'],
                                                                                  eval_results_test['BP'][
                                                                                      'threshold']))
        print("CC:\tF-max: {} Precision: {} Recall: {} Best threshold: {}".format(eval_results_test['CC']['f_max'],
                                                                                  eval_results_test['CC']['precision'],
                                                                                  eval_results_test['CC']['recall'],
                                                                                  eval_results_test['CC'][
                                                                                      'threshold']))
        print("-----------------------")
        return y_test_blast,np.array(self.y_test)


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

