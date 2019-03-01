import numpy as np
import torch
import pandas as pd
from accuracy_measure import *


class BlastKnnModel:

    def __init__(self,idxMF, idxBP, idxCC,x_train,x_val,unique_go_values,y_train,y_val):
        self.idxMF = idxMF
        self.idxBP = idxBP
        self.idxCC = idxCC
        self.y_train = torch.from_numpy(np.array(y_train))
        self.y_val = torch.from_numpy(np.array(y_val))
        self.x_train = x_train
        self.x_val = x_val
        self.unique_go_values = unique_go_values
        # self.blast_knn_result = blast_knn_result

    def blast(self):
        y_train_blast = []
        # print(len(self.x_train))
        for protein_name in self.x_train:
#            print(self.blast_knn_result[protein_name])
            np_predict = np.zeros((len(self.unique_go_values)), dtype=np.float32)
            for i, goterm in enumerate(self.unique_go_values):
                if protein_name not in self.blast_knn_result:
                    print("not found",protein_name)
                    np_predict[i] = 0
                elif self.blast_knn_result[protein_name].get(goterm):
                # if goterm in funcs_dict[protein_name]:
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
        idxMF=np.where(gotermlookup["mainOntology"] == "F")
        idxBP=np.where(gotermlookup["mainOntology"] == "P")
        idxCC=np.where(gotermlookup["mainOntology"] == "C")
        mid1=np.reshape(y_train_blast,np.shape(y_train_blast))
        for i, goterm in enumerate(self.unique_go_values):
            m = np.where(gotermlookup["termID"] == goterm)[0]
            if m:
                # print(m,goterm)
                a = int(m)
                # a = gotermlookup.loc[gotermlookup.termID == goterm]
                # print(a)
                # print(np.shape(mid1))
                # print(np.shape([row[i] for row in y_train_blast])
                # print(np.shape(pred_prob[:, a]))
                pred_prob[:, a] = mid1[:,i]

        #end add

        y_val_blast = []
        for protein_name in self.x_val:
            np_predict = np.zeros((len(self.unique_go_values)), dtype=np.float32)
            # print(self.blast_knn_result[protein_name])
            for i, goterm in enumerate(self.unique_go_values):
                if protein_name not in self.blast_knn_result:
                    print("not found", protein_name)
                    np_predict[i] = 0
                #print(self.blast_knn_result[protein_name])
                elif self.blast_knn_result[protein_name].get(goterm):
                    # if goterm in funcs_dict[protein_name]:
                    np_predict[i] = self.blast_knn_result[protein_name][goterm]
            y_val_blast.append(np_predict)
        naive_probabilities_val = torch.FloatTensor(y_val_blast)
        eval_results_val = evaluate(naive_probabilities_val, self.y_val, self.idxMF, self.idxBP, self.idxCC)
        print("-----------------------")
        print("VALIDATION RESULTS:")
        print("MF:\tF-max: {} Precision: {} Recall: {} Best threshold: {}".format(eval_results_val['MF']['f_max'],
                                                                                  eval_results_val['MF']['precision'],
                                                                                  eval_results_val['MF']['recall'],
                                                                                  eval_results_val['MF']['threshold']))
        print("BP:\tF-max: {} Precision: {} Recall: {} Best threshold: {}".format(eval_results_val['BP']['f_max'],
                                                                                  eval_results_val['BP']['precision'],
                                                                                  eval_results_val['BP']['recall'],
                                                                                  eval_results_val['BP']['threshold']))
        print("CC:\tF-max: {} Precision: {} Recall: {} Best threshold: {}".format(eval_results_val['CC']['f_max'],
                                                                                  eval_results_val['CC']['precision'],
                                                                                  eval_results_val['CC']['recall'],
                                                                                  eval_results_val['CC']['threshold']))
        print("-----------------------")



        return pred_prob,idxMF,idxBP,idxCC


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
            #i=0
            #for w in sorted(mid, key=mid.get, reverse=False):
            #    if(i>20):
            #        break
            #    i+=1
            #    result[key][w] = 1
        self.blast_knn_result = result

        # for w in sorted(d, key=d.get, reverse=True):
