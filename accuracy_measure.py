import torch
import numpy as np

# threshold == tau from paper 1601.00891
def evaluate_pr_rc(prediction_prob, ground_truth):
    """
    Evaluate accuracy based on precision and ground truth
    :param prediction_prob: predictions for all proteins [N*C]
    :param ground_truth: ground truth encoded with multiple labels [N*C]
    :return: accuracy as f measure described in literature (1601.00891)
    """

    f_max = 0
    threshold_max = 0
    precision_max = 0
    recall_max = 0

    # [N*C]  N proteins with C classes
    # prediction, ground_truth both [N*C]-tensors
    # rows: protein1, protein2, protein3, etc.; columns: func1, func2, func3, etc.

    threshold = 0.01
    while threshold <= 0.9:
        precision = 0
        recall = 0

        # get threshold dependant binary prediction
        # prediction = 1 where bigger than threshold
        prediction = (prediction_prob > threshold)
        prediction = prediction.float()

        true_positives = torch.sum(prediction * ground_truth, dim=1)    # sum over columns
        all_predicted_positives = torch.sum(prediction, dim=1)  # sum over columns
        all_real_positives = torch.sum(ground_truth, dim=1)  # sum over columns

        # m: number of sequences with at least one predicted score greater than or equal to threshold
        m = torch.nonzero(all_predicted_positives).size(dim=0)

        if m > 0:
            #replace YW
            # clamp(tensor, min_val, max_val): clamp all elements into range
            # prediction_over_threshold = torch.clamp(all_predicted_positives, 0, 1)
            #
            # true_positives_for_m = true_positives*prediction_over_threshold
            # non_zero_indices = torch.nonzero(true_positives_for_m)
            # if non_zero_indices.size(dim=0) != 0:
            #     non_zero_indices = torch.squeeze(non_zero_indices, 1)
            #     precision = torch.sum(true_positives_for_m[non_zero_indices] / all_predicted_positives[non_zero_indices]) / m
            #by
            non_zero_indices = torch.nonzero(all_predicted_positives)
            precision = torch.sum(true_positives[non_zero_indices] / all_predicted_positives[non_zero_indices]) / m
            #end(replaced)
        non_zero_indices_gt = torch.squeeze(torch.nonzero(all_real_positives), 1)
        n = non_zero_indices_gt.size(dim=0)
        if n > 0:
            recall = torch.sum(true_positives[non_zero_indices_gt] / all_real_positives[non_zero_indices_gt]) / n

        if precision != 0 or recall != 0:
            f = (2 * precision * recall) / (precision + recall)
            if f > f_max:
                f_max = f
                threshold_max = threshold
                precision_max = precision
                recall_max = recall

        threshold += 0.01

    return f_max, precision_max, recall_max, threshold_max

def evaluate(prediction_prob, ground_truth, idxMF, idxBP, idxCC):
    """
    Evaluate accuracy based on precision and ground truth separately for the three main ontologies
    :param prediction_prob: predictions for all proteins [N*C]
    :param ground_truth: ground truth encoded with multiple labels [N*C]
    :param idxMF: List of indices for terms in ground truth for Molecular Function
    :param idxBP: List of indices for terms in ground truth for Biological Process
    :param idxCC: List of indices for terms in ground truth for Cellular Component
    :return: dictionary containing evaluation results for all three ontologies
    """

    # Receiving seperate quality measures for the three main ontologies

    mf_f_max, mf_precision, mf_recall, mf_threshold = 0,0,0,0
    if len(idxMF) > 0:
        mf_f_max, mf_precision, mf_recall, mf_threshold = evaluate_pr_rc(prediction_prob[:,idxMF], ground_truth[:,idxMF])

    bp_f_max, bp_precision, bp_recall, bp_threshold = 0,0,0,0
    if len(idxBP) > 0:
        bp_f_max, bp_precision, bp_recall, bp_threshold = evaluate_pr_rc(prediction_prob[:,idxBP], ground_truth[:,idxBP])

    cc_f_max, cc_precision, cc_recall, cc_threshold = 0,0,0,0
    if len(idxCC) > 0:
        cc_f_max, cc_precision, cc_recall, cc_threshold = evaluate_pr_rc(prediction_prob[:,idxCC], ground_truth[:,idxCC])

    results = {}
    results['MF'] = {'f_max': mf_f_max, 'precision': mf_precision, 'recall': mf_recall, 'threshold': mf_threshold}
    results['BP'] = {'f_max': bp_f_max, 'precision': bp_precision, 'recall': bp_recall, 'threshold': bp_threshold}
    results['CC'] = {'f_max': cc_f_max, 'precision': cc_precision, 'recall': cc_recall, 'threshold': cc_threshold}
    return results
#added YW calculate threshold for each go term.
# def threshold_chosen(prediction_prob, ground_truth):
#     """
#         Evaluate accuracy based on precision and ground truth separately for the three main ontologies
#         :param prediction_prob: predictions for all proteins [N]
#         :param ground_truth: ground truth encoded with multiple labels [N]
#
#         :return: dictionary containing evaluation results for all three ontologies
#         """
#     threshold = 0.1
#     while threshold <= 0.9:
#         precision = 0
#         recall = 0
#         prediction = (prediction_prob > threshold)
#         prediction = prediction.float()
#
#         true_positives = torch.sum(prediction * ground_truth)  # sum over columns
#         all_predicted_positives = torch.sum(prediction)  # sum over columns
#         all_real_positives = torch.sum(ground_truth)  # sum over columns
#
#         # m: number of sequences with at least one predicted score greater than or equal to threshold
#
#         if all_predicted_positives > 0:
#             precision = true_positives / all_predicted_positives
#
#         if all_real_positives > 0:
#             recall = true_positives / all_real_positives
#
#         if precision != 0 or recall != 0:
#             f = (2 * precision * recall) / (precision + recall)
#             if f > f_max:
#                 f_max = f
#                 threshold_max = threshold
#                 precision_max = precision
#                 recall_max = recall
#
#         threshold += 0.05
#     return threshold_max, precision_max, recall_max, prediction
# #end(added)
