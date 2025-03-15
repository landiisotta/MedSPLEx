from scipy.constants import precision
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import labels_idx_mapping


# Evaluation function (exact match)
def validate_answer(example, pred, trace=None):
    y_true = example.valence.lower()
    y_pred = pred.valence.lower()
    return y_true == y_pred


def F1(example, pred):
    y_pred = [labels_idx_mapping[out] for out in pred.valences]
    y_true = [labels_idx_mapping[lab] for lab in example.valences]
    return f1_score(y_true=y_true, y_pred=y_pred, average='macro')


def P(example, pred):
    y_pred = [labels_idx_mapping[out] for out in pred.valences]
    y_true = [labels_idx_mapping[lab] for lab in example.valences]
    return precision_score(y_true=y_true, y_pred=y_pred, average='macro')


def R(example, pred):
    y_pred = [labels_idx_mapping[out] for out in pred.valences]
    y_true = [labels_idx_mapping[lab] for lab in example.valences]
    return recall_score(y_true=y_true, y_pred=y_pred, average='macro')
