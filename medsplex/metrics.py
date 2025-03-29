from sklearn.metrics import f1_score, precision_score, recall_score
from utils import labels_idx_mapping


# Evaluation function (exact match)
def validate_answer(example, pred, trace=None):
    """
    Exact match score.

    :param example: dspy.Example
    :param pred: dspy.Prediction
    :param trace: bool
    :return: bool (True if true label and prediction label are the same)
    """
    y_true = example.valence.lower()
    y_pred = pred.valence.lower()
    return y_true == y_pred


def F1(example, pred, trace=None):
    """
    Cross-example f1 score for dspy.Evaluate. Use with class F1SPLangProgram

    :param pred: dspy.Prediction(valences=List[str])
    :param example: dspy.Example(valences=List())
    :return: float
    """
    y_pred = [labels_idx_mapping[out] for out in pred.valences]
    y_true = [labels_idx_mapping[lab] for lab in example.valences]
    return f1_score(y_true=y_true, y_pred=y_pred, average='macro')


def P(example, pred):
    """
    Cross-example precision score for dspy.Evaluate. Use with class F1SPLangProgram

    :param pred: dspy.Prediction(valences=List[str])
    :param example: dspy.Example(valences=List())
    :return: float
    """

    y_pred = [labels_idx_mapping[out] for out in pred.valences]
    y_true = [labels_idx_mapping[lab] for lab in example.valences]
    return precision_score(y_true=y_true, y_pred=y_pred, average='macro')


def R(example, pred):
    """
    Cross-example recall score for dspy.Evaluate. Use with class F1SPLangProgram

    :param pred: dspy.Prediction(valences=List[str])
    :param example: dspy.Example(valences=List())
    :return: float
    """

    y_pred = [labels_idx_mapping[out] for out in pred.valences]
    y_true = [labels_idx_mapping[lab] for lab in example.valences]
    return recall_score(y_true=y_true, y_pred=y_pred, average='macro')
