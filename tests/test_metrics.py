import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
from metrics.metric import ConfusionMatrix, R2Score, Precision, Recall, F1Score

def test_confusion_matrix():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    cm = ConfusionMatrix()
    matrix = cm.score(y_true, y_pred)

    expected = np.array([[2, 0], [1, 2]])  # TN, FP, FN, TP
    assert np.array_equal(matrix, expected)
    print("Confusion Matrix test passed!")

def test_r2_score():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])  # perfect prediction

    r2 = R2Score()
    score = r2.score(y_true, y_pred)
    assert score == 1.0

    y_pred_bad = np.array([1, 1, 1, 1, 1])  # bad prediction
    score_bad = r2.score(y_true, y_pred_bad)
    assert score_bad < 1.0
    print("R2 Score test passed!")

def test_precision_recall_f1_edge_cases():
    # All true positives
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])

    p = Precision().score(y_true, y_pred)
    r = Recall().score(y_true, y_pred)
    f1 = F1Score().score(y_true, y_pred)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0

    # No positives in true
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])

    p = Precision().score(y_true, y_pred)
    r = Recall().score(y_true, y_pred)
    f1 = F1Score().score(y_true, y_pred)
    assert p == 0.0  # no positives predicted
    assert r == 0.0  # no positives to recall
    assert f1 == 0.0

    print("Precision/Recall/F1 edge cases test passed!")

if __name__ == "__main__":
    test_confusion_matrix()
    test_r2_score()
    test_precision_recall_f1_edge_cases()