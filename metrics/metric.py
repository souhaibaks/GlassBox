import numpy as np


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Metric:

    def score(self, y_true, y_pred):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

class Accuracy(Metric):

    def score(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float(np.mean(y_true == y_pred))


class ConfusionMatrix(Metric):
    """Confusion matrix (binary or multi-class)."""

    def score(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n = len(classes)
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[cls_to_idx[t], cls_to_idx[p]] += 1
        return cm


class Precision(Metric):
    """Precision score with optional averaging."""

    def __init__(self, average='binary', pos_label=1):
        self.average = average
        self.pos_label = pos_label

    def score(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))

        if self.average == 'binary':
            c = self.pos_label
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        precisions, weights = [], []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            weights.append(np.sum(y_true == c))

        if self.average == 'macro':
            return float(np.mean(precisions))
        else:  # weighted
            return float(np.average(precisions, weights=weights))


class Recall(Metric):
    """Recall score with optional averaging."""

    def __init__(self, average='binary', pos_label=1):
        self.average = average
        self.pos_label = pos_label

    def score(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))

        if self.average == 'binary':
            c = self.pos_label
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        recalls, weights = [], []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            weights.append(np.sum(y_true == c))

        if self.average == 'macro':
            return float(np.mean(recalls))
        else:
            return float(np.average(recalls, weights=weights))


class F1Score(Metric):
    """F1 score, computed from precision and recall."""

    def __init__(self, average='binary', pos_label=1):
        self.average = average
        self.pos_label = pos_label

    def score(self, y_true, y_pred):
        p = Precision(average=self.average, pos_label=self.pos_label).score(y_true, y_pred)
        r = Recall(average=self.average, pos_label=self.pos_label).score(y_true, y_pred)
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def classification_report(y_true, y_pred):
    """Return per-class metrics plus accuracy, macro avg, and weighted avg."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    report = {}

    for c in classes:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        support = int(np.sum(y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        report[str(c)] = {
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1-score': round(f1, 4),
            'support': support,
        }

    n = len(y_true)
    class_dicts = [v for v in report.values() if isinstance(v, dict)]
    report['accuracy'] = round(float(np.mean(y_true == y_pred)), 4)
    report['macro avg'] = {
        'precision': round(float(np.mean([v['precision'] for v in class_dicts])), 4),
        'recall': round(float(np.mean([v['recall'] for v in class_dicts])), 4),
        'f1-score': round(float(np.mean([v['f1-score'] for v in class_dicts])), 4),
        'support': n,
    }
    weights = [report[str(c)]['support'] for c in classes]
    report['weighted avg'] = {
        'precision': round(float(np.average([report[str(c)]['precision']
                                             for c in classes], weights=weights)), 4),
        'recall': round(float(np.average([report[str(c)]['recall']
                                          for c in classes], weights=weights)), 4),
        'f1-score': round(float(np.average([report[str(c)]['f1-score']
                                            for c in classes], weights=weights)), 4),
        'support': n,
    }
    return report


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

class MeanAbsoluteError(Metric):
    """Mean Absolute Error (MAE)."""

    def score(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        return float(np.mean(np.abs(y_true - y_pred)))


class MeanSquaredError(Metric):
    """Mean Squared Error (MSE)."""

    def score(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        return float(np.mean((y_true - y_pred) ** 2))


class R2Score(Metric):
    """Coefficient of determination (R²)."""

    def score(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0
