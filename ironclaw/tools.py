"""GlassBox IronClaw MCP Tools.

Implements tools callable by an IronClaw (NEAR AI) agent:
  - autofit_tool  : end-to-end AutoML from CSV data
  - eda_tool      : EDA report only
  - predict_tool  : run predictions with a serialized model state
  - explain_tool  : generate a user-facing explanation from an AutoFit report

All inputs/outputs are JSON-serializable Python dicts.
"""

import csv
import io
import json
import sys
import os
import warnings
from pathlib import Path

# Ensure project root on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


def _resolve_csv_data(csv_data: str = None, csv_path: str = None) -> tuple:
    #Return CSV text plus its source label from inline text or a local path
    if csv_data:
        return csv_data, 'inline'
    if not csv_path:
        raise ValueError('Provide either csv_data or csv_path.')

    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CSV path not found: {csv_path}")
    if not path.is_file():
        raise ValueError(f"CSV path is not a file: {csv_path}")
    return path.read_text(encoding='utf-8'), str(path)


def _parse_csv_rows(csv_data: str, has_header: bool = True) -> tuple:
    reader = csv.reader(io.StringIO(csv_data))
    raw_rows = [[cell.strip() for cell in row] for row in reader if row]
    if not raw_rows:
        return [], []
    if has_header:
        return raw_rows[0], raw_rows[1:]
    width = max(len(row) for row in raw_rows)
    return [str(i) for i in range(width)], raw_rows


def _target_index(header, target_col, has_header):
    if has_header and target_col in header:
        return header.index(target_col)
    try:
        idx = int(target_col)
    except (TypeError, ValueError):
        raise ValueError(f"Column '{target_col}' not found.")
    if idx < 0 or idx >= len(header):
        raise ValueError(f"Column index {idx} is out of range.")
    return idx


def _missing(value) -> bool:
    return value is None or str(value).strip().lower() in ('', 'nan', 'none', 'null', '?')


def _to_float(value):
    if _missing(value):
        return float('nan')
    return float(value)


def _mode(values, default=''):
    valid = [v for v in values if not _missing(v)]
    if not valid:
        return default
    counts = {}
    for value in valid:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def _infer_column(values):
    valid = [v for v in values if not _missing(v)]
    lower = {str(v).strip().lower() for v in valid}
    bool_values = {'true', 'false', 'yes', 'no', '0', '1'}
    if valid and lower.issubset(bool_values):
        return 'boolean'
    try:
        for value in valid:
            float(value)
        return 'numeric'
    except ValueError:
        return 'categorical'


def _bool_to_float(value):
    if _missing(value):
        return float('nan')
    lowered = str(value).strip().lower()
    if lowered in ('true', 'yes', '1'):
        return 1.0
    if lowered in ('false', 'no', '0'):
        return 0.0
    return float('nan')


def _build_preprocessed_dataset(header, rows, target_idx, task_type):
    feature_cols = [i for i in range(len(header)) if i != target_idx]
    columns = []
    feature_names = []
    matrix_rows = []
    y_raw = []

    column_values = {
        idx: [row[idx] if idx < len(row) else '' for row in rows]
        for idx in feature_cols
    }

    for idx in feature_cols:
        name = header[idx]
        values = column_values[idx]
        kind = _infer_column(values)
        if kind == 'categorical':
            fill = _mode(values, default='missing')
            categories = []
            seen = set()
            for value in values:
                key = fill if _missing(value) else value
                if key not in seen:
                    seen.add(key)
                    categories.append(key)
            out_names = [f'{name}={cat}' for cat in categories]
            columns.append({
                'index': idx,
                'name': name,
                'type': kind,
                'fill_value': fill,
                'categories': categories,
                'output_features': out_names,
            })
            feature_names.extend(out_names)
        else:
            out_name = name
            columns.append({
                'index': idx,
                'name': name,
                'type': kind,
                'output_features': [out_name],
            })
            feature_names.append(out_name)

    for row in rows:
        encoded = []
        for meta in columns:
            value = row[meta['index']] if meta['index'] < len(row) else ''
            if meta['type'] == 'numeric':
                try:
                    encoded.append(_to_float(value))
                except ValueError:
                    encoded.append(float('nan'))
            elif meta['type'] == 'boolean':
                encoded.append(_bool_to_float(value))
            else:
                value = meta['fill_value'] if _missing(value) else value
                encoded.extend([1.0 if value == cat else 0.0
                                for cat in meta['categories']])
        matrix_rows.append(encoded)
        y_raw.append(row[target_idx] if target_idx < len(row) else '')

    X = np.asarray(matrix_rows, dtype=float)
    if task_type == 'regression':
        y = np.array([float(value) for value in y_raw], dtype=float)
    else:
        y = np.asarray(y_raw)

    preprocessor = {
        'target_column': header[target_idx],
        'target_index': target_idx,
        'input_columns': header,
        'columns': columns,
        'feature_names': feature_names,
    }
    return X, y, preprocessor


def _transform_with_preprocessor(records, preprocessor):
    columns = preprocessor['columns']
    matrix = []
    for record in records:
        encoded = []
        for meta in columns:
            if isinstance(record, dict):
                value = record.get(meta['name'], '')
            else:
                value = record[meta['index']] if meta['index'] < len(record) else ''
            if meta['type'] == 'numeric':
                try:
                    encoded.append(_to_float(value))
                except ValueError:
                    encoded.append(float('nan'))
            elif meta['type'] == 'boolean':
                encoded.append(_bool_to_float(value))
            else:
                value = meta['fill_value'] if _missing(value) else value
                encoded.extend([1.0 if value == cat else 0.0
                                for cat in meta['categories']])
        matrix.append(encoded)
    return np.asarray(matrix, dtype=float)


def _artifact_model_name(best_model, task_type):
    mapping = {
        ('DecisionTree', 'classification'): 'DecisionTreeClassifier',
        ('DecisionTree', 'regression'): 'DecisionTreeRegressor',
        ('RandomForest', 'classification'): 'RandomForestClassifier',
        ('RandomForest', 'regression'): 'RandomForestRegressor',
        ('LogisticRegression', 'classification'): 'LogisticRegression',
        ('LinearRegression', 'regression'): 'LinearRegression',
        ('NaiveBayes', 'classification'): 'NaiveBayes',
        ('KNN', 'classification'): 'KNNClassifier',
        ('KNN', 'regression'): 'KNNRegressor',
    }
    return mapping[(best_model, task_type)]


# ---------------------------------------------------------------------------
# Tool: EDA
# ---------------------------------------------------------------------------

def eda_tool(csv_data: str = None, csv_path: str = None,
             has_header: bool = True) -> dict:
    #Perform Automated EDA on CSV data
    from transformers.EDA import EDAInspector

    try:
        csv_data, source = _resolve_csv_data(csv_data, csv_path)
        lines = [l.strip() for l in csv_data.strip().split('\n') if l.strip()]
        header = None
        if has_header and lines:
            header = [h.strip() for h in lines[0].split(',')]
            lines = lines[1:]

        rows = []
        for line in lines:
            parts = line.split(',')
            row = []
            for p in parts:
                p = p.strip()
                try:
                    row.append(float(p))
                except ValueError:
                    row.append(float('nan'))
            rows.append(row)

        if not rows:
            return {'error': 'No data rows found.'}

        X = np.array(rows, dtype=float)
        inspector = EDAInspector()
        inspector.fit(X)
        report = inspector.report()

        # Attach column names if header available
        if header:
            report['columns'] = header

        return {'status': 'ok', 'source': source, 'eda': report}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# ---------------------------------------------------------------------------
# Tool: AutoFit
# ---------------------------------------------------------------------------

def autofit_tool(csv_data: str = None, target_col: str = None,
                 csv_path: str = None,
                 task_type: str = 'classification',
                 has_header: bool = True,
                 cv: int = 5,
                 search_strategy: str = 'grid',
                 n_iter: int = 10) -> dict:

    #Run AutoML on CSV data targeting a specific column

    from autofit import AutoFit

    try:
        if target_col is None:
            return {'status': 'error', 'message': 'target_col is required.'}

        csv_data, source = _resolve_csv_data(csv_data, csv_path)
        header, rows = _parse_csv_rows(csv_data, has_header=has_header)
        if not header:
            return {'status': 'error', 'message': 'Empty CSV data.'}
        if not rows:
            return {'status': 'error', 'message': 'No data rows.'}

        try:
            target_idx = _target_index(header, target_col, has_header)
            X, y, preprocessor = _build_preprocessed_dataset(
                header, rows, target_idx, task_type)
        except ValueError as err:
            return {'status': 'error', 'message': str(err)}

        af = AutoFit(task_type=task_type, cv=cv, random_state=42,
                     search_strategy=search_strategy, n_iter=n_iter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            af.fit(X, y)
        report = af.report()

        report['feature_names'] = preprocessor['feature_names']
        if report.get('feature_importance'):
            for item in report['feature_importance']:
                idx = item['feature_index']
                names = preprocessor['feature_names']
                item['feature_name'] = names[idx] if idx < len(names) else str(idx)

        report['source'] = source
        report['preprocessing'] = preprocessor
        report['model_artifact'] = {
            'version': 1,
            'task_type': task_type,
            'model_name': _artifact_model_name(report['best_model'], task_type),
            'model_params': report['best_params'],
            'preprocessing': preprocessor,
            'X_train': X.tolist(),
            'y_train': y.tolist(),
        }

        return {'status': 'ok', 'report': report}
    except Exception as e:
        import traceback
        return {'status': 'error', 'message': str(e),
                'traceback': traceback.format_exc()}


# ---------------------------------------------------------------------------
# Tool: Predict (stateless — accepts serialized model config)
# ---------------------------------------------------------------------------

def predict_tool(model_name: str = None, model_params: dict = None,
                 X_data: list = None, y_train: list = None,
                 X_train: list = None,
                 model_artifact: dict = None,
                 task_type: str = 'classification') -> dict:
    
    #Run predictions using a named GlassBox model
    
    try:
        MODEL_MAP = _get_model_map()
        model_params = model_params or {}

        if model_artifact is not None:
            model_name = model_artifact.get('model_name')
            model_params = model_artifact.get('model_params', {})
            X_train = model_artifact.get('X_train')
            y_train = model_artifact.get('y_train')
            task_type = model_artifact.get('task_type', task_type)
            preprocessor = model_artifact.get('preprocessing')
            if preprocessor is not None:
                X_pred = _transform_with_preprocessor(X_data or [], preprocessor)
            else:
                X_pred = np.array(X_data or [], dtype=float)
        else:
            X_pred = np.array(X_data or [], dtype=float)

        if model_name is None:
            return {'status': 'error', 'message': 'model_name or model_artifact is required.'}
        if model_name not in MODEL_MAP:
            return {'status': 'error',
                    'message': f"Unknown model '{model_name}'. "
                               f"Choose from: {list(MODEL_MAP.keys())}"}

        if X_train is None or y_train is None:
            return {'status': 'error',
                    'message': 'X_train and y_train are required unless model_artifact includes them.'}

        X_tr = np.array(X_train, dtype=float)
        if task_type == 'regression':
            y_tr = np.array(y_train, dtype=float)
        else:
            y_tr = np.array(y_train)

        model_cls = MODEL_MAP[model_name]
        model = model_cls(**model_params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_pred)

        return {
            'status': 'ok',
            'model': model_name,
            'predictions': preds.tolist(),
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# ---------------------------------------------------------------------------
# Tool: Explain — generate natural language explanation from an AutoFit report
# ---------------------------------------------------------------------------

def explain_tool(report: dict, feature_names: list = None) -> dict:
    """Generate a natural language explanation from an AutoFit report

    This is the core explainability feature: because GlassBox models are
    built from scratch, every decision can be traced and explained in plain
    language that an IronClaw agent can relay to the user"""

    try:
        if not isinstance(report, dict):
            return {'status': 'error', 'message': 'report must be a dict.'}

        names = (feature_names
                 or report.get('feature_names')
                 or [f'Feature {i}' for i in range(report.get('n_features', 0))])

        best_model = report.get('best_model', 'Unknown')
        best_params = report.get('best_params', {})
        cv_score = report.get('best_cv_score', 0)
        cv_folds = report.get('cv', 5)
        task = report.get('task_type', 'classification')
        n_samples = report.get('n_samples', '?')
        n_features = report.get('n_features', len(names))
        eval_metrics = report.get('evaluation', {})
        all_results = report.get('all_results', [])
        fi = report.get('feature_importance')
        eda = report.get('eda', {})

        # ── Performance summary ──────────────────────────────────────────
        metric_name = 'accuracy' if task == 'classification' else 'R2 score'
        metric_val = (eval_metrics.get('accuracy')
                      if task == 'classification'
                      else eval_metrics.get('r2'))
        metric_str = f"{metric_val:.2%}" if metric_val is not None else 'N/A'

        # ── Runner-up ────────────────────────────────────────────────────
        others = sorted(
            [r for r in all_results if r['model'] != best_model],
            key=lambda r: r['cv_score'], reverse=True
        )
        runner_up = f"{others[0]['model']} ({others[0]['cv_score']:.2%})" if others else None

        # ── Feature explanation ──────────────────────────────────────────
        method_labels = {
            'split_frequency': 'split usage in decision tree nodes',
            'coefficient_magnitude': 'regression coefficient magnitude',
            'weight_magnitude': 'logistic regression weight magnitude',
            'class_mean_variance': 'discriminative variance between class means',
        }
        fi_lines = []
        decision_factors = []
        if fi:
            method = fi[0].get('method', '')
            method_label = method_labels.get(method, 'feature importance')
            for rank, item in enumerate(fi[:5], 1):
                idx = item['feature_index']
                fname = names[idx] if idx < len(names) else f'Feature {idx}'
                pct = item['importance'] * 100
                fi_lines.append(f"  {rank}. '{fname}' ({pct:.1f}%, {method_label})")
                decision_factors.append({
                    'rank': rank,
                    'feature': fname,
                    'feature_index': idx,
                    'importance_pct': round(pct, 2),
                    'method': method,
                })

        # ── EDA highlights ───────────────────────────────────────────────
        eda_notes = []
        stats = eda.get('statistics', {})
        for col_key, col_stats in stats.items():
            missing = col_stats.get('n_missing', 0)
            if missing > 0:
                col_name = names[int(col_key)] if int(col_key) < len(names) else col_key
                eda_notes.append(f"'{col_name}' had {missing} missing value(s) (imputed with mean)")

        # ── Model params note ────────────────────────────────────────────
        params_str = ', '.join(f'{k}={v}' for k, v in best_params.items()) if best_params else 'default'

        # ── Build summary paragraph ──────────────────────────────────────
        lines = [
            f"GlassBox AutoFit analysed {n_samples} samples with {n_features} features "
            f"for a {task} task.",
        ]
        if eda_notes:
            lines.append("Data quality: " + "; ".join(eda_notes) + ".")
        lines.append(
            f"After {cv_folds}-fold cross-validation over {len(all_results)} candidate model(s), "
            f"the best model was {best_model} (CV score: {cv_score:.2%}, params: {params_str})."
        )
        if runner_up:
            lines.append(f"The runner-up was {runner_up}.")
        if metric_val is not None:
            lines.append(
                f"On the full training set the model achieved {metric_name} = {metric_str}."
            )
        if fi_lines:
            fi_method = method_labels.get(fi[0].get('method', ''), 'importance')
            top_name = names[fi[0]['feature_index']] if fi[0]['feature_index'] < len(names) else 'Feature 0'
            lines.append(
                f"The most influential feature was '{top_name}' "
                f"({fi[0]['importance'] * 100:.1f}% of {fi_method})."
            )
            lines.append("Top features by importance:")
            lines.extend(fi_lines)
        else:
            lines.append(
                "Feature importance is not available for this model type "
                "(KNN uses distance-based voting with no inherent feature ranking)."
            )

        summary = " ".join(lines[:4]) + "\n" + "\n".join(lines[4:])

        return {
            'status': 'ok',
            'summary': summary,
            'best_model': best_model,
            'best_params': best_params,
            'cv_score': cv_score,
            'full_metric': {metric_name: metric_val},
            'decision_factors': decision_factors,
            'data_quality_notes': eda_notes,
        }

    except Exception as e:
        import traceback
        return {'status': 'error', 'message': str(e),
                'traceback': traceback.format_exc()}


def _get_model_map():
    from models import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        LinearRegression,
        LogisticRegression,
        NBClassifier,
        KNNClassifier,
        KNNRegressor,
    )

    return {
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'RandomForestClassifier': RandomForestClassifier,
        'RandomForestRegressor': RandomForestRegressor,
        'LogisticRegression': LogisticRegression,
        'LinearRegression': LinearRegression,
        'NaiveBayes': NBClassifier,
        'KNNClassifier': KNNClassifier,
        'KNNRegressor': KNNRegressor,
    }
