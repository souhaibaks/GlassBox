"""
GlassBox-AutoML — main.py demo
Demonstrates end-to-end AutoFit on Iris dataset (numpy only, no sklearn).
"""

import sys
sys.path.insert(0, '.')

import numpy as np

# Synthetic Iris-like data (no sklearn needed)
np.random.seed(42)

# Class 0: setosa-like
X0 = np.random.randn(50, 4) * np.array([0.4, 0.4, 0.2, 0.1]) + np.array([5.0, 3.4, 1.5, 0.3])
# Class 1: versicolor-like
X1 = np.random.randn(50, 4) * np.array([0.5, 0.3, 0.5, 0.2]) + np.array([5.9, 2.8, 4.2, 1.3])
# Class 2: virginica-like
X2 = np.random.randn(50, 4) * np.array([0.6, 0.3, 0.6, 0.3]) + np.array([6.6, 3.0, 5.5, 2.0])

X = np.vstack([X0, X1, X2])
y = np.array([0] * 50 + [1] * 50 + [2] * 50)

print("=" * 60)
print("GlassBox-AutoML - End-to-End Demo")
print("=" * 60)
print(f"Dataset: 150 samples, 4 features, 3 classes")
print()

# AutoFit
from autofit import AutoFit

af = AutoFit(task_type='classification', cv=5, random_state=42)
af.fit(X, y)
report = af.report()

print(f"Best Model   : {report['best_model']}")
print(f"Best Params  : {report['best_params']}")
print(f"Best CV Score: {report['best_cv_score']:.4f}")
print(f"Elapsed      : {report['elapsed_seconds']}s")
print()
print("Feature Importance:")
if report.get('feature_importance'):
    for item in report['feature_importance']:
        print(f"  Feature {item['feature_index']:>2}: {item['importance']:.4f}")
print()
print("All Candidate Results:")
for r in report['all_results']:
    print(f"  {r['model']:<22}: cv={r['cv_score']:.4f}  params={r['best_params']}")
print()
print("Evaluation Report (full training set):")
for key, val in report['evaluation'].items():
    if isinstance(val, dict):
        print(f"  {key}: {val}")
    elif key in ('accuracy', 'macro avg', 'weighted avg'):
        print(f"  {key}: {val}")
