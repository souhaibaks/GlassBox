import csv
import sys
from pathlib import Path

sys.path.insert(0, '.')
from ironclaw.tools import autofit_tool, explain_tool

def build_titanic_numeric_csv(csv_path: Path) -> str:
    """Create a numeric-only CSV payload for autofit_tool."""
    selected_cols = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'survived']
    lines = [','.join(selected_cols)]

    with csv_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lines.append(','.join((row.get(c, '') or '').strip() for c in selected_cols))

    return '\n'.join(lines)


titanic_path = Path('titanic.csv')
if not titanic_path.exists():
    raise FileNotFoundError('titanic.csv was not found in the repository root.')

csv_data = build_titanic_numeric_csv(titanic_path)

# ---- Step 1: Agent triggers autofit_tool ----
print('Agent: calling autofit_tool...')
result = autofit_tool(csv_data, target_col='survived',
                      task_type='classification', cv=3)

if result.get('status') != 'ok':
    raise RuntimeError(f"autofit_tool failed: {result}")

report = result['report']
print(f'Best model  : {report["best_model"]}')
print(f'CV score    : {report["best_cv_score"]}')
print(f'Feature imp : {report["feature_importance"]}')

# ---- Step 2: Agent calls explain_tool ----
print()
print('Agent: generating explanation...')
explanation = explain_tool(report)
print()
print('=== Agent response to user ===')
print(explanation['summary'])
print()
print('Decision factors:')
for f in explanation['decision_factors']:
    print(f'  {f["rank"]}. {f["feature"]:12} {f["importance_pct"]:5.1f}%  ({f["method"]})')