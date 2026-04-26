import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""Test: IronClaw MCP tools JSON roundtrip."""
import sys
sys.path.insert(0, '.')
import json
import os
import tempfile
import threading
import urllib.request


def test_eda_tool():
    from ironclaw.tools import eda_tool

    csv = """sepal_length,sepal_width,petal_length
5.1,3.5,1.4
4.9,3.0,1.4
6.7,3.1,4.4
5.8,2.7,5.1
7.1,3.0,5.9
"""
    result = eda_tool(csv, has_header=True)
    assert result['status'] == 'ok', result
    eda = result['eda']
    assert 'statistics' in eda
    assert 'correlations' in eda
    assert 'columns' in eda
    assert eda['columns'] == ['sepal_length', 'sepal_width', 'petal_length']
    print(f"[OK] eda_tool passed: columns={eda['columns']}")


def test_autofit_tool():
    from ironclaw.tools import autofit_tool, explain_tool

    csv = """x1,x2,x3,label
0.1,0.2,0.3,0
0.2,0.1,0.4,0
0.9,0.8,0.7,1
0.8,0.9,0.6,1
0.1,0.3,0.2,0
0.9,0.7,0.8,1
0.2,0.2,0.3,0
0.8,0.8,0.9,1
0.05,0.1,0.15,0
0.95,0.85,0.75,1
0.15,0.25,0.35,0
0.75,0.65,0.85,1
"""
    result = autofit_tool(csv, target_col='label',
                          task_type='classification', cv=3)
    assert result['status'] == 'ok', result
    report = result['report']
    assert report['best_model'] is not None
    assert report['cv'] == 3
    assert 'feature_names' in result['report']
    assert result['report']['feature_names'] == ['x1', 'x2', 'x3']
    explanation = explain_tool(report)
    assert "After 3-fold cross-validation" in explanation["summary"]
    print(f"[OK] autofit_tool passed: best={report['best_model']}, "
          f"score={report['best_cv_score']}")


def test_autofit_tool_with_csv_path():
    from ironclaw.tools import autofit_tool

    csv = """x1,x2,label
0.1,0.2,0
0.9,0.8,1
0.2,0.1,0
0.8,0.9,1
0.15,0.25,0
0.85,0.75,1
"""
    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False,
                                    encoding='utf-8') as f:
        f.write(csv)
        csv_path = f.name

    result = autofit_tool(csv_path=csv_path, target_col='label',
                          task_type='classification', cv=3)
    try:
        assert result['status'] == 'ok', result
        assert result['report']['source'] == csv_path
        assert result['report']['feature_names'] == ['x1', 'x2']
        print(f"[OK] autofit_tool csv_path passed: source={csv_path}")
    finally:
        os.unlink(csv_path)


def test_autofit_tool_mixed_csv_and_artifact_predict():
    from ironclaw.tools import autofit_tool, predict_tool

    csv = """age,sex,embarked,fare,survived
25,male,S,10.0,0
40,female,C,80.0,1
21,male,S,8.0,0
55,female,C,90.0,1
31,male,Q,15.0,0
48,female,C,85.0,1
"""
    result = autofit_tool(csv_data=csv, target_col='survived',
                          task_type='classification', cv=3)
    assert result['status'] == 'ok', result
    report = result['report']
    feature_names = report['feature_names']
    assert 'sex=male' in feature_names
    assert 'sex=female' in feature_names
    assert 'embarked=S' in feature_names
    assert 'model_artifact' in report

    pred = predict_tool(
        model_artifact=report['model_artifact'],
        X_data=[
            {'age': '24', 'sex': 'male', 'embarked': 'S', 'fare': '12.0'},
            {'age': '52', 'sex': 'female', 'embarked': 'C', 'fare': '88.0'},
        ],
    )
    assert pred['status'] == 'ok', pred
    assert len(pred['predictions']) == 2
    print(f"[OK] mixed CSV + artifact predict passed: predictions={pred['predictions']}")


def test_autofit_tool_random_search():
    from ironclaw.tools import autofit_tool

    csv = """x1,x2,label
0.1,0.2,0
0.2,0.1,0
0.9,0.8,1
0.8,0.9,1
0.15,0.25,0
0.85,0.75,1
"""
    result = autofit_tool(csv_data=csv, target_col='label',
                          task_type='classification', cv=3,
                          search_strategy='random', n_iter=2)
    assert result['status'] == 'ok', result
    assert result['report']['search_strategy'] == 'random'
    assert all(r['search_strategy'] == 'random'
               for r in result['report']['all_results'])
    print(f"[OK] autofit_tool random search passed: best={result['report']['best_model']}")


def test_chat_sim_parser_and_run():
    from ironclaw.chat_sim import parse_request, run_chat_sim

    csv = """x1,x2,label
0.1,0.2,0
0.9,0.8,1
0.2,0.1,0
0.8,0.9,1
0.15,0.25,0
0.85,0.75,1
"""
    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False,
                                    encoding='utf-8') as f:
        f.write(csv)
        csv_path = f.name

    try:
        message = f'Build a model to predict "label" using "{csv_path}"'
        parsed = parse_request(message)
        assert parsed['target_col'] == 'label'
        assert parsed['csv_path'] == csv_path
        assert parsed['task_type'] == 'classification'

        response = run_chat_sim(message)
        assert response['status'] == 'ok', response
        assert response['report']['best_model'] is not None
        assert 'GlassBox AutoFit analysed' in response['explanation']['summary']
        print(f"[OK] chat_sim passed: best={response['report']['best_model']}")
    finally:
        os.unlink(csv_path)


def test_predict_tool():
    from ironclaw.tools import predict_tool
    import numpy as np

    X_train = [[0.1, 0.2], [0.9, 0.8], [0.2, 0.1], [0.8, 0.9],
               [0.15, 0.25], [0.85, 0.75]]
    y_train = [0, 1, 0, 1, 0, 1]
    X_pred = [[0.1, 0.2], [0.9, 0.8]]

    result = predict_tool(
        model_name='DecisionTreeClassifier',
        model_params={'max_depth': 3},
        X_data=X_pred,
        X_train=X_train,
        y_train=y_train,
        task_type='classification',
    )
    assert result['status'] == 'ok', result
    assert len(result['predictions']) == 2
    print(f"[OK] predict_tool passed: predictions={result['predictions']}")


def test_mcp_server_list():
    from ironclaw.mcp_server import handle_request

    req = json.dumps({"jsonrpc": "2.0", "id": 1,
                      "method": "tools/list", "params": {}})
    resp = json.loads(handle_request(req))
    assert resp['id'] == 1
    assert 'result' in resp
    tools = resp['result']['tools']
    names = [t['name'] for t in tools]
    assert 'eda_tool' in names
    assert 'autofit_tool' in names
    assert 'predict_tool' in names
    print(f"[OK] MCP tools/list: {names}")


def test_mcp_server_call():
    from ironclaw.mcp_server import handle_request

    csv = "a,b,label\n0.1,0.2,0\n0.9,0.8,1\n0.2,0.1,0\n0.8,0.9,1"
    req = json.dumps({
        "jsonrpc": "2.0", "id": 2,
        "method": "tools/call",
        "params": {
            "name": "eda_tool",
            "arguments": {"csv_data": csv, "has_header": True}
        }
    })
    resp = json.loads(handle_request(req))
    assert resp['id'] == 2
    content = resp['result']['content']
    assert content[0]['type'] == 'text'
    data = json.loads(content[0]['text'])
    assert data['status'] == 'ok'
    print(f"[OK] MCP tools/call eda_tool via server: columns={data['eda']['columns']}")


def test_mcp_http_transport():
    from http.server import ThreadingHTTPServer
    from ironclaw.mcp_server import McpHttpHandler

    server = ThreadingHTTPServer(('127.0.0.1', 0), McpHttpHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/list",
            "params": {}
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://{host}:{port}",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        assert data["id"] == 3
        names = [t["name"] for t in data["result"]["tools"]]
        assert "autofit_tool" in names
        print(f"[OK] MCP HTTP transport: {names}")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    test_eda_tool()
    test_autofit_tool()
    test_autofit_tool_with_csv_path()
    test_autofit_tool_mixed_csv_and_artifact_predict()
    test_autofit_tool_random_search()
    test_chat_sim_parser_and_run()
    test_predict_tool()
    test_mcp_server_list()
    test_mcp_server_call()
    test_mcp_http_transport()
    print("\nAll IronClaw MCP tests passed!")
