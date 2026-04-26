"""GlassBox IronClaw MCP Server — stdio JSON-RPC 2.0 server.

Runs inside the IronClaw (NEAR AI) secure sandbox. Messages are read from
stdin and written to stdout as newline-delimited JSON.

Supports:
  - tools/list  : return all available tool definitions + schemas
  - tools/call  : invoke a named tool with JSON arguments

Usage (WASM / CLI):
    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python ironclaw/mcp_server.py
"""

import argparse
import json
import sys
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ironclaw.tools import eda_tool, autofit_tool, predict_tool, explain_tool


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "eda_tool",
        "description": (
            "Perform Automated Exploratory Data Analysis (EDA) on CSV data. "
            "Returns statistics, correlations, outlier bounds, and column types."
        ),
        "annotations": {
            "destructive_hint": False,
            "side_effects_hint": False,
            "read_only_hint": True,
            "execution_time_hint": "medium"
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "csv_data": {
                    "type": "string",
                    "description": "Raw CSV text (rows separated by newlines, columns by commas)."
                },
                "csv_path": {
                    "type": "string",
                    "description": "Path to a CSV file on the machine running the MCP server."
                },
                "has_header": {
                    "type": "boolean",
                    "description": "Whether the first row is a header.",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "autofit_tool",
        "description": (
            "Run an end-to-end AutoML pipeline on CSV data. Automatically performs EDA, "
            "preprocessing, model search with K-Fold cross-validation, and returns a full "
            "explainability report including the best model, parameters, and feature importance."
        ),
        "annotations": {
            "destructive_hint": False,
            "side_effects_hint": False,
            "read_only_hint": True,
            "execution_time_hint": "slow"
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "csv_data": {
                    "type": "string",
                    "description": "Raw CSV text. Provide either csv_data or csv_path."
                },
                "csv_path": {
                    "type": "string",
                    "description": "Path to a CSV file on the machine running the MCP server. Provide either csv_data or csv_path."
                },
                "target_col": {
                    "type": "string",
                    "description": "Name (or 0-based index) of the target column."
                },
                "task_type": {
                    "type": "string",
                    "enum": ["classification", "regression"],
                    "default": "classification",
                    "description": "Type of ML task."
                },
                "has_header": {
                    "type": "boolean",
                    "default": True
                },
                "cv": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of K-Fold cross-validation folds."
                },
                "search_strategy": {
                    "type": "string",
                    "enum": ["grid", "random"],
                    "default": "grid",
                    "description": "Hyperparameter search strategy."
                },
                "n_iter": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of random configurations when search_strategy='random'."
                }
            },
            "required": ["target_col"]
        }
    },
    {
        "name": "explain_tool",
        "description": (
            "Generate a natural language explanation from an AutoFit report. "
            "Because GlassBox models are built from scratch, every decision can be "
            "traced: feature importance (split frequency for trees, coefficient "
            "magnitude for linear models, class-mean variance for Naive Bayes), "
            "data quality notes, and a ranked list of decision factors are all "
            "returned in plain language the agent can relay directly to the user."
        ),
        "annotations": {
            "destructive_hint": False,
            "side_effects_hint": False,
            "read_only_hint": True,
            "execution_time_hint": "fast"
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "report": {
                    "type": "object",
                    "description": "The 'report' dict from an autofit_tool response."
                },
                "feature_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of feature names (overrides names in report).",
                    "default": None
                }
            },
            "required": ["report"]
        }
    },
    {
        "name": "predict_tool",
        "description": (
            "Run predictions with either a model_artifact returned by autofit_tool "
            "or explicit training data plus a model name."
        ),
        "annotations": {
            "destructive_hint": False,
            "side_effects_hint": False,
            "read_only_hint": True,
            "execution_time_hint": "medium"
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "enum": [
                        "DecisionTreeClassifier", "DecisionTreeRegressor",
                        "RandomForestClassifier", "RandomForestRegressor",
                        "LogisticRegression", "LinearRegression",
                        "NaiveBayes", "KNNClassifier", "KNNRegressor"
                    ],
                    "description": "The GlassBox model to use."
                },
                "model_params": {
                    "type": "object",
                    "description": "Hyperparameters for the model.",
                    "default": {}
                },
                "model_artifact": {
                    "type": "object",
                    "description": "JSON artifact returned in autofit_tool report['model_artifact']."
                },
                "X_train": {
                    "type": "array",
                    "description": "Training feature matrix (list of lists)."
                },
                "y_train": {
                    "type": "array",
                    "description": "Training labels."
                },
                "X_data": {
                    "type": "array",
                    "description": "Prediction rows. With model_artifact, may be list of dicts or raw rows."
                },
                "task_type": {
                    "type": "string",
                    "enum": ["classification", "regression"],
                    "default": "classification"
                }
            },
            "required": ["X_data"]
        }
    }
]

TOOL_DISPATCH = {
    'eda_tool': eda_tool,
    'autofit_tool': autofit_tool,
    'predict_tool': predict_tool,
    'explain_tool': explain_tool,
}


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def _ok(request_id, result):
    return json.dumps({
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result
    })


def _error(request_id, code, message):
    return json.dumps({
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message}
    })


def handle_request(raw: str) -> str:
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        return _error(None, -32700, f"Parse error: {e}")

    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params", {})

    if method == "tools/list":
        return _ok(req_id, {"tools": TOOL_SCHEMAS})

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if tool_name not in TOOL_DISPATCH:
            return _error(req_id, -32601,
                          f"Tool '{tool_name}' not found. "
                          f"Available: {list(TOOL_DISPATCH.keys())}")
        try:
            result = TOOL_DISPATCH[tool_name](**arguments)
            # MCP result format: content array with text
            return _ok(req_id, {
                "content": [{"type": "text", "text": json.dumps(result)}]
            })
        except Exception as e:
            return _error(req_id, -32603,
                          f"Tool execution error: {e}")

    elif method == "initialize":
        return _ok(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "GlassBox-AutoML",
                "version": "1.0.0",
                "description": "White-box AutoML for IronClaw agents."
            }
        })

    else:
        return _error(req_id, -32601, f"Method not found: '{method}'")


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------

class McpHttpHandler(BaseHTTPRequestHandler):
    #Minimal HTTP transport for IronClaw's MCP JSON-RPC client

    server_version = "GlassBoxMCP/1.0"

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._write_json(200, {
                "status": "ok",
                "name": "GlassBox-AutoML",
                "transport": "http-json-rpc",
            })
            return
        self._write_json(404, {"status": "error", "message": "Not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        response = handle_request(raw)
        self._write_text(200, response)

    def log_message(self, fmt, *args):
        print(fmt % args, file=sys.stderr)

    def _write_json(self, status, payload):
        self._write_text(status, json.dumps(payload))

    def _write_text(self, status, text):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_http_server(host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), McpHttpHandler)
    print(f"GlassBox MCP HTTP server listening on http://{host}:{port}", file=sys.stderr)
    server.serve_forever()


# ---------------------------------------------------------------------------
# Main loops
# ---------------------------------------------------------------------------

def run_stdio_server():
    print("SERVER STARTED", file=sys.stderr)
    #Read newline-delimited JSON from stdin, respond to stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = handle_request(line)
        sys.stdout.write(response + '\n')
        sys.stdout.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description="GlassBox-AutoML MCP server")
    parser.add_argument("--http", action="store_true",
                        help="Serve MCP over HTTP POST instead of stdio.")
    parser.add_argument("--host", default=os.environ.get("GLASSBOX_MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("GLASSBOX_MCP_PORT", "8765")))
    args = parser.parse_args(argv)

    if args.http:
        run_http_server(args.host, args.port)
    else:
        run_stdio_server()


def main_http():
    main(["--http"])


if __name__ == '__main__':
    main()
