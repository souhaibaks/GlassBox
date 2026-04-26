# GlassBox IronClaw Deployment

This directory exposes GlassBox-AutoML as an IronClaw-compatible MCP server.

## Local smoke test

Run the stdio transport:

```bash
echo "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}" | python ironclaw/mcp_server.py
```

Run the HTTP transport expected by IronClaw's MCP client:

```bash
python ironclaw/mcp_server.py --http --host 0.0.0.0 --port 8765
```

Health check:

```bash
curl http://127.0.0.1:8765/health
```

List tools over HTTP:

```bash
curl -s http://127.0.0.1:8765 \
  -H "Content-Type: application/json" \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}"
```

## Remote IronClaw host

Connect to the NEAR AI IronClaw machine:

```powershell
ssh -i "C:\Users\AMBEN\Desktop\agent-private-key.pem" -p 24114 agent@baremetal0.agents.near.ai
```

On the remote machine, copy or clone the GlassBox project, then install the only
runtime dependency:

```bash
cd GlassBox
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
python ironclaw/mcp_server.py --http --host 0.0.0.0 --port 8765
```

If `python3 -m venv .venv` fails because `ensurepip` is unavailable and you
cannot install `python3.11-venv`, use the Debian-managed Python workaround:

```bash
python3 -m pip install --user --break-system-packages numpy
```

In another shell on the same host, register the MCP endpoint with IronClaw:

```bash
ironclaw mcp add glassbox http://127.0.0.1:8765 \
  --description "GlassBox white-box AutoML tools"
ironclaw mcp test glassbox
```

The agent can then handle requests like:

```text
Build a model to predict "survived" using /home/agent/data/titanic.csv.
```

If the CSV is already pasted into the chat, the agent should call
`autofit_tool` with `csv_data`. If the user provides a path, it should call
`autofit_tool` with `csv_path`. The path must exist on the machine running the
MCP server. A Windows path such as `C:\Users\...\data.csv` only works when the
server is running on that Windows machine; on the NEAR AI host, copy or upload
the CSV there first and use the remote path.

Expected tool flow:

1. `autofit_tool` receives CSV text or a CSV path, target column, task type, CV folds, and optional `search_strategy` (`grid` or `random`).
2. GlassBox infers numeric, boolean, and text categorical columns, then encodes mixed CSV data into a numeric matrix.
3. GlassBox runs EDA, cleaning, model search, evaluation, and importance ranking.
4. `autofit_tool` returns a JSON `model_artifact` for later prediction.
5. `explain_tool` converts the JSON report into a user-facing explanation.

Prediction from the artifact:

```bash
curl -s http://127.0.0.1:8765 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"predict_tool","arguments":{"model_artifact":MODEL_ARTIFACT_HERE,"X_data":[{"pclass":"3","sex":"male","age":"24","fare":"7.25"}]}}}'
```

## Free chat-style simulator

If the hosted IronClaw chat UI is unavailable, run the same tool flow from a
natural-language prompt:

```bash
python ironclaw/chat_sim.py \
  "Build a model to predict \"survived\" using /home/agent/GlassBox/titanic.csv"
```

This simulator is deterministic and small. It parses the target column and CSV
path, calls `autofit_tool`, calls `explain_tool`, then prints the agent-style
answer.
