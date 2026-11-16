"""Minimal MCP server: newline-delimited JSON-RPC 2.0 over stdio."""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from .. import __version__
from .tools import call_tool, list_tools


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    request_id = message.get("id")
    if request_id is None:  # notification (e.g. notifications/initialized)
        return None
    method = message.get("method")
    params = message.get("params") or {}
    try:
        if method == "initialize":
            result = {
                "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "scirag", "version": __version__},
            }
        elif method == "tools/list":
            result = {"tools": list_tools()}
        elif method == "tools/call":
            result = call_tool(str(params.get("name", "")), params.get("arguments") or {})
        elif method == "ping":
            result = {}
        else:
            return _error(request_id, -32601, f"method not found: {method}")
    except Exception as exc:  # pragma: no cover - defensive
        return _error(request_id, -32603, str(exc).splitlines()[0][:300])
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def run_stdio(stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            response = handle(message) if isinstance(message, dict) else _error(None, -32600, "invalid request")
        except json.JSONDecodeError as exc:
            response = _error(None, -32700, f"parse error: {exc}")
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
            stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_stdio())
