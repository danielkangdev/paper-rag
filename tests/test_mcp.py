import io
import json

from scirag import config
from scirag.mcp.server import handle, run_stdio
from scirag.mcp.tools import call_tool, list_tools

from .conftest import PAPER_A, make_pdf


def _payload(result: dict) -> dict:
    return json.loads(result["content"][0]["text"])


def test_lists_six_tools_with_json_schema():
    tools = {t["name"]: t for t in list_tools()}
    assert set(tools) == {"ingest_paper", "list_papers", "search_papers", "ask_paper",
                          "get_section", "delete_paper"}
    schema = tools["search_papers"]["inputSchema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["query"]
    assert set(schema["properties"]["mode"]["enum"]) == {"dense", "sparse", "hybrid", "hybrid_rerank"}


def test_invalid_arguments_rejected(settings):
    r = call_tool("search_papers", {"query": "", "top_k": 500})
    assert r["isError"] and _payload(r)["error"] == "invalid_arguments"
    r = call_tool("ingest_paper", {"arxiv_id": "1", "pdf_path": "a.pdf"})
    assert r["isError"] and _payload(r)["error"] == "invalid_arguments"


def test_unknown_tool(settings):
    assert _payload(call_tool("rm_rf", {}))["error"] == "unknown_tool"


def test_writes_refused_by_default(settings, corpus):
    r = call_tool("delete_paper", {"paper_id": corpus[0]})
    assert r["isError"] and _payload(r)["error"] == "writes_disabled"
    r = call_tool("ingest_paper", {"pdf_path": "a.pdf"})
    assert _payload(r)["error"] == "writes_disabled"


def test_ingest_only_from_pdf_dir(settings, tmp_path):
    settings.security.allow_writes = True
    outside = tmp_path / "outside"
    outside.mkdir()
    make_pdf(outside / "x.pdf", PAPER_A)
    for path in ("../outside/x.pdf", str(outside / "x.pdf")):
        r = call_tool("ingest_paper", {"pdf_path": path})
        assert _payload(r)["error"] == "path_not_allowed", path
    make_pdf(tmp_path / "pdfs" / "ok.pdf", PAPER_A)
    r = call_tool("ingest_paper", {"pdf_path": "ok.pdf"})
    assert not r["isError"] and _payload(r)["status"] == "indexed"


def test_read_tools(settings, corpus):
    papers = _payload(call_tool("list_papers", {}))["papers"]
    assert {p["paper_id"] for p in papers} == set(corpus)
    res = _payload(call_tool("search_papers", {"query": "reflection tokens", "top_k": 3}))
    assert res["hits"] and res["hits"][0]["paper_id"] == corpus[0]
    secs = _payload(call_tool("get_section", {"paper_id": corpus[0]}))["sections"]
    assert "Method" in [s["name"] for s in secs]
    sec = _payload(call_tool("get_section", {"paper_id": corpus[0], "section": "method"}))
    assert "Reflection tokens" in sec["text"]
    assert _payload(call_tool("get_section", {"paper_id": "nope"}))["error"] == "not_found_or_invalid"


def test_delete_when_allowed(settings, corpus):
    config.get_settings().security.allow_writes = True
    r = _payload(call_tool("delete_paper", {"paper_id": corpus[1]}))
    assert r["deleted"] is True
    papers = _payload(call_tool("list_papers", {}))["papers"]
    assert [p["paper_id"] for p in papers] == [corpus[0]]


def test_jsonrpc_stdio_roundtrip(settings, corpus):
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "list_papers", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 4, "method": "nope"},
    ]
    stdin = io.StringIO("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    stdout = io.StringIO()
    run_stdio(stdin, stdout)
    out = [json.loads(x) for x in stdout.getvalue().splitlines()]
    assert [o.get("id") for o in out] == [1, 2, 3, 4, None]
    assert out[0]["result"]["serverInfo"]["name"] == "scirag"
    assert len(out[1]["result"]["tools"]) == 6
    assert out[3]["error"]["code"] == -32601
    assert out[4]["error"]["code"] == -32700


def test_notification_gets_no_response():
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
