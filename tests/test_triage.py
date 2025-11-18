import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app, ingest_node, fetch_order_node


def test_ingest_node_extracts_order_id_and_appends_message():
    state = {
        "ticket_text": "Hello, I'm checking on order ord1005 which arrived late.",
        "messages": [],
    }

    updated = ingest_node(state)

    assert updated["order_id"] == "ORD1005"
    assert len(updated["messages"]) == 1
    assert isinstance(updated["messages"][0], HumanMessage)
    assert updated["messages"][0].content == state["ticket_text"]


def test_fetch_order_node_requires_order_id():
    with pytest.raises(ValueError) as excinfo:
        fetch_order_node({})

    assert "order_id is required" in str(excinfo.value)


def test_triage_invoke_runs_full_workflow():
    client = TestClient(app)
    payload = {
        "ticket_text": "I received a damaged item in my order ORD1001 and need help."
    }

    response = client.post("/triage/invoke", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["order_id"] == "ORD1001"
    assert body["issue_type"] == "damaged_item"
    assert "Ava Chen" in body["reply_text"]
    assert "issue_type=damaged_item" in body["evidence"]
