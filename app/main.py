from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import re
from typing import Optional, Dict, Any, List

from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------

app = FastAPI(title="Ticket Triage – LangGraph Workflow")

# ---------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOCK_DIR = os.path.join(ROOT, "mock_data")


def load(name: str):
    """Load a JSON file from the mock_data directory."""
    with open(os.path.join(MOCK_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


ORDERS = load("orders.json")
ISSUES = load("issues.json")
REPLIES = load("replies.json")


# ---------------------------------------------------------
# LangGraph State Definition
# ---------------------------------------------------------

class TriageState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph.

    Fields:
      - messages: running conversation / trace (for tools or LLMs later)
      - ticket_text: original ticket text
      - order_id: extracted or provided order id
      - issue_type: classified issue type (refund_request, damaged_item, etc.)
      - evidence: short string explaining why we classified that way
      - recommendation: suggested action for the agent / human
      - order: fake order record loaded from mock_data/orders.json
      - reply_text: final drafted reply to send to the customer
    """

    # messages aggregates using add_messages so each node can append
    messages: Annotated[List[BaseMessage], add_messages]
    ticket_text: str
    order_id: Optional[str]
    issue_type: Optional[str]
    evidence: Optional[str]
    recommendation: Optional[str]
    order: Optional[Dict[str, Any]]
    reply_text: Optional[str]


# ---------------------------------------------------------
# Node helpers (pure business logic)
# ---------------------------------------------------------

def _classify_issue_text(ticket_text: str) -> Dict[str, str]:
    """Rule-based classification using issues.json."""
    text_lower = ticket_text.lower()
    issue_type = "other"

    for row in ISSUES:
        keyword = row["keyword"].lower()
        if keyword in text_lower:
            issue_type = row["issue_type"]
            break

    evidence = f"Matched keyword for issue_type={issue_type}"

    recommendation_map = {
        "refund_request": "Issue a refund after verification.",
        "damaged_item": "Send a replacement and consider partial refund.",
        "late_delivery": "Apologize and offer discount or refund shipping.",
        "missing_item": "Ship the missing item and confirm address.",
    }
    recommendation = recommendation_map.get(issue_type, "Review ticket manually.")

    return {
        "issue_type": issue_type,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _draft_reply(issue_type: str, order: Dict[str, Any]) -> str:
    """Render a reply template from replies.json with order data."""
    template = None
    for row in REPLIES:
        if row["issue_type"] == issue_type:
            template = row["template"]
            break

    if template is None:
        template = (
            "Hi {{customer_name}}, thanks for reaching out about order {{order_id}}. "
            "Our team is reviewing your request and will follow up shortly."
        )

    customer_name = order.get("customer_name", "there")
    order_id = order.get("order_id", "")

    return (
        template
        .replace("{{customer_name}}", customer_name)
        .replace("{{order_id}}", order_id)
    )


# ---------------------------------------------------------
# LangGraph Nodes
# ---------------------------------------------------------

def ingest_node(state: TriageState) -> TriageState:
    """
    Ingest node:
      - Ensures ticket_text is present
      - Extracts order_id from text if missing
      - Appends a HumanMessage to messages
    """
    if "ticket_text" not in state:
        raise ValueError("ticket_text is required in state for ingest_node")

    ticket_text = state["ticket_text"]
    order_id = state.get("order_id")

    # Try to extract order_id like ORD1001 from free text if not provided.
    if not order_id:
        match = re.search(r"(ORD\d{4})", ticket_text, re.IGNORECASE)
        if match:
            order_id = match.group(1).upper()

    messages = list(state.get("messages", []))
    messages.append(HumanMessage(content=ticket_text))

    new_state: TriageState = dict(state)
    new_state["messages"] = messages
    new_state["order_id"] = order_id
    return new_state


def classify_issue_node(state: TriageState) -> TriageState:
    """
    Classify node:
      - Reads ticket_text
      - Sets issue_type, evidence, recommendation
    """
    ticket_text = state.get("ticket_text") or ""
    classification = _classify_issue_text(ticket_text)

    new_state: TriageState = dict(state)
    new_state.update(classification)
    return new_state


def fetch_order_node(state: TriageState) -> TriageState:
    """
    Fetch order node:
      - Requires order_id
      - Loads fake order from ORDERS
    """
    order_id = state.get("order_id")
    if not order_id:
        raise ValueError("order_id is required before fetch_order_node")

    order = next((o for o in ORDERS if o["order_id"] == order_id), None)
    if not order:
        # We propagate an error; FastAPI layer will turn this into HTTP.
        raise ValueError(f"order not found for id={order_id}")

    new_state: TriageState = dict(state)
    new_state["order"] = order
    return new_state


def draft_reply_node(state: TriageState) -> TriageState:
    """
    Draft reply node:
      - Requires issue_type and order
      - Produces reply_text
    """
    issue_type = state.get("issue_type") or "other"
    order = state.get("order")

    if order is None:
        raise ValueError("order is required before draft_reply_node")

    reply_text = _draft_reply(issue_type, order)

    new_state: TriageState = dict(state)
    new_state["reply_text"] = reply_text
    return new_state


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

def build_triage_graph():
    """
    Build a deterministic workflow:

      ingest -> classify_issue -> fetch_order -> draft_reply -> END
    """
    workflow = StateGraph(TriageState)

    workflow.add_node("ingest", ingest_node)
    workflow.add_node("classify_issue", classify_issue_node)
    workflow.add_node("fetch_order", fetch_order_node)
    workflow.add_node("draft_reply", draft_reply_node)

    workflow.set_entry_point("ingest")

    workflow.add_edge("ingest", "classify_issue")
    workflow.add_edge("classify_issue", "fetch_order")
    workflow.add_edge("fetch_order", "draft_reply")
    workflow.add_edge("draft_reply", END)

    return workflow.compile()


graph = build_triage_graph()


# ---------------------------------------------------------
# FastAPI models
# ---------------------------------------------------------

class TriageInput(BaseModel):
    ticket_text: str
    order_id: Optional[str] = None


# ---------------------------------------------------------
# FastAPI endpoint that invokes the LangGraph
# ---------------------------------------------------------

@app.post("/triage/invoke")
def triage_invoke(body: TriageInput):
    """
    FastAPI entrypoint that delegates to the LangGraph workflow.

    - Initializes TriageState
    - Runs the compiled graph
    - Maps graph state to an HTTP JSON response
    """
    initial_state: TriageState = {
        "messages": [],
        "ticket_text": body.ticket_text,
        "order_id": body.order_id,
        "issue_type": None,
        "evidence": None,
        "recommendation": None,
        "order": None,
        "reply_text": None,
    }

    try:
        final_state = graph.invoke(initial_state)
    except ValueError as e:
        # Turn node-level errors into HTTP responses
        message = str(e)
        if "order_id is required" in message:
            raise HTTPException(status_code=400, detail=message)
        if "order not found" in message:
            raise HTTPException(status_code=404, detail=message)
        raise HTTPException(status_code=500, detail=message)

    if not final_state.get("order"):
        # Safety net: should not happen if fetch_order_node succeeded
        raise HTTPException(status_code=500, detail="order missing in final state")

    return {
        "order_id": final_state.get("order_id"),
        "issue_type": final_state.get("issue_type"),
        "order": final_state.get("order"),
        "reply_text": final_state.get("reply_text"),
        "evidence": final_state.get("evidence"),
        "recommendation": final_state.get("recommendation"),
    }