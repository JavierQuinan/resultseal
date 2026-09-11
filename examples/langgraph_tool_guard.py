"""Guard LangGraph ToolNode outputs before they enter graph state.

Install ``langgraph`` to run the complete example. The ResultSeal guard itself
remains framework-free, so the core test suite can exercise it without the
optional dependency installed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from resultseal.contracts import load_contract_file
from resultseal.limits import Limits
from resultseal.models import Decision
from resultseal.normalize import normalize
from resultseal.rules import ReferenceClock, evaluate

CONTRACT_PATH = Path(__file__).with_name("customer_contract.json")


class BlockedObservation(RuntimeError):
    """Raised when ResultSeal blocks an unsafe tool observation."""


def guard_search_result(result: object) -> object:
    """Return a verified result or block it before LangGraph state injection."""
    clock = ReferenceClock(now=datetime.now(UTC))
    contract = load_contract_file(CONTRACT_PATH, Limits())

    normalization = normalize(
        {
            "kind": "json",
            "source_ref": "mcp://crm-server",
            "target_ref": "customer:42",
            "tool_name": "search_customer",
            "body": result,
        },
        clock,
    )

    evaluation = evaluate(
        normalization.envelope,
        normalization.payload,
        contract,
        clock,
    )

    if evaluation.decision is not Decision.SEALED:
        codes = ", ".join(evaluation.reason_codes)
        raise BlockedObservation(
            f"ResultSeal blocked the tool observation: {codes}"
        )

    return normalization.payload


def build_node():  # type annotations would require optional framework dependencies
    """Create a LangGraph ToolNode whose tool result is guarded before state use."""
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode

    @tool
    def search_customer(query: str) -> object:
        """Search the customer database."""
        # Stand-in for a database/API call. Guarding here means ToolNode never
        # receives an unverified empty payload to wrap as a ToolMessage.
        database_result: list[object] = []
        return guard_search_result(database_result)

    return ToolNode([search_customer])


if __name__ == "__main__":
    # Building the node requires the optional LangGraph dependency. The
    # framework-free guard above is intentionally importable without it.
    build_node()
