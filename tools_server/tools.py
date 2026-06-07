"""
IGPO Tool Server - Tool Definitions (Web Search Only)
"""

from typing import Dict, Any

# Web Search Tool Definition
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search a local knowledge base using keyword-based search. Use concise English keywords (2-5 words). For multi-hop questions, search one step at a time.",
    "inputs": {
        "query": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of 1-3 short keyword queries. Use specific names and nouns, not full sentences."
        }
    },
    "example": {"query": ["Martin Frič director", "Martin Frič death date"]}
}


def get_tools(config: Dict[str, Any] = None) -> Dict[str, Dict]:
    """Get available tools (web_search only)."""
    return {"web_search": WEB_SEARCH_TOOL}


def get_tool_names(config: Dict[str, Any] = None) -> list:
    """Get list of available tool names."""
    return ["web_search"]
