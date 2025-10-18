"""
Demo script for testing the code search API.

Usage: python demo_query.py 'your question' [k]
"""

import os
import sys
import json
import httpx

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def main() -> None:
    """
    Execute a search query against the code search API.
    
    Args:
        sys.argv[1]: Search query string
        sys.argv[2]: Optional number of results to return (default: 5)
    """
    if len(sys.argv) < 2:
        print("Usage: python demo_query.py 'your question' [k]")
        sys.exit(1)
    
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    payload = {"query": query, "k": k}
    
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{API_URL}/search", json=payload)
        resp.raise_for_status()
        data = resp.json()
    
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
