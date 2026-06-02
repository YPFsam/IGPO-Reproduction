#!/usr/bin/env python3
"""
IGPO Tool Server - Verification Test Script

Run this script to verify the lightweight tool_server works correctly
without requiring network access.

Usage:
    python3 tools_server/test_tools_server.py
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools_server.util import MessageClient
from tools_server.handler import Handler
from tools_server.search.search_api import web_search, mock_search_results, local_tantivy_search


def test_mock_search():
    """Test mock search functionality."""
    print("\n" + "=" * 60)
    print("Test 1: Mock Search Results")
    print("=" * 60)
    
    results = mock_search_results("What is machine learning?")
    
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all('title' in r and 'link' in r and 'snippet' in r for r in results), "Missing required fields"
    
    print(f"✓ Mock search returned {len(results)} results")
    print(f"  - Result 1: {results[0]['title'][:50]}...")
    print("✓ Test passed!")
    return True


def test_web_search_with_mock():
    """Test web_search function with mock mode."""
    print("\n" + "=" * 60)
    print("Test 2: web_search with Mock Mode")
    print("=" * 60)
    
    config = {
        'mock_mode': True,
        'search_engine': 'google',
        'search_top_k': 10
    }
    
    results = web_search("Python programming", config)
    
    assert len(results) > 0, "Expected non-empty results"
    assert 'title' in results[0], "Missing 'title' field"
    
    print(f"✓ web_search returned {len(results)} mock results")
    print("✓ Test passed!")
    return True


def test_local_tantivy_search():
    """Test flat, list, and DR-Venus grouped local-search responses."""
    print("\n" + "=" * 60)
    print("Test 3: Local Tantivy Adapter")
    print("=" * 60)

    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    class LocalSearchHandler(BaseHTTPRequestHandler):
        response_bodies = [
            {"results": [{"query": "q", "passages": [{"title": "Grouped", "text": "grouped body"}]}]},
            [{"title": "List", "text": "list body"}],
            {"results": [{"title": "Flat", "snippet": "flat body"}]},
        ]
        requests = []

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            LocalSearchHandler.requests.append(json.loads(self.rfile.read(length)))
            body = json.dumps(LocalSearchHandler.response_bodies.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), LocalSearchHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = {
        "local_search_url": f"http://127.0.0.1:{server.server_port}/search",
        "local_search_topk": 5,
    }

    try:
        grouped = local_tantivy_search("q", config)
        listed = local_tantivy_search("q", config)
        flat = local_tantivy_search("q", config)
    finally:
        server.shutdown()
        server.server_close()

    assert grouped[0]["snippet"] == "grouped body", grouped
    assert listed[0]["snippet"] == "list body", listed
    assert flat[0]["snippet"] == "flat body", flat
    assert LocalSearchHandler.requests[0]["queries"] == ["q"]
    assert LocalSearchHandler.requests[0]["topk"] == 5
    print("✓ Local Tantivy response variants are compatible")
    print("✓ Test passed!")
    return True


def test_handler_processing():
    """Test Handler processing with mock mode."""
    print("\n" + "=" * 60)
    print("Test 4: Handler Task Processing")
    print("=" * 60)
    
    config = {
        'mock_mode': True,
        'search_engine': 'google',
        'search_top_k': 10,
        'cache_dir': './cache/test_tool_cache'
    }
    
    handler = Handler(config)
    
    # Create test tasks (same format as generation.py sends)
    task_list = [
        {
            "idx": 0,
            "question": "What is deep learning?",
            "think": "I need to search for information about deep learning",
            "tool_call": {
                "name": "web_search",
                "arguments": {
                    "query": ["deep learning basics", "neural network introduction"]
                }
            },
            "total_number": 2
        },
        {
            "idx": 1,
            "question": "How does GPT work?",
            "think": "I should look up how GPT models work",
            "tool_call": {
                "name": "web_search",
                "arguments": {
                    "query": ["GPT model architecture"]
                }
            },
            "total_number": 2
        }
    ]
    
    # Process tasks
    results = handler.handle_all(task_list)
    
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    
    for i, result in enumerate(results):
        assert 'content' in result, f"Task {i} missing 'content'"
        content = result['content']
        assert isinstance(content, list), f"Task {i} content should be a list"
        print(f"✓ Task {i}: processed {len(content)} search queries")
        for item in content:
            assert 'search_query' in item, "Missing 'search_query'"
            assert 'web_page_info_list' in item, "Missing 'web_page_info_list'"
    
    print("✓ Test passed!")
    return True


def test_message_client():
    """Test MessageClient integration."""
    print("\n" + "=" * 60)
    print("Test 5: MessageClient Integration")
    print("=" * 60)
    
    # Patch config to use mock mode
    original_load_config = None
    
    class MockMessageClient(MessageClient):
        def _load_config(self):
            config = super()._load_config()
            config['mock_mode'] = True
            return config
    
    client = MockMessageClient(path='./cache/test_queue')
    
    # Create test task
    task_list = [
        {
            "idx": 0,
            "question": "Test question",
            "think": "Test thinking",
            "tool_call": {
                "name": "web_search",
                "arguments": {
                    "query": ["test query 1", "test query 2"]
                }
            },
            "total_number": 1
        }
    ]
    
    results = client.submit_tasks(task_list)
    
    assert len(results) == 1, "Expected 1 result"
    assert 'content' in results[0], "Missing 'content' in result"
    
    print(f"✓ MessageClient processed {len(results)} tasks")
    print("✓ Test passed!")
    return True


def test_message_client_env_overrides():
    """Test per-run environment overrides without editing tracked config."""
    print("\n" + "=" * 60)
    print("Test 6: MessageClient Environment Overrides")
    print("=" * 60)

    from unittest.mock import patch

    with patch.dict(os.environ, {
        "IGPO_MOCK_SEARCH": "false",
        "IGPO_SEARCH_ENGINE": "local",
        "IGPO_SERPER_API_KEY": "serper-test-key",
        "IGPO_AZURE_BING_SEARCH_SUBSCRIPTION_KEY": "bing-test-key",
        "IGPO_LOCAL_SEARCH_URL": "http://localhost:9999/search",
        "IGPO_LOCAL_SEARCH_TOPK": "7",
        "IGPO_LOCAL_SEARCH_TIMEOUT_SECONDS": "2.5",
    }):
        config = MessageClient()._load_config()

    assert config["mock_mode"] is False
    assert config["search_engine"] == "local"
    assert config["serper_api_key"] == "serper-test-key"
    assert config["azure_bing_search_subscription_key"] == "bing-test-key"
    assert config["local_search_url"] == "http://localhost:9999/search"
    assert config["local_search_topk"] == 7
    assert config["local_search_timeout_seconds"] == 2.5
    print("✓ Environment overrides loaded correctly")
    print("✓ Test passed!")
    return True


def test_invalid_tool_call():
    """Test handling of invalid tool calls."""
    print("\n" + "=" * 60)
    print("Test 7: Invalid Tool Call Handling")
    print("=" * 60)
    
    config = {
        'mock_mode': True,
        'search_engine': 'google',
        'cache_dir': './cache/test_tool_cache'
    }
    
    handler = Handler(config)
    
    # Test various invalid inputs
    test_cases = [
        # Empty tool_call
        {"idx": 0, "tool_call": {}},
        # tool_call is a list (the bug we fixed)
        {"idx": 1, "tool_call": ["web_search", {"query": ["test"]}]},
        # Unknown tool name
        {"idx": 2, "tool_call": {"name": "unknown_tool", "arguments": {}}},
        # Missing arguments
        {"idx": 3, "tool_call": {"name": "web_search"}},
        # query is string instead of list
        {"idx": 4, "tool_call": {"name": "web_search", "arguments": {"query": "single query"}}},
    ]
    
    for i, task in enumerate(test_cases):
        task["question"] = "Test"
        task["think"] = "Test"
        task["total_number"] = len(test_cases)
    
    # Should not raise any exceptions
    try:
        results = handler.handle_all(test_cases)
        print(f"✓ Handled {len(test_cases)} edge cases without crashing")
        for i, result in enumerate(results):
            print(f"  - Case {i}: content type = {type(result.get('content', 'N/A')).__name__}")
        print("✓ Test passed!")
        return True
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False


def test_cache_functionality():
    """Test cache read/write functionality."""
    print("\n" + "=" * 60)
    print("Test 8: Cache Functionality")
    print("=" * 60)
    
    import tempfile
    import shutil
    
    # Use temporary directory for cache
    temp_dir = tempfile.mkdtemp()
    
    try:
        config = {
            'mock_mode': True,
            'search_engine': 'google',
            'cache_dir': temp_dir,
            'cache_ttl_days': 7
        }
        
        handler = Handler(config)
        
        # First request - should use mock and cache
        task_list = [
            {
                "idx": 0,
                "question": "Cache test",
                "think": "Testing cache",
                "tool_call": {
                    "name": "web_search",
                    "arguments": {"query": ["cache test query"]}
                },
                "total_number": 1
            }
        ]
        
        results1 = handler.handle_all(task_list)
        assert 'content' in results1[0], "First request failed"
        
        # Check cache was written
        assert len(handler.search_cache) > 0, "Cache should not be empty"
        print(f"✓ Cache now contains {len(handler.search_cache)} entries")
        
        # Second request - should hit cache
        results2 = handler.handle_all(task_list)
        assert 'content' in results2[0], "Second request failed"
        
        print("✓ Cache read/write working correctly")
        print("✓ Test passed!")
        return True
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  IGPO Tools Server Verification Tests")
    print("  (Using Mock Mode - No Network Required)")
    print("=" * 70)
    
    tests = [
        ("Mock Search", test_mock_search),
        ("Web Search with Mock", test_web_search_with_mock),
        ("Local Tantivy Adapter", test_local_tantivy_search),
        ("Handler Processing", test_handler_processing),
        ("MessageClient Integration", test_message_client),
        ("MessageClient Environment Overrides", test_message_client_env_overrides),
        ("Invalid Tool Call Handling", test_invalid_tool_call),
        ("Cache Functionality", test_cache_functionality),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"  Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✅ All tests passed! The lightweight tool_server is working correctly.\n")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
