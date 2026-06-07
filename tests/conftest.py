"""Test configuration: force deterministic mock mode for every test.

Tests never hit the network or an LLM - they run against the deterministic
``MockChatModel`` and the inline service mocks.
"""

import os

import pytest

os.environ["HIREGRAPH_USE_MOCKS"] = "true"


@pytest.fixture(autouse=True)
def _fresh_settings():
    """Clear the cached Settings so the mock env is always picked up."""
    from hiregraph.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
