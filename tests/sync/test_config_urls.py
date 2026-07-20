"""SSRF guard contract: only http/https base URLs with a host may ever reach
urllib in the naming/embedding backends — file://, gopher:// etc. must fail
the health-check path without any request being attempted."""

from __future__ import annotations

import pytest

from graphify_mesh.sync.config import is_valid_http_base_url


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://host:70/x",
        "ftp://host/",
        "unix:///var/run/x.sock",
        "http://",  # no host
        "https://",
        "not-a-url",
        "",
        "//host-without-scheme",
        "javascript:alert(1)",
    ],
)
def test_invalid_base_urls_rejected(url):
    assert is_valid_http_base_url(url) is False


@pytest.mark.parametrize(
    "url",
    ["http://localhost:11434", "https://ollama.internal:11434", "http://127.0.0.1:11434/v1"],
)
def test_http_https_with_host_accepted(url):
    assert is_valid_http_base_url(url) is True
