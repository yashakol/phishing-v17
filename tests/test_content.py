"""Safety and semantic checks for static content extraction."""

from unittest.mock import patch

from phishing_url.content import extract_html, MAX_TEXT_CHARS, MAX_MARKUP_CHARS


def test_static_form_external_domain_and_offline():
    page = '<html lang="en"><title>Bank login</title><form action="https://other.invalid/collect" method="POST"><input type="password" required><input type="hidden"></form><a href="#">Login</a><script src="https://cdn.invalid/code.js">throw new Error("execute");</script><p>Enter password</p></html>'
    with patch("socket.create_connection", side_effect=AssertionError("No network")), patch("socket.getaddrinfo", side_effect=AssertionError("No DNS")):
        record, numeric = extract_html("https://example.invalid/", page)
    assert numeric["html_forms_external"] == 1
    assert numeric["html_forms_post"] == 1
    assert numeric["html_input_password"] == 1
    assert numeric["html_links_empty"] == 1
    assert numeric["html_scripts_external"] == 1
    assert numeric["cat_html_language"] == "en"
    assert "throw new" not in record["text"]
    assert "Enter password" in record["text"]
    assert record["title"] == "Bank login"


def test_relative_form_and_bounded_views():
    record, numeric = extract_html("https://example.invalid/a", '<html><form action="/send"><input type="email"></form><p>' + "hello " * 10000 + '</p></html>')
    assert numeric["html_forms_external"] == 0
    assert numeric["html_input_email"] == 1
    assert len(record["text"]) <= MAX_TEXT_CHARS
    assert len(record["markup"]) <= MAX_MARKUP_CHARS


def test_empty_and_invalid_pages_keep_schema():
    empty, features = extract_html("https://example.invalid/", "")
    good, valid = extract_html("https://example.invalid/", "<html><p>Hello</p></html>")
    assert features["html_empty"] == 1
    assert features["html_parse_error"] == 1
    assert set(features) == set(valid)
    assert empty["text"] == ""
