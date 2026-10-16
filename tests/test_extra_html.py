import socket

from phishing_url.extra_html import extract_extra, attributes


def test_inert_code_and_external_metadata(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    record, features = extract_extra("https://copy.example.net/login", '''
      <meta property="og:url" content="https://bank.example.org/">
      <link rel="canonical" href="https://bank.example.org/login">
      <script src="https://static.example.org/app.js"></script>
      <script>fetch('/collect'); localStorage.setItem('password','x');</script>
      <style>.signin { display:none }</style>
    ''')
    assert "localStorage" in record["scripts"]
    assert ".signin" in record["styles"]
    assert features["extra_canonical_mismatch"] == features["extra_og_mismatch"] == 1
    assert features["extra_inline_scripts"] == 1


def test_invalid_embedded_url_does_not_abort_extraction():
    record, features = extract_extra("https://example.org/", '<img src="http://[malformed/image"><link rel="canonical" href="http://[">')
    assert features["extra_invalid_resources"] == 2
    assert features["cat_extra_canonical_domain"] == "INVALID"


def test_attributes_keep_empty_values_and_decode_entities():
    assert attributes('name="" content="a&amp;b" id=example') == {"name": "", "content": "a&b", "id": "example"}


def test_large_payload_is_bounded_without_execution():
    record, features = extract_extra("https://example.org/", '<script>' + 'window.location.href=\"/login\"; ' * 20000 + '</script>')
    assert len(record["scripts"]) <= 18000
    assert features["extra_script_length"] > len(record["scripts"])
