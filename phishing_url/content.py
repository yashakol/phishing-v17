"""Static HTML features. Parsing never follows links or executes page code."""

import math
import re
from collections import Counter
from urllib.parse import urljoin, urlsplit

from lxml import etree, html

from .features import group_for_url, split_url


EXTRACTOR_VERSION = 1
MAX_PARSE_CHARS = 1_000_000
MAX_TEXT_CHARS = 12_000
MAX_MARKUP_CHARS = 16_000
TAGS = ("a", "form", "input", "button", "img", "script", "iframe", "link", "meta",
        "div", "span", "p", "table", "style", "svg", "video", "select", "textarea")
INPUT_TYPES = ("password", "email", "text", "hidden", "submit", "checkbox", "radio", "tel", "number")
WORDS = ("login", "password", "verify", "account", "bank", "secure", "wallet", "payment",
         "copyright", "privacy", "contact", "cookie", "forgot", "sign in", "register", "download")
BRANDS = ("paypal", "microsoft", "facebook", "google", "apple", "amazon", "netflix",
          "instagram", "outlook", "dropbox", "dhl", "yahoo", "adobe", "coinbase")
SPACE = re.compile(r"\s+")
LONG_TOKEN = re.compile(r"[a-zA-Z0-9_-]{50,}")
EVENT = re.compile(r"\bon(?:click|load|submit|error|keydown|contextmenu)\s*=", re.I)
EMPTY_LINKS = {"", "#", "javascript:void(0)", "javascript:void(0);", "javascript:;"}


def bounded_text(value, limit):
    value = SPACE.sub(" ", value).strip()
    if len(value) <= limit:
        return value
    half = limit // 2
    return value[:half] + " " + value[-(limit - half - 1):]


def _external(base_url, target, base_group):
    if not target or target.startswith(("javascript:", "mailto:", "tel:", "data:", "#")):
        return False
    try:
        resolved = urljoin(base_url, target)
        return bool(urlsplit(resolved).hostname) and group_for_url(resolved) != base_group
    except ValueError:
        return False


def extract_html(url, content):
    """Return bounded text views and a fixed set of categorical/numeric features."""
    source = content[:MAX_PARSE_CHARS]
    low = source.lower()
    features = {"html_length": len(content), "html_log_length": math.log1p(len(content)),
                "html_truncated": int(len(content) > MAX_PARSE_CHARS),
                "html_empty": int(not content.strip()), "html_parse_error": 0,
                "html_comment_count": low.count("<!--"), "html_event_handlers": len(EVENT.findall(source)),
                "html_data_uri": low.count("data:"), "html_base64": low.count("base64"),
                "html_eval": low.count("eval("), "html_atob": low.count("atob("),
                "html_unescape": low.count("unescape("), "html_document_write": low.count("document.write"),
                "html_window_location": low.count("window.location"),
                "html_rightclick_disabled": int("oncontextmenu" in low),
                "html_percent_ratio": source.count("%") / max(len(source), 1),
                "html_digit_ratio": sum(c.isdigit() for c in source) / max(len(source), 1),
                "html_non_ascii_ratio": sum(ord(c) > 127 for c in source) / max(len(source), 1),
                "cat_html_language": "EMPTY", "cat_html_generator": "EMPTY"}
    for tag in TAGS:
        features["html_tag_" + tag] = 0
    for kind in INPUT_TYPES:
        features["html_input_" + kind] = 0
    for key in ("nodes", "links_external", "links_internal", "links_empty", "links_javascript",
                "forms_external", "forms_empty", "forms_post", "scripts_external", "scripts_inline",
                "images_external", "images_no_alt", "iframes_external", "meta_refresh", "hidden_nodes",
                "anchors_email", "links_http", "inputs_autocomplete_off", "html5_required"):
        features["html_" + key] = 0
    title = ""
    text = ""
    markup = []
    markup_length = 0
    base_group = group_for_url(url)
    try:
        # lxml HTML parsing is local. Network access is explicitly disabled.
        parser = html.HTMLParser(encoding="utf-8", recover=True, no_network=True,
                                 remove_comments=True, huge_tree=True)
        root = html.fromstring(source.encode("utf-8", errors="replace"), parser=parser)
        title = bounded_text(" ".join(root.xpath("//title//text()")), 600)
        text = bounded_text(" ".join(root.xpath(
            "//text()[not(ancestor::script) and not(ancestor::style) and not(ancestor::noscript) and not(ancestor::title)]"
        )), MAX_TEXT_CHARS)
        features["cat_html_language"] = (root.get("lang") or root.get("xml:lang") or "EMPTY").lower()[:40]
        for node in root.iter():
            if not isinstance(node.tag, str):
                continue
            tag = node.tag.lower().rsplit("}", 1)[-1]
            features["html_nodes"] += 1
            if tag in TAGS:
                features["html_tag_" + tag] += 1
            attrs = node.attrib
            if tag == "input":
                kind = (attrs.get("type") or "text").lower()
                if kind in INPUT_TYPES:
                    features["html_input_" + kind] += 1
                features["html_inputs_autocomplete_off"] += attrs.get("autocomplete", "").lower() == "off"
                features["html_html5_required"] += "required" in attrs
            if tag == "a":
                href = attrs.get("href", "").strip().lower()
                if href in EMPTY_LINKS:
                    features["html_links_empty"] += 1
                elif href.startswith("javascript:"):
                    features["html_links_javascript"] += 1
                elif href.startswith("mailto:"):
                    features["html_anchors_email"] += 1
                elif _external(url, href, base_group):
                    features["html_links_external"] += 1
                else:
                    features["html_links_internal"] += 1
                features["html_links_http"] += href.startswith("http:")
            elif tag == "form":
                action = attrs.get("action", "").strip().lower()
                features["html_forms_empty"] += action in EMPTY_LINKS
                features["html_forms_external"] += _external(url, action, base_group)
                features["html_forms_post"] += attrs.get("method", "get").lower() == "post"
            elif tag == "script":
                src = attrs.get("src", "")
                features["html_scripts_external"] += _external(url, src, base_group)
                features["html_scripts_inline"] += not bool(src)
            elif tag == "img":
                features["html_images_external"] += _external(url, attrs.get("src", ""), base_group)
                features["html_images_no_alt"] += not bool(attrs.get("alt", "").strip())
            elif tag == "iframe":
                features["html_iframes_external"] += _external(url, attrs.get("src", ""), base_group)
            elif tag == "meta":
                features["html_meta_refresh"] += attrs.get("http-equiv", "").lower() == "refresh"
                if attrs.get("name", "").lower() == "generator":
                    features["cat_html_generator"] = bounded_text(attrs.get("content", "").lower(), 80) or "EMPTY"
            style = attrs.get("style", "").lower().replace(" ", "")
            features["html_hidden_nodes"] += "hidden" in attrs or "display:none" in style or "visibility:hidden" in style
            if markup_length < MAX_MARKUP_CHARS:
                tokens = ["tag_" + tag]
                for key in ("type", "name", "class", "id", "method", "action", "src", "href", "rel"):
                    if key in attrs:
                        value = LONG_TOKEN.sub("LONGTOKEN", attrs[key].lower())
                        tokens.append(key + "_" + bounded_text(value, 180))
                chunk = " ".join(tokens)
                markup.append(chunk)
                markup_length += len(chunk) + 1
    except (etree.Error, ValueError, TypeError):
        features["html_parse_error"] = 1
        text = bounded_text(re.sub(r"<[^>]*>", " ", source), MAX_TEXT_CHARS)
    features["html_title_length"] = len(title)
    features["html_text_length"] = len(text)
    features["html_text_source_ratio"] = len(text) / max(len(source), 1)
    features["html_links_external_ratio"] = features["html_links_external"] / max(features["html_tag_a"], 1)
    features["html_links_empty_ratio"] = features["html_links_empty"] / max(features["html_tag_a"], 1)
    features["html_input_password_ratio"] = features["html_input_password"] / max(features["html_tag_input"], 1)
    features["html_form_external_ratio"] = features["html_forms_external"] / max(features["html_tag_form"], 1)
    words = (title + " " + text).lower()
    for word in WORDS:
        features["html_word_" + word.replace(" ", "_")] = int(word in words)
    host = (split_url(url).hostname or "").lower()
    for brand in BRANDS:
        features["html_brand_" + brand] = int(brand in words)
        features["html_brand_mismatch_" + brand] = int(brand in title.lower() and brand not in host)
    record = {"url": url, "title": title, "text": text,
              "markup": " ".join(markup)[:MAX_MARKUP_CHARS]}
    return record, features
