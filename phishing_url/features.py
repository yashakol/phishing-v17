"""URL text and structural features. No DNS, HTTP or WHOIS requests."""

import ipaddress
import math
import re
from collections import Counter
from functools import lru_cache
from urllib.parse import unquote, urlsplit, SplitResult

import pandas as pd
import tldextract


# The package's bundled Public Suffix List is used offline, including private hosting suffixes.
EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")
KEYWORDS = ("login", "signin", "sign-in", "verify", "verification", "account", "secure", "security",
            "update", "password", "recover", "confirm", "wallet", "payment", "invoice", "bank",
            "paypal", "microsoft", "office", "outlook", "support", "auth", "session", "webscr",
            "suspend", "unlock", "paczka", "platnosc", "tracking", "bonus", "gift", "free")
SHORTENERS = {"bit.ly", "t.co", "tinyurl.com", "is.gd", "goo.gl", "ow.ly", "cutt.ly", "rb.gy"}


def split_url(url):
    try:
        return urlsplit(url.strip())
    except ValueError:
        return SplitResult("", "", url, "", "")


@lru_cache(maxsize=150000)
def domain_parts(host):
    try:
        ipaddress.ip_address(host)
        return host, "", "", host, True
    except ValueError:
        parts = EXTRACTOR(host)
        domain = f"{parts.domain}.{parts.suffix}" if parts.suffix else host
        return domain or host, parts.suffix, parts.subdomain, parts.domain, False


def group_for_url(url):
    parsed = split_url(url)
    return domain_parts((parsed.hostname or "").lower())[0] or "INVALID"


def entropy(text):
    if not text:
        return 0.0
    return -sum((n / len(text)) * math.log2(n / len(text)) for n in Counter(text).values())


def longest_run(text, pattern):
    return max((len(m.group()) for m in re.finditer(pattern, text)), default=0)


def text_tokens(url):
    parsed = split_url(url)
    host = (parsed.hostname or "").lower()
    domain, suffix, subdomain, stem, _ = domain_parts(host)
    tokens = [f"scheme={parsed.scheme.lower()}", f"host={host}", f"domain={domain}", f"suffix={suffix}"]
    tokens += ["hostword=" + token.lower() for token in WORD_PATTERN.findall(host)]
    tokens += ["pathword=" + token.lower() for token in WORD_PATTERN.findall(unquote(parsed.path))]
    tokens += ["queryword=" + token.lower() for token in WORD_PATTERN.findall(unquote(parsed.query))]
    tokens += ["word=" + token.lower() for token in WORD_PATTERN.findall(unquote(url))]
    return tokens


def feature_row(url):
    parsed = split_url(url)
    host = (parsed.hostname or "").lower()
    domain, suffix, subdomain, stem, is_ip = domain_parts(host)
    path = unquote(parsed.path).lower()
    query = unquote(parsed.query).lower()
    lower = url.lower()
    segments = [segment for segment in path.split("/") if segment]
    try:
        port = parsed.port or 0
        invalid_port = 0
    except ValueError:
        port, invalid_port = 0, 1
    out = {
        "cat_scheme": parsed.scheme.lower() or "EMPTY", "cat_host": host or "EMPTY",
        "cat_domain": domain or "EMPTY", "cat_suffix": suffix or "EMPTY",
        "cat_extension": (segments[-1].rsplit(".", 1)[-1][:16] if segments and "." in segments[-1] else "NONE"),
        "cat_first_path": segments[0][:80] if segments else "EMPTY",
        "is_ip": int(is_ip), "is_https": int(parsed.scheme.lower() == "https"),
        "has_credentials": int(parsed.username is not None), "port": port, "invalid_port": invalid_port,
        "host_labels": len(host.split(".")), "subdomain_labels": len(subdomain.split(".")) if subdomain else 0,
        "stem_length": len(stem), "suffix_length": len(suffix), "subdomain_length": len(subdomain),
        "path_depth": len(segments), "max_path_segment": max(map(len, segments), default=0),
        "query_parameters": parsed.query.count("&") + bool(parsed.query), "fragment_length": len(parsed.fragment),
        "punycode": int("xn--" in host), "non_ascii": sum(ord(c) > 127 for c in url),
        "host_starts_www": int(host.startswith("www.")), "shortener": int(domain in SHORTENERS),
        "embedded_http": max(0, lower.count("http") - 1), "percent_encoded": url.count("%"),
        "host_in_path": int(bool(host) and host in path), "domain_in_path": int(bool(domain) and domain in path),
        "ends_slash": int(parsed.path.endswith("/")), "has_query": int(bool(parsed.query)),
    }
    for name, text in [("url", url), ("host", host), ("path", path), ("query", query), ("stem", stem)]:
        size = max(len(text), 1)
        out[f"{name}_length"] = len(text)
        out[f"{name}_digits"] = sum(c.isdigit() for c in text)
        out[f"{name}_digit_ratio"] = out[f"{name}_digits"] / size
        out[f"{name}_letters_ratio"] = sum(c.isalpha() for c in text) / size
        out[f"{name}_entropy"] = entropy(text)
        out[f"{name}_max_digit_run"] = longest_run(text, r"\d+")
        out[f"{name}_max_alpha_run"] = longest_run(text, r"[a-zA-Z]+")
        for char, label in [(".", "dots"), ("-", "hyphens"), ("_", "underscores"), ("/", "slashes"),
                            ("@", "ats"), ("=", "equals"), ("%", "percents"), ("~", "tildes")]:
            out[f"{name}_{label}"] = text.count(char)
    out["uppercase_ratio"] = sum(c.isupper() for c in url) / max(len(url), 1)
    for keyword in KEYWORDS:
        out["keyword_" + keyword] = int(keyword in lower)
        out["host_keyword_" + keyword] = int(keyword in host)
    return out


def make_features(urls):
    return pd.DataFrame([feature_row(url) for url in urls])


def categorical_columns(frame):
    return [column for column in frame if column.startswith("cat_")]
