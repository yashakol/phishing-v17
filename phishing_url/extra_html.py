"""Extract additional inert script, CSS, metadata and resource text locally."""

import html
import math
import re
from collections import Counter
from urllib.parse import urljoin

from .features import group_for_url

VERSION = 1
SCRIPT = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.I | re.S)
STYLE = re.compile(r"<style\b[^>]*>(.*?)</style\s*>", re.I | re.S)
TAGS = re.compile(r"<(meta|link|iframe|form|img|base|input)\b([^>]{0,6000})>", re.I | re.S)
ATTR = re.compile(r'''([a-zA-Z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))''')
LONG = re.compile(r"[a-zA-Z0-9_+/=-]{60,}")
SPACE = re.compile(r"\s+")
SIGNALS = ("xmlhttprequest", "fetch(", "axios", "preventdefault", "localstorage", "document.cookie",
           "location.replace", "location.href", "fromcharcode", "telegram", "webhook", "onerror",
           "type=password", "password", "recaptcha", "google-analytics", "googletagmanager", "eval(", "atob(")


def attributes(source):
    return {match.group(1).lower(): html.unescape(next(value for value in match.groups()[1:] if value is not None))
            for match in ATTR.finditer(source)}


def bounded(source, limit):
    source = SPACE.sub(" ", LONG.sub("LONGTOKEN", source)).strip()
    if len(source) <= limit:
        return source
    return source[:limit // 2] + " " + source[-(limit - limit // 2 - 1):]


def resource_group(url, target):
    try:
        return group_for_url(urljoin(url, target))
    except ValueError:
        return "INVALID"


def extract_extra(url, content):
    source = content[:1_000_000]
    scripts, resources, styles, meta = [], [], [], []
    canonical, og_url = "", ""
    host = group_for_url(url)
    resource_domains = Counter()
    inline_count = 0
    script_length = 0
    for match in SCRIPT.finditer(source):
        attr = attributes(match.group(1))
        if attr.get("src"):
            resources.append("script " + attr["src"])
            resource_domains[resource_group(url, attr["src"])] += 1
        else:
            inline_count += 1
            script_length += len(match.group(2))
            scripts.append(bounded(match.group(2), 10000))
    for match in STYLE.finditer(source):
        styles.append(bounded(match.group(1), 16000))
    for match in TAGS.finditer(source):
        tag, raw = match.groups()
        tag = tag.lower()
        attr = attributes(raw)
        if tag == "meta":
            name = attr.get("name", attr.get("property", attr.get("http-equiv", ""))).lower()
            value = attr.get("content", "")
            meta.append(name + " " + value)
            if name == "og:url":
                og_url = value
        if tag == "link" and "canonical" in attr.get("rel", "").lower():
            canonical = attr.get("href", "")
        for name in ("href", "src", "action"):
            if attr.get(name):
                value = attr[name]
                resources.append(tag + " " + name + " " + value)
                resource_domains[resource_group(url, value)] += 1
    js = bounded(" ".join(scripts), 18000)
    css = bounded(" ".join(styles), 18000)
    resource = bounded(" ".join(resources), 16000)
    metadata = bounded(" ".join(meta), 6000)
    canonical_domain = resource_group(url, canonical) if canonical else "EMPTY"
    og_domain = resource_group(url, og_url) if og_url else "EMPTY"
    features = {"extra_script_length": script_length, "extra_script_log_length": math.log1p(script_length),
                "extra_inline_scripts": inline_count, "extra_style_length": sum(map(len, styles)),
                "extra_meta_length": len(metadata), "extra_resources": sum(resource_domains.values()),
                "extra_resource_domains": len(resource_domains),
                "extra_invalid_resources": resource_domains["INVALID"],
                "extra_resources_same_domain": resource_domains[host],
                "extra_canonical_mismatch": int(bool(canonical) and canonical_domain != host),
                "extra_og_mismatch": int(bool(og_url) and og_domain != host),
                "cat_extra_canonical_domain": canonical_domain, "cat_extra_og_domain": og_domain,
                "cat_extra_main_resource_domain": resource_domains.most_common(1)[0][0] if resource_domains else "EMPTY"}
    lower = (js + " " + metadata + " " + resource).lower()
    for n, signal in enumerate(SIGNALS):
        features[f"extra_signal_{n}"] = lower.count(signal)
    return {"url": url, "scripts": js, "styles": css, "meta": metadata, "resources": resource}, features
