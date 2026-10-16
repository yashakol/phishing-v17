"""Train-only conditional origin affinity, using full titles and cached domains.

No classifier, raw HTML normalization, handlisted brands, network or JavaScript.
"""
import hashlib
import html
import ipaddress
import json
import re
import unicodedata
from collections import defaultdict

FAMILIES = ("title", "canonical_title", "og_title", "resource_title")
FIELDS = {"canonical_title": "canonical", "og_title": "og", "resource_title": "resource"}
STOP = frozenset("a an and are as at be by for from in into is it of on or our the this to with www your welcome home loading error page untitled redirecting please wait just moment".split())
TOKENS = re.compile(r"[^\W\d_]+", re.UNICODE)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_title(text):
    if not isinstance(text, str): raise TypeError("Title must be cached plain text")
    return " ".join(unicodedata.normalize("NFKC", html.unescape(text)).casefold().split())


def valid_title(title):
    words = {word for word in TOKENS.findall(title) if len(word) >= 3 and word not in STOP}
    return len(title) >= 12 and len(words) >= 2


def valid_domain(value):
    if not isinstance(value, str): return ""
    value = value.strip().lower().rstrip(".")
    if not value or value in {"empty", "invalid", "none", "nan"} or "." not in value or any(c in value for c in "/:@ "):
        return ""
    try:
        ipaddress.ip_address(value)
        return ""
    except ValueError:
        return value


def prepare_row(index, url, title, domain, canonical, og, resource):
    title = normalize_title(title)
    row = dict(index=int(index), url=url, domain=valid_domain(domain), title=title,
               canonical=valid_domain(canonical), og=valid_domain(og), resource=valid_domain(resource))
    row["keys"] = row_keys(row)
    return row


def row_keys(row):
    if not row["domain"] or not valid_title(row["title"]): return {}
    result = {"title": "title:" + digest(row["title"])}
    for family, field in FIELDS.items():
        if row[field]: result[family] = family + ":" + digest(json.dumps([row["title"], row[field]], ensure_ascii=True, separators=(",", ":")))
    return result


def index_digest(indices):
    import numpy as np
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def training_digest(rows, labels):
    h = hashlib.sha256()
    for row, label in zip(rows, labels, strict=True):
        h.update(row["url"].encode("utf-8")); h.update(b"\x00"); h.update(bytes([int(label)]))
    return h.hexdigest()


class OriginAffinity:
    def __init__(self, families=FAMILIES, minimum_benign_urls=3, minimum_phishing_urls=3,
                 minimum_phishing_domains=2, maximum_source_domains=2, minimum_urls_per_source=2,
                 alpha=1., weight_pseudocount=1., maximum_weight=.9, confidence_limit=.98):
        self.config = dict(families=list(families), minimum_benign_urls=minimum_benign_urls, minimum_phishing_urls=minimum_phishing_urls,
                           minimum_phishing_domains=minimum_phishing_domains, maximum_source_domains=maximum_source_domains,
                           minimum_urls_per_source=minimum_urls_per_source, alpha=alpha, weight_pseudocount=weight_pseudocount,
                           maximum_weight=maximum_weight, confidence_limit=confidence_limit)
        if not families or not set(families) <= set(FAMILIES): raise ValueError("Unknown families")
        if minimum_benign_urls < 3 or minimum_phishing_urls < 3 or minimum_phishing_domains < 2: raise ValueError("Unsafe support")
        if maximum_source_domains not in (1, 2) or minimum_urls_per_source < 2: raise ValueError("Unsafe sources")
        if alpha <= 0 or weight_pseudocount <= 0 or not 0 < maximum_weight <= 1 or confidence_limit not in (.98, 1.): raise ValueError("Invalid confidence/count policy")

    def fit(self, rows, labels, expected_indices, forbidden_indices=()):
        import numpy as np
        indices = [int(row["index"]) for row in rows]
        expected = np.asarray(expected_indices, dtype=np.int64)
        # Reject validation-row use BEFORE examining any labels.
        if len(set(indices)) != len(indices) or not np.array_equal(indices, expected) or set(indices) & set(forbidden_indices):
            raise ValueError("Training row guard rejected")
        labels = np.asarray(labels)
        if labels.shape != (len(rows),) or not np.isin(labels, [0, 1]).all(): raise ValueError("Aligned binary training labels required")
        if len({r["url"] for r in rows}) != len(rows): raise ValueError("Training URLs must be unique")
        for row in rows:
            if row["keys"] != row_keys(row): raise ValueError("Stale row keys")
        title_domain = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        groups = {}
        self.known_training_domains = sorted({row["domain"] for row in rows if row["domain"]})
        for row, label in zip(rows, labels, strict=True):
            if not row["keys"]: continue
            label = int(label)
            title_domain[row["title"]][row["domain"]][label] += 1
            for family in self.config["families"]:
                key = row["keys"].get(family)
                if key is None: continue
                if key not in groups:
                    groups[key] = dict(family=family, title=row["title"], origin=row[FIELDS[family]] if family in FIELDS else "", domain_counts={}, training_indices=[])
                g = groups[key]
                g["domain_counts"].setdefault(row["domain"], [0, 0])[label] += 1
                g["training_indices"].append(row["index"])
        self.title_benign_domains = {title: sorted(d for d, n in counts.items() if n[0]) for title, counts in title_domain.items()}
        self.evidence, self.ledger, self.statistics = {}, {}, {}
        for family in self.config["families"]: self.statistics[family] = {"observed": 0, "accepted": 0, "rejections": {}}
        c = self.config
        for key, g in sorted(groups.items()):
            counts = g["domain_counts"]
            benign = sum(n[0] for n in counts.values())
            phishing = sum(n[1] for n in counts.values())
            sources = sorted(d for d, n in counts.items() if n[0])
            foreign = sorted(d for d, n in counts.items() if n[1])
            if any(min(n) > 0 for n in counts.values()): reason = "mixed_key_domain"
            elif any(min(title_domain[g["title"]][d]) > 0 for d in counts): reason = "mixed_full_title_domain"
            elif benign < c["minimum_benign_urls"]: reason = "insufficient_benign_urls"
            elif phishing < c["minimum_phishing_urls"]: reason = "insufficient_phishing_urls"
            elif len(sources) > c["maximum_source_domains"]: reason = "too_many_source_domains"
            elif any(counts[d][0] < c["minimum_urls_per_source"] for d in sources): reason = "weak_source_domain"
            elif len(foreign) < c["minimum_phishing_domains"]: reason = "insufficient_phishing_domains"
            elif g["origin"] and g["origin"] not in sources: reason = "untrusted_declared_origin"
            else: reason = "accepted"
            g.update(benign_urls=benign, phishing_urls=phishing, source_domains=sources, phishing_domains=foreign, status=reason)
            stats = self.statistics[g["family"]]
            stats["observed"] += 1
            if reason == "accepted":
                g["source_posterior"] = c["alpha"] / (benign + 2*c["alpha"])
                g["foreign_posterior"] = (phishing + c["alpha"]) / (phishing + 2*c["alpha"])
                support = min(benign, phishing)
                g["weight"] = min(c["maximum_weight"], support/(support+c["weight_pseudocount"]))
                stats["accepted"] += 1
                self.evidence[key] = g
            else: stats["rejections"][reason] = stats["rejections"].get(reason, 0) + 1
            self.ledger[key] = g
        self.scope = dict(training_rows=len(rows), training_indices_sha256=index_digest(indices), ordered_training_sha256=training_digest(rows, labels))
        return self

    def predict(self, rows, baseline, hard_mask=None):
        import numpy as np
        p = np.asarray(baseline, dtype=float)
        if p.shape != (len(rows),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any(): raise ValueError("Invalid baseline")
        hard = np.zeros(len(rows), bool) if hard_mask is None else np.asarray(hard_mask, bool)
        if hard.shape != p.shape: raise ValueError("Invalid hard mask")
        hard = hard | (p == 0) | (p == 1)
        limit = self.config["confidence_limit"]
        protected = hard | (p < 1-limit) | (p > limit)
        out = p.copy()
        flags = {name: np.zeros(len(rows), bool) for name in ("matched", "eligible", "adjusted", "conflict", "protected", "source", "new_foreign", "globally_unseen_foreign", "known_foreign_only")}
        flags["hard_protected"] = hard
        traces = []
        known_domains = set(self.known_training_domains)
        for i, row in enumerate(rows):
            if row["keys"] != row_keys(row): raise ValueError("Stale prediction keys")
            candidates, matching, known_foreign = [], [], []
            for family in self.config["families"]:
                key = row["keys"].get(family)
                if key not in self.evidence: continue
                g = self.evidence[key]
                matching.append(key)
                if row["domain"] in g["source_domains"]:
                    direction, posterior = 0, g["source_posterior"]
                elif row["domain"] in g["domain_counts"]:
                    known_foreign.append(key)
                    continue
                elif row["domain"] in self.title_benign_domains.get(row["title"], []):
                    flags["conflict"][i] = True
                    continue
                else:
                    direction, posterior = 1, g["foreign_posterior"]
                candidates.append((g["weight"] * abs(posterior-.5), key, direction, posterior, g["weight"]))
            flags["matched"][i] = bool(matching)
            flags["conflict"][i] |= len({item[2] for item in candidates}) > 1
            flags["known_foreign_only"][i] = bool(known_foreign) and not candidates
            trace = dict(index=row["index"], matched_keys=matching, known_foreign_keys=known_foreign, chosen_key=None)
            if candidates and not flags["conflict"][i]:
                _, chosen, direction, posterior, weight = min(candidates)
                flags["eligible"][i] = True
                flags["source"][i] = direction == 0
                flags["new_foreign"][i] = direction == 1
                flags["globally_unseen_foreign"][i] = direction == 1 and row["domain"] not in known_domains
                flags["protected"][i] = protected[i]
                trace.update(chosen_key=chosen, direction=direction, posterior=posterior, weight=weight)
                if not protected[i]:
                    out[i] = (1-weight)*p[i] + weight*posterior
                    flags["adjusted"][i] = True
            traces.append(trace)
        assert np.array_equal(out[hard], p[hard])
        assert np.array_equal(out[flags["conflict"]], p[flags["conflict"]])
        return out, flags, traces

    def to_dict(self):
        return dict(format_version=1, config=self.config, scope=self.scope, evidence=self.evidence, statistics=self.statistics,
                    title_benign_domains=self.title_benign_domains, known_training_domains=self.known_training_domains)

    @classmethod
    def from_dict(cls, saved, expected_scope=None):
        if saved["format_version"] != 1: raise ValueError("Unknown format")
        if expected_scope is not None and saved["scope"] != expected_scope: raise ValueError("Saved training scope differs")
        obj = cls(**saved["config"])
        for name in ("scope", "evidence", "statistics", "title_benign_domains", "known_training_domains"): setattr(obj, name, saved[name])
        return obj
