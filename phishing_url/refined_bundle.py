"""Inference of the learned HTML ensemble with training-only refined evidence."""

import numpy as np

from .content_bundle import predict_content
from .content_refinement import REFINEMENT_VERSION, apply_evidence, signature_keys


def predict_refined(bundle, cache, url_frame=None):
    if bundle['refinement_version'] != REFINEMENT_VERSION:
        raise ValueError('Refinement model and extractor versions differ')
    p, known = predict_content(bundle['base'], cache, url_frame)
    keys = signature_keys(cache['records'], cache['hashes'])
    p, matched = apply_evidence(bundle['memory'], keys, p, bundle['refinement_config'])
    # Exact training URLs keep their training labels even when weaker signatures disagree.
    for i,url in enumerate(cache['urls']):
        if url in bundle['base']['known_urls']:
            p[i] = bundle['base']['known_urls'][url]
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid refined probabilities')
    return p, {'refinement_match':matched,'known_url':known['known_url']}
