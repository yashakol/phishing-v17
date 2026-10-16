"""Smoothed training-only evidence from domains and complete page titles."""

import unicodedata
from collections import Counter,defaultdict

import numpy as np
from scipy.special import expit,logit

from .features import group_for_url


def context_keys(records):
    return {'domain':[group_for_url(record['url']) for record in records],
            'title':[' '.join(unicodedata.normalize('NFKC',record.get('title','')).lower().split())
                     or None for record in records]}


def fit_context(keys,labels):
    y=np.asarray(labels,dtype=np.int8)
    if not np.array_equal(np.unique(y),[0,1]) or any(len(values)!=len(y) for values in keys.values()):
        raise ValueError('Context training requires aligned binary labels')
    counts={}
    for kind,values in keys.items():
        groups=defaultdict(Counter)
        for key,label in zip(values,y,strict=True):
            if key is not None:
                groups[key][int(label)]+=1
        counts[kind]={key:(group[0],group[1]) for key,group in groups.items()}
    return {'counts':counts,'prior':float(y.mean()),'training_rows':len(y)}


def lookup_counts(context,keys):
    return {kind:np.asarray([context['counts'][kind].get(key,(0,0)) for key in values],dtype=np.int32)
            for kind,values in keys.items()}


def apply_count_context(base,counts,prior,config):
    base=np.asarray(base,dtype=float)
    adjustment=np.zeros(len(base))
    for kind in ('domain','title'):
        weight=config.get(kind+'_weight',0.)
        if not weight:
            continue
        zero,one=counts[kind].T
        total=zero+one
        alpha=config['alpha']
        posterior=(one+alpha*prior)/(total+alpha)
        eligible=total>0
        if kind=='domain':
            # Large pure groups are already handled by frozen v3 rules. Mixed
            # domains may contain compromised legitimate sites, so skip them.
            eligible&=(total<5)&((zero==0)|(one==0))
        else:
            eligible&=total>=config.get('title_minimum_support',2)
        adjustment[eligible]+=weight*(logit(np.clip(posterior[eligible],1e-8,1-1e-8))-logit(prior))
    eligible=(base>=.05)&(base<=.95)&(adjustment!=0)
    result=base.copy()
    result[eligible]=expit(logit(base[eligible])+adjustment[eligible])
    return result


def apply_context(context,records,base,config):
    return apply_count_context(base,lookup_counts(context,context_keys(records)),context['prior'],config)
