"""Inference for a complementary tree ensemble selected on development OOF."""

import numpy as np

from .content_bundle import prediction_frame,predict_content
from .content_memory import apply_memory


class TextMixture:
    def __init__(self,old,new,weight):
        self.old=old
        self.new=new
        self.weight=weight

    def predict_proba(self,records):
        if self.weight==1:
            return self.new.predict_proba(records)
        return (1-self.weight)*self.old.predict_proba(records)+self.weight*self.new.predict_proba(records)


def mix_probabilities(text,numeric,url,leaf,cat,config,hashed=None):
    hash_weight=config.get('hash_weight',0.)
    if not 0<=hash_weight<=1:
        raise ValueError('Invalid hashed text weight')
    text=np.asarray(text)
    if hash_weight:
        if hashed is None:
            raise ValueError('Hashed text predictions are required')
        text=(1-hash_weight)*text+hash_weight*np.asarray(hashed)
    html=config['text_weight']*text+(1-config['text_weight'])*np.asarray(numeric)
    base=(1-config['aux_url_weight'])*html+config['aux_url_weight']*np.asarray(url)
    leaf_weight=config.get('leaf_weight',0.)
    cat_weight=config.get('cat_weight',0.)
    if leaf_weight<0 or cat_weight<0 or leaf_weight+cat_weight>1:
        raise ValueError('Invalid complementary model weights')
    result=(1-leaf_weight-cat_weight)*base
    if leaf_weight:
        if leaf is None:
            raise ValueError('Leaf model predictions are required')
        result+=leaf_weight*np.asarray(leaf)
    if cat_weight:
        if cat is None:
            raise ValueError('New CatBoost predictions are required')
        result+=cat_weight*np.asarray(cat)
    if not np.isfinite(result).all() or (result<0).any() or (result>1).any():
        raise ValueError('Invalid ensemble probabilities')
    return result


def predict_v5(bundle,cache,url_frame=None):
    config=bundle['config']
    base=dict(bundle['base'])
    base.update(text_weight=config['text_weight'],aux_url_weight=config['aux_url_weight'],
                memory_minimum_support=0,domain_minimum_support=0,memory={},known_urls={})
    hash_weight=config.get('hash_weight',0.)
    if hash_weight:
        base['text']=TextMixture(base['text'],bundle['hashed'],hash_weight)
    p,_=predict_content(base,cache,url_frame)
    frame=prediction_frame(cache,url_frame)
    if list(frame.columns)!=base['feature_columns']:
        raise ValueError('V5 feature schema differs')
    leaf_weight=config.get('leaf_weight',0.)
    cat_weight=config.get('cat_weight',0.)
    p*=1-leaf_weight-cat_weight
    if leaf_weight:
        p+=leaf_weight*bundle['leaf'].predict_proba(frame)
    if cat_weight:
        p+=cat_weight*bundle['cat'].predict_proba(frame)[:,1]
    old,hits=apply_memory(bundle['memory'],cache['records'],cache['hashes'],p)
    if 'context_config' in bundle:
        from .soft_context import apply_context
        p=apply_context(bundle['context'],cache['records'],p,bundle['context_config'])
        p[hits]=old[hits]
    else:
        p=old
    known_hits=np.zeros(len(p),dtype=bool)
    for i,url in enumerate(cache['urls']):
        if url in bundle['base']['known_urls']:
            p[i]=bundle['base']['known_urls'][url]
            known_hits[i]=True
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('Invalid V5 probabilities')
    return p,{'training_match':hits,'known_url':known_hits}
