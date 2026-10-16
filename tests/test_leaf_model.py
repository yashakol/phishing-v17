import joblib
import numpy as np
import pandas as pd
import pytest

from phishing_url.leaf_model import LeafModel


def test_unknown_categories_are_missing_and_schema_is_checked():
    train=pd.DataFrame({'cat_host':['a','b'],'length':[10,20]})
    model=LeafModel()
    model.prepare(train,fit=True)
    held=pd.DataFrame({'cat_host':['unseen'],'length':[12]})
    prepared=model.prepare(held)
    assert prepared.iloc[0,0]==-1
    assert model.categories_['cat_host']==['a','b']
    with pytest.raises(ValueError,match='order'):
        model.prepare(held[['length','cat_host']])


def test_actual_leaf_fit_predict_and_serialization(tmp_path):
    frame=pd.DataFrame({'cat_host':['a']*30+['b']*30,'length':np.arange(60)})
    labels=np.asarray([0]*30+[1]*30)
    model=LeafModel({'n_estimators':20,'num_leaves':7,'min_child_samples':2}).fit(frame,labels)
    p=model.predict_proba(frame)
    assert p[:30].mean()<p[30:].mean()
    path=tmp_path/'leaf.joblib'
    joblib.dump(model,path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(frame),p)
