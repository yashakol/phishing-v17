"""A complementary leaf-wise tree model with training-only category vocabularies."""

import numpy as np
import pandas as pd
import lightgbm as lgb

from .features import categorical_columns


class LeafModel:
    def __init__(self,config=None,seed=17021,threads=2):
        self.config={} if config is None else dict(config)
        self.seed=seed
        self.threads=threads

    def prepare(self,frame,fit=False):
        if fit:
            self.columns_=list(frame.columns)
            self.category_columns_=categorical_columns(frame)
            self.categories_={name:pd.unique(frame[name].astype(str)).tolist()
                              for name in self.category_columns_}
        if list(frame.columns)!=self.columns_:
            raise ValueError('Leaf model feature columns or order differ')
        result={}
        for n,name in enumerate(self.columns_):
            safe_name=f'f{n:03d}'
            if name in self.categories_:
                vocabulary={value:code for code,value in enumerate(self.categories_[name])}
                result[safe_name]=frame[name].astype(str).map(vocabulary).fillna(-1).to_numpy(dtype=np.int32)
            else:
                result[safe_name]=np.asarray(frame[name],dtype=np.float64)
        return pd.DataFrame(result)

    def fit(self,frame,y,evaluation=None):
        labels=np.asarray(y)
        if len(frame)!=len(labels) or not np.array_equal(np.unique(labels),[0,1]):
            raise ValueError('Leaf model requires aligned binary training labels')
        x=self.prepare(frame,fit=True)
        self.model_=lgb.LGBMClassifier(**self.config,random_state=self.seed,n_jobs=self.threads,
                                       objective='binary',verbosity=-1,deterministic=True,force_col_wise=True)
        categorical=[f'f{self.columns_.index(name):03d}' for name in self.category_columns_]
        kwargs={'categorical_feature':categorical}
        if evaluation is not None:
            valid,valid_y=evaluation
            kwargs.update(eval_set=[(self.prepare(valid),np.asarray(valid_y))],eval_metric='binary_logloss',
                          callbacks=[lgb.early_stopping(150,verbose=False)])
        self.model_.fit(x,labels,**kwargs)
        self.iterations_=int(self.model_.best_iteration_ or self.model_.n_estimators_)
        return self

    def predict_proba(self,frame):
        p=self.model_.predict_proba(self.prepare(frame))[:,1]
        if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
            raise ValueError('Invalid leaf model probabilities')
        return p
