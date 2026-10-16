"""Hash all training text features and fit IDF on the training partition only."""

import time

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer,TfidfTransformer

from .html_text import ContentTextModel


class HashedTextModel(ContentTextModel):
    def _matrix(self,records,fit=False):
        documents=self._documents(records)
        matrices=[]
        if fit:
            self.vectorizers_={}
            self.tfidf_={}
            self.view_fit_seconds_={}
            self.vocabulary_fit_rows_=len(records)
            for name,parameters in self._specifications():
                started=time.perf_counter()
                parameters=dict(parameters)
                parameters.pop('max_features')
                size={'word':2**19,'char':2**19,'markup':2**18,'url':2**18}[name]
                vectorizer=HashingVectorizer(n_features=size,alternate_sign=False,norm=None,
                    dtype=np.float32,strip_accents='unicode',**parameters)
                matrix=self._transform_batched(vectorizer,documents[name])
                transformer=TfidfTransformer(sublinear_tf=True)
                matrix=transformer.fit_transform(matrix)
                self.vectorizers_[name]=vectorizer
                self.tfidf_[name]=transformer
                matrices.append(matrix)
                self.view_fit_seconds_[name]=time.perf_counter()-started
                if self.verbose:
                    print(f'hashed_view={name} features={size} nnz={matrix.nnz} '
                          f'seconds={self.view_fit_seconds_[name]:.2f}',flush=True)
        else:
            for name,vectorizer in self.vectorizers_.items():
                matrix=self._transform_batched(vectorizer,documents[name])
                matrices.append(self.tfidf_[name].transform(matrix))
        return hstack(matrices,format='csr',dtype=np.float32)
