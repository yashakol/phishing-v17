import joblib
import numpy as np

from phishing_url.hashed_text import HashedTextModel


def test_hash_idf_is_fitted_on_train_and_survives_roundtrip(tmp_path):
    records=[{'url':f'https://example.test/{i}','title':word,'text':word*4,'markup':'tag_p'}
             for i,word in enumerate(['banklogin','banklogin','trustedshop','trustedshop'])]
    model=HashedTextModel(c=4,include_char=False,include_markup=False,include_url=False,
                          verbose=False).fit(records,np.array([1,1,0,0]))
    idf=model.tfidf_['word'].idf_.copy()
    p=model.predict_proba(records)
    assert p[:2].mean()>p[2:].mean()
    model.predict_proba([{'url':'https://unseen.test','title':'unseen-token','text':'other','markup':''}])
    np.testing.assert_array_equal(idf,model.tfidf_['word'].idf_)
    path=tmp_path/'hash.joblib'
    joblib.dump(model,path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(records),p)
