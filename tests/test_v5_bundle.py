import numpy as np
import pytest

from phishing_url.v5_bundle import mix_probabilities


def test_v3_weights_recover_baseline():
    text=np.array([.1,.9])
    numeric=np.array([.4,.7])
    url=np.array([.3,.8])
    config={'text_weight':.5,'aux_url_weight':.2,'leaf_weight':0.,'cat_weight':0.}
    np.testing.assert_allclose(mix_probabilities(text,numeric,url,None,None,config),
                               .4*text+.4*numeric+.2*url,rtol=0,atol=1e-15)


def test_missing_models_and_negative_weights_are_rejected():
    p=np.array([.2,.8])
    config={'text_weight':.5,'aux_url_weight':.2,'leaf_weight':.3,'cat_weight':0.}
    with pytest.raises(ValueError,match='required'):
        mix_probabilities(p,p,p,None,None,config)
    config['cat_weight']=.8
    with pytest.raises(ValueError,match='weights'):
        mix_probabilities(p,p,p,p,p,config)
