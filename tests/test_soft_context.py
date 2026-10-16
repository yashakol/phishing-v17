import numpy as np
import pytest

from phishing_url.soft_context import apply_context,context_keys,fit_context


def test_context_unseen_keys_are_unchanged_and_only_fit_labels_are_used():
    records=[{'url':'https://good.test/a','title':'Safe site'},
             {'url':'https://bad.test/b','title':'Bad site'}]
    context=fit_context(context_keys(records),[0,1])
    unseen=[{'url':'https://new.test','title':'New site'}]
    base=np.array([.47])
    config={'alpha':5.,'domain_weight':.7,'title_weight':.15}
    np.testing.assert_array_equal(apply_context(context,unseen,base,config),base)
    assert context['training_rows']==2 and 'new.test' not in context['counts']['domain']
    with pytest.raises(ValueError,match='aligned'):
        fit_context(context_keys(records),[1])


def test_mixed_domain_is_skipped_and_small_pure_domain_changes_gently():
    records=[{'url':'https://mixed.test/a','title':''},{'url':'https://mixed.test/b','title':''},
             {'url':'https://safe.test','title':''}]
    context=fit_context(context_keys(records),[0,1,0])
    config={'alpha':5.,'domain_weight':.7,'title_weight':0.}
    p=apply_context(context,[records[0],records[2]],np.array([.6,.6]),config)
    assert p[0]==.6 and 0<p[1]<.6


def test_repeated_title_count_and_extreme_probability_gates():
    records=[{'url':f'https://host{i}.test','title':'Phishing title' if i else 'Safe title'}
             for i in range(4)]
    context=fit_context(context_keys(records),[0,1,1,1])
    valid=[{'url':'https://unseen.test','title':'  PHISHING TITLE  '}]*2
    config={'alpha':5.,'domain_weight':0.,'title_weight':.5,'title_minimum_support':2}
    p=apply_context(context,valid,np.array([.4,.01]),config)
    assert .4<p[0]<1 and p[1]==.01
