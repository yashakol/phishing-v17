"""Independent training counts and inference replay; no new model selection."""
import run as screen
import gzip, json, hashlib
from collections import defaultdict, Counter
import numpy as np
from origin import OriginAffinity, training_digest, index_digest, row_keys

OUT = screen.OUT


def read(name):
    return json.loads((OUT/name).read_text())


def zipped(name):
    with gzip.open(OUT/name,"rt",encoding="utf-8") as f: return json.load(f)


def main():
    frozen, protocol = read("frozen_run.json"), read("protocol.json")
    assert frozen["protocol"] == protocol and len(protocol["configurations"]) == 3
    for path, value in frozen["sources"].items(): assert screen.sha(screen.ROOT/path) == value
    provenance = read("provenance.json")
    assert screen.sha(OUT/"training_rows.json.gz") == provenance["training_rows_sha256"]
    assert screen.sha(OUT/"validation_keys.json.gz") == provenance["validation_keys_sha256"]
    training, validation = zipped("training_rows.json.gz"), zipped("validation_keys.json.gz")
    rows, labels = [r["row"] for r in training], [r["label"] for r in training]
    with np.load(OUT/"inner_split.npz",allow_pickle=False) as a:
        tr,va=a["inner_train"],a["inner_validation"]
    np.testing.assert_array_equal([r["index"] for r in rows],tr)
    np.testing.assert_array_equal([r["index"] for r in validation],va)
    assert len(tr)==34663 and len(va)==6118 and set(tr).isdisjoint(va)
    assert len({r["url"] for r in rows})==len(rows)
    assert {r["url"] for r in rows}.isdisjoint(r["url"] for r in validation)
    assert training_digest(rows,labels)==provenance["original_c16_guard"]["ordered_training_sha256"]
    assert index_digest(tr)==provenance["original_c16_guard"]["ordered_training_indices_sha256"]
    for r in rows+validation: assert row_keys(r)==r["keys"]
    # Reconstruction uses only guarded training labels, never validation truth.
    title_domains=defaultdict(lambda:defaultdict(Counter))
    keyed=defaultdict(list)
    by_index={r["index"]:(r,label) for r,label in zip(rows,labels,strict=True)}
    for row,label in zip(rows,labels,strict=True):
        if row["keys"]: title_domains[row["title"]][row["domain"]][label]+=1
        for key in row["keys"].values(): keyed[key].append(row["index"])
    ledger=zipped("training_group_ledger.json.gz")
    assert set(ledger)==set(keyed)
    reasons=Counter()
    policy=protocol["shared_policy"]
    for key,g in ledger.items():
        assert g["training_indices"]==keyed[key]
        counts=defaultdict(lambda:[0,0])
        for rid in keyed[key]:
            row,label=by_index[rid]
            counts[row["domain"]][label]+=1
            assert row["title"]==g["title"]
        assert dict(counts)==g["domain_counts"]
        benign=sum(n[0] for n in counts.values()); phishing=sum(n[1] for n in counts.values())
        sources=sorted(d for d,n in counts.items() if n[0]); foreign=sorted(d for d,n in counts.items() if n[1])
        assert g["benign_urls"]==benign and g["phishing_urls"]==phishing
        assert g["source_domains"]==sources and g["phishing_domains"]==foreign
        if any(min(n)>0 for n in counts.values()): reason="mixed_key_domain"
        elif any(title_domains[g["title"]][d][0] and title_domains[g["title"]][d][1] for d in counts): reason="mixed_full_title_domain"
        elif benign<policy["minimum_benign_urls"]: reason="insufficient_benign_urls"
        elif phishing<policy["minimum_phishing_urls"]: reason="insufficient_phishing_urls"
        elif len(sources)>policy["maximum_source_domains"]: reason="too_many_source_domains"
        elif any(counts[d][0]<policy["minimum_urls_per_source"] for d in sources): reason="weak_source_domain"
        elif len(foreign)<policy["minimum_phishing_domains"]: reason="insufficient_phishing_domains"
        elif g["origin"] and g["origin"] not in sources: reason="untrusted_declared_origin"
        else: reason="accepted"
        assert reason==g["status"]
        reasons[reason]+=1
        if reason=="accepted":
            assert set(sources).isdisjoint(foreign)
            assert g["source_posterior"]==policy["alpha"]/(benign+2*policy["alpha"])
            assert g["foreign_posterior"]==(phishing+policy["alpha"])/(phishing+2*policy["alpha"])
            assert g["weight"]==min(policy["maximum_weight"],min(benign,phishing)/(min(benign,phishing)+policy["weight_pseudocount"]))
    learned_benign={t:sorted(d for d,c in counts.items() if c[0]) for t,counts in title_domains.items()}
    known_domains=sorted({r["domain"] for r in rows if r["domain"]})
    prescore=read("fits_before_scoring.json")
    assert prescore["all_three_fits_complete"] and not prescore["validation_labels_passed_to_fit"]
    results,selection=read("results.json"),read("selection.json")
    baseline=results["baseline"]; threshold=baseline["threshold"]
    guard_rejections=0
    class NoLabels:
        def __array__(self,*args,**kwargs): raise AssertionError("Validation label access")
    for spec in protocol["configurations"]:
        name=spec["name"]
        model=read(name+"_model.json")
        assert screen.sha(OUT/(name+"_model.json"))==prescore["candidate_model_hashes"][name]
        assert model["scope"]==prescore["scope"]==provenance["scope"]
        assert model["title_benign_domains"]==learned_benign and model["known_training_domains"]==known_domains
        expected={k:g for k,g in ledger.items() if g["family"] in spec["families"] and g["status"]=="accepted"}
        assert model["evidence"]==expected
        # Real validation-row attack rejected before labels can be read.
        try: OriginAffinity(**model["config"]).fit([validation[0]],NoLabels(),tr,forbidden_indices=va)
        except ValueError as exc:
            assert "row guard" in str(exc); guard_rejections+=1
        else: raise AssertionError("Held-out row accepted")
        result=read(name+"_results.json")
        with np.load(OUT/(name+"_inner.npz"),allow_pickle=False) as a:
            np.testing.assert_array_equal(a["indices"],va)
            p,base,y=a["probabilities"],a["baseline"],a["labels"]
            independently_predicted=base.copy()
            limit=model["config"]["confidence_limit"]
            # Independent inference: train-fitted membership, posterior and weight only.
            for i,row in enumerate(validation):
                if base[i] in (0.,1.) or base[i]<1-limit or base[i]>limit: continue
                votes=[]; veto=False
                for family in model["config"]["families"]:
                    key=row["keys"].get(family)
                    if key not in expected: continue
                    g=expected[key]
                    if row["domain"] in g["source_domains"]: direction,q=0,g["source_posterior"]
                    elif row["domain"] in g["domain_counts"]: continue
                    elif row["domain"] in learned_benign.get(row["title"],[]): veto=True; continue
                    else: direction,q=1,g["foreign_posterior"]
                    votes.append((g["weight"]*abs(q-.5),key,direction,q,g["weight"]))
                if votes and not veto and len({v[2] for v in votes})==1:
                    _,_,_,q,w=min(votes)
                    independently_predicted[i]=(1-w)*base[i]+w*q
            np.testing.assert_array_equal(independently_predicted,p)
            for tag,cut in (("fixed",threshold),("optimum",result["optimum"]["threshold"])):
                pred=p>=cut
                tp=int(((y==1)&pred).sum()); fp=int(((y==0)&pred).sum()); fn=int(((y==1)&~pred).sum())
                assert result[tag]["errors"]==fp+fn and result[tag]["fp"]==fp and result[tag]["fn"]==fn
                assert abs(result[tag]["f1"]-2*tp/(2*tp+fp+fn))<1e-14
            changed=(p>=threshold)!=(base>=threshold)
            assert result["corrected"]==int((changed&((p>=threshold)==y)).sum())
            assert result["damaged"]==int((changed&((base>=threshold)==y)).sum())
            np.testing.assert_array_equal(p[a["conflict"]],base[a["conflict"]])
            assert result["gate_passed"]==bool(baseline["errors"]-result["fixed"]["errors"]>=2 and result["fixed"]["f1"]>=baseline["f1"])
    winner=min(results["candidates"],key=lambda r:(r["fixed"]["errors"],-r["fixed"]["f1"],r["order"]))
    assert selection["winner_result"]==winner and selection["gate_passed"]==winner["gate_passed"]
    best=read("best_model.json")
    assert best["guard"]["provenance_sha256"]==screen.sha(OUT/"provenance.json")
    assert best["model"]==read(selection["winner"]+"_model.json")
    reload=read("reload_check.json")
    assert reload["rows"]==6118 and reload["probability_bitwise_equal"] and reload["wrong_scope_rejected"]
    assert reload["model_sha256"]==screen.sha(OUT/"best_model.json")
    execution=read("execution.json")
    assert execution["original_sources_inputs_unchanged"] and execution["fitted_models_unchanged_after_scoring"]
    for denied in execution["denied_events"]:
        assert denied["event"]=="socket.__new__" and "_has_ipv6" in "".join(denied["stack"])
    assert all(t["num_threads"]<=2 for t in execution["threadpools"])
    assert execution["classifier_fits"]==0 and execution["configurations"]==3
    artifact_hashes={p.name:screen.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {"verification.json","verify.log","manifest.json"}}
    screen.write("manifest.json",artifact_hashes)
    verified=dict(status="passed",independent_training_keys=len(ledger),training_key_status_counts=dict(reasons),candidate_predictions_independently_reproduced=3,rows_per_candidate=6118,real_validation_row_fits_rejected_before_label_access=guard_rejections,all_frozen_models_unchanged_after_scoring=True,baseline_errors=baseline["errors"],winner=winner["name"],winner_errors=winner["fixed"]["errors"],gate_passed=winner["gate_passed"],no_dataset_reload=True,no_new_fit=True,no_new_configuration=True,verified_source="guarded saved training rows only",audit_socket_creation_blocked=len(execution["denied_events"]))
    screen.write("verification.json",verified)
    print(json.dumps(verified,sort_keys=True))


if __name__=="__main__":
    with screen.threadpool_limits(limits=2): main()
