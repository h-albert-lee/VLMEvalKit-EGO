"""Historical paired outcome analysis extracted without numerical changes.

The legacy result keys name the reference condition sparse. Intervals are
pointwise percentile intervals over paired items, not corrected for multiplicity.
"""
import collections
import random

def outcome(p):return 'ERROR'if p.get('error')else p.get('pred_label')or'UNPARSED'

def paired_summary(rows,base,other,bootstrap=False):
    pairs=[(outcome(base[r['id']])==r['label'],outcome(other[r['id']])==r['label'])for r in rows]
    diffs=[int(b)-int(a)for a,b in pairs];n=len(rows)
    result=dict(n=n,correct_to_wrong=sum(a and not b for a,b in pairs),wrong_to_correct=sum(not a and b for a,b in pairs),both_correct=sum(a and b for a,b in pairs),both_wrong=sum(not a and not b for a,b in pairs),accuracy_delta_other_minus_sparse=sum(diffs)/n if n else None)
    result['amb_transitions']=dict(collections.Counter(str(outcome(base[r['id']])=='AMBIGUOUS')+'->'+str(outcome(other[r['id']])=='AMBIGUOUS')for r in rows))
    result['accuracy_delta_sparse_minus_other']=-sum(diffs)/n if n else None
    if bootstrap and n:
        # Both cohorts have one ID per video/session. Resample paired IDs, not conditions.
        rng=random.Random(20261113);values=sorted(sum(rng.choices(diffs,k=n))/n for _ in range(5000))
        result['paired_bootstrap_95_percentile']=[values[124],values[4874]];result['bootstrap_replicates']=5000
        result['sparse_minus_other_95_percentile']=[-values[4874],-values[124]]
    return result
