"""Historical temporal-cohort selection, extracted without changing its algorithm.

Supply licensed metadata explicitly. No data, model calls, or deployment code.
"""
import collections
import hashlib
SELECTION_SEED = 'egoown-supplement-20261113-v1'

def sha(b): return hashlib.sha256(b).hexdigest()

def group(r): return r['group_id']

def rank(tier, r): return sha((SELECTION_SEED+'|'+tier+'|'+r['id']).encode())

def select_cohort(final, old, panel, demos):
    excluded = {group(r) for r in panel+demos}
    eligible = [r for r in final if group(r) not in excluded and r['target_name']==old[r['id']]['target_name']]
    chosen, used, audit = [], set(excluded), {}
    for tier in ('T3', 'T4'):
        candidates = [r for r in eligible if r['taxonomy']==tier and group(r) not in used]
        one_per_group = {}
        for r in sorted(candidates, key=lambda r: rank(tier, r)):
            one_per_group.setdefault(group(r), r)
        by_source = collections.defaultdict(list)
        for r in one_per_group.values(): by_source[r['source']].append(r)
        total = len(one_per_group)
        assert total >= 100
        quotas = {s: 100*len(v)//total for s,v in by_source.items()}
        for s in sorted(by_source, key=lambda s: (-(100*len(by_source[s])%total), s))[:100-sum(quotas.values())]: quotas[s]+=1
        selected=[]
        for s in sorted(by_source): selected += sorted(by_source[s], key=lambda r: rank(tier,r))[:quotas[s]]
        assert len(selected)==100
        audit[tier] = dict(eligible_items=len(candidates),eligible_groups=total,source_group_counts={s:len(v)for s,v in by_source.items()},selected_source_counts=quotas,selected_label_counts=dict(collections.Counter(r['label']for r in selected)))
        chosen += selected; used.update(group(r)for r in selected)
    assert len({group(r)for r in chosen})==200
    return sorted(chosen,key=lambda r:r['id']), audit, sorted(excluded)
