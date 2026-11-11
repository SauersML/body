# state (for my own continuity)
- 8B raw: out8b_{none,short,long,short_person,short_person_rp(running),long_person,long_person_rp (rerun via run8b3.sh)}; reduced red8b_*.npz; metrics m8b_*.json (ana_run.py); cmp.py prints.
- 32B reduced: red32_{none,short,long,long_person}; pod2 (body32b) still to deliver long_person_rp, short_person; pod3 (body32c) raw reply L50,53,57 short prefix -> $BODY_RUNS/body32c/raw_short
- ana3.py (reliable subspace/crossnobis/isomap/diffusion/local patches): ana3_s27.json done (8B short L27): all methods 14-16cm v1set; held-out reliable dims 153; local rho 0.17-0.26.
- key: 8B short L27 PCA = v1 (15.6cm, chain .22 all framings/.44 v1like, wf rank 37/24). Long 'imagine' prefix hurts nn5 (25) but helps probes (20, alt 11). Whitened concat chain .77.
- 32B short L53: 10.0cm, chain .50, partner 1.0, wf 15, alt 13. 32B long_person L53: 9.7cm, chain .30, wf 22, alt 6.
- lexical: 31.6cm, chain -.05, wf 67; shaft order: model states encode joint proximity (U), along-shaft rho<0.
- framings agree .8-.96; selection no help. medical/injury/cold families best nn5.
