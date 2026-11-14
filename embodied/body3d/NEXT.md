# body3d NEXT: learning from its own experience (12 Nov, hypha subagent body-next)

## Proposal (written before any run)
Question: can embodied Qwen3-8B (named ckpt) get better at reach / "touch where you were touched" / grasp than the scripted expert it imitated, by practising on its own, and does it work when the reward is *its own felt judgement* ("Did you succeed?") instead of the simulator's?
Setup: expert iteration from the 8B named checkpoint. Each round: 160 fresh scenes per task x 4 noisy attempts (smooth noise on arm/finger targets), keep the successful episodes, train 400 steps on 50% kept self-episodes (action + forward loss, no QA) + 50% the usual expert stream. 4 rounds. Three arms from the same start: SIM (keep if the simulator says success), SELF (keep if Qwen answers "yes" to "Did you succeed?" after its own episode), CTRL (same steps on expert stream only, so gains aren't just more training). Eval each round: 96 held-out scenes per task, no noise, same scenes for every arm and for the expert.
Meaning: SIM > CTRL and > expert on the same scenes (paired, McNemar) = it learned beyond its teacher from experience. SELF ~ SIM = its sense of its own success (from skin) is good enough to bootstrap on; SELF < SIM with low self-judge accuracy on own episodes = it fools itself (also measured: judge accuracy vs sim truth per round). Then body-map metrics (RSA vs skin geodesic, 3D decode) on the final checkpoints vs CTRL: sharper / same / worse.
Budget: one H200-class box, ~1.5 h, < $10.

## CHECKPOINT (launch ledger: read before launching anything)
CONFIRMED live: NONE (all sp2 pods ended and deleted by 05:40 PT; listed below for the record):
- hypha-prod-b3d-sp2-sim  pod 641tkatqe1cqjx
- hypha-prod-b3d-sp2-self pod qr6du0n5drzvn0
- hypha-prod-b3d-sp2-ctrl pod 9z50inl7tnzilo
  outputs: ~/mpd-data/runpod/hypha-prod-b3d-sp2-*/ (run.log, selfplay_*.json, final/). Runner logs: compute/body3d/launch_sp2_*.log
CONFIRMED ended/deleted: sp2-ctrl 9z50inl7tnzilo ($0.72), sp2-sim 641tkatqe1cqjx ($1.41), sp2-self qr6du0n5drzvn0 ($1.44), all status 0, outputs copied; sp-sim okss5gt33pzewk, sp-self 1iai21qpk6bymp, sp-ctrl tojinrklcu2oym (first attempt, stopped on purpose); probe pods l8wm1annhbnj4k, 5j9b085ka41tbi, 25sw2pf9laolr6, q4jzdvhxxwtvs5 deleted by rp-run. vast sp-* never created an instance ($0).
PENDING/ambiguous: none. Do NOT relaunch sp2-*; if the host restarts, the pods keep running and self-delete at cap; check `hypha-rp-run status` and the run.log files.

## Log
- 02:35 launched hypha-prod-b3d-sp-{sim,self,ctrl} on vast (H100/H200, cap $10 each, 2.5 h). Code: selfplay.py (c5 staged in mpd-data/scratch/hypha/body3d/c5).
- vast launch failed (offer cache stale, searches used up). 02:39 relaunched same 3 runs on RunPod via hypha-rp-run 48gb (A40-class, ~$0.6/h, 3 h cap, RP_SELF_STOP=1); outputs in ~/mpd-data/runpod/hypha-prod-b3d-sp-{sim,self,ctrl}.
- First attempt (sp-*) stopped after round 1 ($0.6): fresh AdamW at lr 3e-4/5e-5 with no warmup knocked reach from .50 to .25 even in CTRL; noise σ .1–.3 rad cut rollout success .49→.10–.16. Also: identical r0 evals differ a bit across pods (bf16 + chaotic sim). Finding already: Qwen's "Did you succeed?" is 100% on expert episodes but only 67–76% right on its own episodes.
- 03:03 relaunched sp2-{sim,self,ctrl}: lr 1e-4/2e-5 with 100-step warmup, σ {0,.03,.06,.1}, NEVAL 192.
- 03:13 sp2 command start. r0 (identical across the 3 pods now, 192 held-out scenes): reach .48 (expert .59), poke_reach .28 (.46), grasp .63 (.80). Qwen's "Did you succeed?" on its OWN episodes: 64–82% right (100% on expert episodes).
- Rollouts r1: even σ .03 rad noise cuts success a lot (reach .49→.32; grasp σ0 .61 → all-σ .28). Self-judge precision on own noisy episodes ~.39–.50 (SELF arm keeps ~half failures).
- CTRL r1: reach .47, poke_reach .31, grasp .71 (no harm from continued training at the low lr).
- ~03:55 sim/self in round 2 rollouts/training. ~25–30 min per round → selfplay done ~05:15–05:30 PT, then analyze3d rep (~5 min) writes final/analysis_*.json. Caps 06:53.

## RESUME (for a successor)
- Do NOT relaunch. Check: `grep -E "EVAL|ROLLOUT|DONE|Trace" ~/mpd-data/runpod/hypha-prod-b3d-sp2-{sim,self,ctrl}/run.log`; `hypha-rp-run status`.
- When all 3 show "ANALYSIS DONE" (after SELFPLAY DONE), pods end by themselves; confirm deleted in launch_sp2_*.log, update CHECKPOINT.
- Stats: `python3 compute/body3d/sp_stats.py hypha-prod-b3d-sp2` (Wilson CIs per round, McNemar final vs r0, vs expert, between arms; judge accuracy).
- Map: compare final/analysis_Qwen3-8B_named.json per arm (rsa final_token, xyz_r2, proxdist_r2, lr_decode) against CTRL and base (~/mpd-data/vast/hypha-prod-b3d-q8b-named/analysis_Qwen3-8B_named.json, NP=600 same).
- Videos: selfplay_vids_Qwen3-8B_named_{arm}.pkl has keys r0_<task>, r4_<task> (6 scenes each, same scenes): `cd compute/body3d && ~/.venv/bin/python render3d.py PKL next/sim` then ffmpeg hstack r0 vs r4 per scene → compute/body3d/next/.
- Figure plan (made by a figure agent, not me): one panel per task, success by round for SIM / SELF / CTRL with CIs, expert as a line; second panel judge accuracy on own episodes. Write results into notes/research/body-3d.md section "Learning from own experience".

## DONE (12 Nov 05:40)
Results in notes/research/body-3d.md 'Learning from own experience'; figure compute/fig/body3d_practice.png; videos compute/body3d/next/. Practice never beat the expert or plain extra expert training; Qwen-judged practice made grasp collapse; body map unchanged.
