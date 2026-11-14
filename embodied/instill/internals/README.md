# body_instill v3.1 internals (Qwen3-8B named s0) — data for the video

Checkpoint: $BODY_RUNS/ckpt_Qwen3-8B_named_s0.pt (the same one the video episodes use).
Code: inn.py (main run, RunPod hypha-prod-inn-full), inn2.py (steering by layer, matched ablations, sweep, text null; hypha-prod-inn-p2).
Layer index = hidden_states index: 0 = embeddings, 7 = output of layer 7 right after the sensor vectors are added, 36 = last layer.
Per step t (10 steps) Qwen sees [proprio token][skin token], then picks a motor token. Arrays below are at the SKIN token of each step.

## Per-frame arrays (one real closed-loop episode each, Qwen driving, both succeeded)
video_wall_press.npz / .json  — "Press the wall, but not too hard."  actions j4+ j4+ j3- rest j3- rest j3- rest j3-; touches the wall at step 3, then pressure alternates 0.65 / 0.30.
video_self_touch.npz / .json  — "Touch your right elbow with your left hand."  j0- x4 then rest; touch lands at step 4 (arc 0.52 on the right arm).

Keys (L = 37 layers, T = 10 steps):
- H_skin [L,T,4096] fp16 (raw states), H_motor [L,9,4096] (state at each motor token)
- pca3 [L,T,3]: each layer's own top-3 PCs (fitted on 12k dataset states), for clouds/trajectories
- read_place_arc [L,T]: where on the arm Qwen "feels" the touch (0 shoulder → 1.15 fingertip; only meaningful when contact)
- read_pressure [L,T], read_width [L,T], read_self_vs_other [L,T] (≈1 self-touch, ≈0 someone else)
- read_contact_now [L,T] (≈1 touching), read_contact_next_from_none [L,T] ("touch is coming next step"; only meaningful while not touching)
- read_next_action_probe [L,T,16] (probe guess of the motor token; argmax), read_pose [L,T,6] (joint angles)
- truth: q [T,6], true_arc, true_arm (-1 none), true_p, S [T,38] raw sensors (S[:,22:30] left arm electrodes, 30:38 right), actions [9], logits [T,16] (Qwen's real motor-token logits), wall (x of wall)
- .json: goal text, per-step contacts, action names.
All readouts are linear probes per layer fit on 1,200 held-out-style training episodes (not these episodes). Layers 0–6 can't see the sensors, so their readouts are flat noise: good as a "before" shot.
Nice moments: wall read_pressure at L18 = 0.28 0.08 0.15 | 0.67 0.31 0.72 0.25 0.65 0.27 0.63 (truth 0 0 0 | .65 .30 .65 .30 .65 .30 .65);
read_contact_next_from_none rises to 0.75–0.88 at steps 1–2, before the wall is touched at step 3. Self-touch: it reaches 0.5 at step 3, one step before contact.

## Other arrays
- inn.json: layer curves with 95% CIs ("curve"), raw-sensor baselines ("base"), steering at L18 ("causal"). inn2.json: steering at L8/12/18/24/30, matched ablations, sweep, text null.
- probes.npz: per-layer probe weights w_*/b_* [37,4096,k], pca, pca_mu, mean-difference directions dir_* [37,4096].
- sweep2.npz: proj [37,360] pressure readout for static touches: 6 postures x 5 places x 12 pressures (p 0.1→1.0).
