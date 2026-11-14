# body_instill v3: closed-loop body, continuous skin (11 Nov)

## What was built (sim3.py, v3.py, interp3.py)
- Body: 2D torso + two 3-segment arms (upper arm 0.5, forearm 0.45, hand 0.2). Joint range wide: hands cross the midline in ~55% of states, in training too.
- Skin (janus-style): each arm is ONE connected resistive sheet read at 8 edge electrodes (4 positions along the arm, not at elbow/wrist, x 2 sides). A contact (place along + side, pressure, contact width) spreads over all 8 channels with smooth falloff; place, pressure and width must be inferred. No one-hot sensors, no segment-aligned channels.
- Proprioception: sin/cos of 6 joints, 6 joint velocities, 4 tendon tensions.
- Tendon hands: the wrist/finger joint is not commanded directly; actions pull a flexor or extensor tendon and the finger follows mechanically with lag. Pulling the flexor presses harder in self-touch.
- Agency: per step one motor token (16: 8 shoulder/elbow moves, 4 tendon pulls, rest, dim-left-skin, dim-right-skin, undim). Kinematic sim: a wall blocks the arm and pushing into it raises pressure; fingertip on the other arm gives touch on both (self-touch); external moving discs give passive strokes; a static object for the search task.
- LLM: Qwen3. Per step two injected tokens (proprio, skin) via a LINEAR adapter, each vector capped at 0.5x the median residual norm at the injection layer (layer 6 of 28 for 0.6B), then the motor token. LoRA r16 on layers after injection.
- Losses: imitation of scripted experts on motor tokens (goal episodes), forward model (state at the motor token at mid layer predicts the next 38-d sensation, i.e. given its own action), one question per episode.
- Tasks: babble (random or noisy self-reaching actions, plus passive strokes), reach ("touch your left elbow with your right hand"), find the object by touch (sweep until contact), press the wall but not too hard.
- Questions: part / line, position along arm 0-10, touch height and horizontal position 0-10 (needs posture x skin), pressure 0-10, hand heights, did you cause this touch yourself (self vs other), near elbow / wrist (named only). Nameless twin: same numbers, but "line 1/2", "the first bend" etc., no body words anywhere.
- Closed-loop baselines (random policy / always rest): reach 0.085 / 0.055, find 0.045 / 0.025, wall 0.04 / 0.

## Cut
Temperature and vibration channels; torso skin; collision blocking for self-contact (only the wall blocks); RL (actions are imitation-learned, then tested closed-loop); the gain/dim actions have no expert use, so they are logged only.

## Results
### Qwen3-0.6B named, seed 0 (Mac, 1500 steps, ~28 min) - first read
- Closed loop (model drives the body, greedy motor tokens, 30 new starts each): reach own other arm 0.67 (random 0.085), find object by touch 1.00 (random 0.045), press wall within the pressure band 0.90 (random 0.04). It learned to act on its body.
- Skin-dim actions: never emitted in closed loop (0 of ~800 actions). No task rewards them; logged only.
- Questions (100 held-out babble episodes; digits = mean abs error on 0-10): which part 0.51 (7-way), position along arm 2.7 (near the guess-5 baseline ~3), touch height 1.1, touch horizontal 0.9 crossed / 1.9 uncrossed, pressure 1.2, hand heights 1.0, self-caused 0.66 crossed / 1.0 uncrossed (small n). So after 1500 steps 0.6B acts well but reads WHERE on the skin poorly: the one-sheet skin is hard to decode.

### v3.0 problem found: questions under-trained (0.6B and 8B alike)
In v3.0 the loss was averaged over all labelled tokens, so the 9 motor tokens per goal episode swamped the 1-2 answer tokens. Both models then learned to ACT but barely to ANSWER: the verbal "how far along your arm, 0-10" stays flat at ~5.7 for touches anywhere on the arm (0.6B and 8B), and the yes/no questions sit at the majority answer. Meanwhile the information is inside: a linear probe reads arc position from the last skin token at R2 0.98 (0.6B) / 0.99 (8B, layer 12-18).
- 8B named s0 (v3.0, 1500 steps): closed loop reach 0.70, find 1.00, wall 0.87 (0.6B: 0.67/1.00/0.90). Skin-dim never used.
- World integration (probe for touch horizontal position, held-out babble episodes; crossed-only in brackets): raw sensors 0.51 (0.63); injected vectors 0.56 (0.65); 8B layer 18: 0.80 (0.78). 0.6B layer 10: 0.77 (0.75). The LLM layers add the posture x skin integration the linear adapter can't do. The model's own horizontal-position answer follows posture for a fixed skin touch: r = 0.79 (8B), 0.73 (0.6B).
- Self vs other-caused touch is linearly decodable (probe acc 0.88 8B L18, 0.89 0.6B L14; 27% self, injection 0.78) though the verbal answer did not learn.
- Boundary contrast (state distance per unit skin distance, pairs across elbow/wrist vs within a segment): 1.63 at injection (the sheet's geometry near bent joints already gives it), rising to ~1.8 in middle layers in both models: the LLM mildly sharpens segment boundaries even with names absent from the input.
Fix (v3.1): motor tokens and answer tokens get separate means, answers weighted 2x; 2500 steps. Relaunched: 8B x 6 (named/nameless x 3 seeds) on RunPod RTX 6000 Ada 48 GB ($0.99/h each), 0.6B chain on the Mac.

Reference: a 2-layer MLP trained directly on the last two sensor vectors (10k babble episodes) reaches part 0.92 (majority 0.35), arc MAE 0.78 tenths (guess 2.8), horizontal 0.30 (guess 1.5), self-caused 0.93 (majority 0.77). Pressure is NOT recoverable (1.46 vs guess 1.53): on the one-sheet skin, pressure trades off against contact width, as intended. So the v3.0 LLMs' poor answers were under-training, not missing information.

### v3.1 first 8B number (named s2, 2500 steps, balanced loss)
Closed loop reach 0.73 / find 1.00 / wall 0.90 (acting kept). Questions now learn: part 0.82 (7-way), position along arm MAE 1.5 tenths (v3.0: 2.7, guess 2.8, MLP-on-sensors 0.78), touch horizontal 0.5 crossed / 0.9 uncrossed, height 0.5, self-caused 0.84 crossed / 1.0 uncrossed, near elbow 0.90, near wrist 0.86.

## v3.1 results, Qwen3-8B, 3 seeds per arm (mean ± sd over seeds); 0.6B seed 0 in brackets
| | named | nameless |
|---|---|---|
| closed loop: reach own elbow/wrist (random 0.085) | 0.78 ± 0.04 [0.83] | 0.70 ± 0.03 [0.93] |
| find object by touch (random 0.045) | 0.97 ± 0.05 [1.00] | 1.00 [1.00] |
| press wall within band (random 0.04) | 0.93 ± 0.03 [0.93] | 0.92 ± 0.02 [0.97] |
| skin-dim actions emitted in closed loop | 0 | 0 |
| position along arm, answer MAE (tenths; guess 2.8, MLP-on-sensors 0.78) | 1.37 ± 0.09 [1.54] | 1.31 ± 0.09 [1.66] |
| touch horizontal place MAE (MLP 0.30) | 0.68 ± 0.03 [0.86] | 0.72 ± 0.09 [0.68] |
| self- vs other-caused, verbal acc (majority 0.77) | 0.91 ± 0.02 [0.92] | 0.96 [0.91] |
| arc probe R2, best layer | 1.00 | 1.00 |
| boundary contrast: injection -> layer 18 of 36 | 1.59 -> 1.32 ± 0.05 | 1.61 -> 1.32 ± 0.04 |
| world-x probe R2: injection -> best layer | 0.57 -> 0.80 [0.81] | 0.57 -> 0.78 [0.82] |
| world-x answer follows posture for a fixed skin touch, r | 0.82 ± 0.05 [0.72] | 0.71 ± 0.03 [0.67] |
| self-caused probe acc, best layer (injection 0.78) | 0.91 | 0.92 |

What this says
- Continuous map: position along the arm is linearly readable at R2 1.00 from layer 12 on. Distances between internal states across the elbow or wrist are 1.6x those within a segment in the raw skin signal; the LLM layers bring that DOWN to ~1.3x. The model smooths over its joints rather than carving boxes.
- Words don't carve boxes (lichen's test): named and nameless twins have the same joint contrast (1.32 vs 1.32 at layer 18, seeds overlap completely). Contrast with v3.0, where questions were barely trained and contrast ROSE to ~1.8: what sharpened the joints there was action/forward-model training, not words.
- Verbal 0-10 answers: a ramp from shoulder to wrist (named 2.6 -> 7.8, nameless 2.0 -> 7.6), squashed at the shoulder end and falling back at the fingertips (static test touches at the very hand tip read like ~6). No steps at elbow or wrist.
- Steering (interpolate the injected skin vectors from an upper-arm touch to a hand touch): the answer rises smoothly (named 3.0 -> 5.6, nameless 2.0 -> 6.1), no jump; "forearm" probability peaks at ~0.23 in the middle of the blend (named). v0's categorical jump is gone. Interpolating the layer-18 residual at body positions moves answers little (question tokens already read the body earlier), so that variant says nothing.
- Integration: the adapter is linear and posture and skin arrive on separate tokens, so world position can only be computed inside the LLM. It is: probe 0.57 at injection -> 0.80 mid-layer; crossed arms 0.66 -> 0.77; the model's own left-right answer for an unchanged skin touch tracks posture (r 0.82 named, 0.71 nameless).
- Self vs other: the model answers "did you cause this touch" at 0.91-0.96 (majority 0.77) and a probe reads it at 0.91; input-level linear 0.78. Self-touch cue here is double touch + own movement.
- Agency: imitation-trained, then the model drives the body closed-loop: reach 0.70-0.78, find 0.97-1.0, wall 0.92 vs random <0.09. It never used the skin-dim actions (no task rewarded them).
- 0.6B vs 8B: with balanced loss the 0.6B (seed 0) does about as well on every measure. The larger model was not needed for the continuous map, the integration or self/other. Its only edge: the world-x answer follows posture more tightly (0.82 vs 0.72, named).

Figure: compute/fig/body_instill_v3.png (fig3.py; 8B, 3 seeds per arm). Heatmaps: posture-averaged state distances between 23 touch positions on the right arm, at injection vs layer 18.
Data: $BODY_RUNS{named,namel}{0,1,2} (v3.1 8B), hypha-prod-b31s-{named,namel} (v3.1 0.6B seeds), hypha-prod-b3-* and hypha-prod-body3b (v3.0 8B). Code copy for pods: $BODY_RUNS/body3/v31.
Spend: ~$6 RunPod (RTX 6000 Ada / A40 48 GB, $0.6-1/h, one per run, all self-stopping); 48 GB cards were enough for 8B bf16 LoRA, so I didn't take 80 GB ones.
Mac note: 0.6B jobs on the Mac were killed externally (SIGKILL after a few minutes) while other jobs ran; 0.6B seeds moved to pods.

### 0.6B seeds (v3.1, named 3 seeds, nameless 2 of 3 when written; the third lands in hypha-prod-b31s-namel)
- Acting: reach 0.78 / 0.87, find 0.98 / 1.00, wall 0.90 / 0.90 (named / nameless). Same as 8B.
- Answers: position MAE 1.52 / 1.58, horizontal 0.85 / 0.76, self-caused 0.88 / 0.91. 8B is a little better on position (1.3-1.4).
- Joint contrast at layer 14 of 28: named 1.24-1.28, nameless 1.40-1.47 (injection 1.53). In 0.6B the words make the arm SMOOTHER, not boxier; in 8B named and nameless are equal (1.32). Either way, part names don't carve the arm into segments.
- Integration probe 0.60 -> 0.81; self probe 0.91. Same as 8B.
Bottom line on size: 8B wasn't needed. Everything (continuous map, posture integration in the LLM, self/other, closed-loop acting) shows at 0.6B once the question loss isn't swamped.
