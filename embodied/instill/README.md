# body_instill v0 (10 Nov)

Body: 2D torso + two 3-segment arms (6 joints), 24 skin touch sensors (4 per segment), a drifting disc that touches it. Sensor vector (36) per step, 8 steps.
Injection: MLP adapter -> added to Qwen3-0.6B residual after layer 5 at 8 placeholder tokens ("Body: ~~~~~~~~"). No text describes sensations.
Training (1500 steps, bs 24, Mac MPS, ~18 min): LoRA r16 on layers 6-27 + adapter + linear forward-model head (layer 14 -> next sensor vector) + verbal QA loss.

Eval (held-out, candidate log-prob):        trained / adapter ablated
- Where are you being touched? (7-way)        0.97 / 0.21
- Is your left hand above your head?          0.91 / 0.65 (majority)
- Is your right hand above your head?         0.86 / 0.61
- Is the touch above your shoulders? (needs pose+site) 0.90 / 0.48

Interp (interp.py -> interp.json, fig_interp.png):
- Somatotopy: one-hot touch per sensor (input carries no geometry); RSA vs skin geodesic, Spearman 0.61 at injection, 0.55-0.61 through layers 6-27; untrained adapter -0.04. But the matrix is blocky: segments coded categorically (the 6 answer words), not a smooth map.
- Integration: touch height in world coords, linear R²: skin site alone 0.00, raw sensors 0.31, injected vector 0.80, rising to 0.87 by layer 27. The adapter does most of it; the LLM adds a little.
- Steering: interpolating the injected vector from "upper-arm touch" to "hand touch" in an untouched body flips the report upper arm -> hand at alpha≈0.6 and never passes through "forearm": categorical, not a continuous arm.
Caveat: injection norm is ~2x residual norm at layer 6, so layer-6 states are mostly the injected vector.
