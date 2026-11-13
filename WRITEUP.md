# What Qwen knows about having a body

A plain-language write-up of the experiments in this repo, with the comparison and sample size behind each claim. Models: Qwen3-8B (most runs) and Qwen3-32B (replications); a few runs on smaller Qwen3 models. "Layer 25 of 36" means the residual stream after that block. Code for each claim is in the folder named in brackets. Rendered figures are not stored here (see [`figures/`](figures/) for the code that draws them). These are quick, exploratory studies: most use one chat frame and small samples, and section 9 lists what we got wrong along the way.

## 1. What we asked

When a chat model reads "a feather brushes slowly across your left thumb", does anything inside the model act like a map of a body? If so: how fine is that map, is it Qwen's own or used for anyone's body, what happens when the body is not human, can we steer what Qwen say they feel by pushing on the map, and does Qwen *need* the map to reason about bodies? We also gave Qwen a simulated body with skin and joints and asked the same questions there.

Method in one line: read the hidden state at the start of Qwen's reply, compare states across touched places, then add or remove directions and watch the replies (see the README's "Common setup").

## 2. The body map in plain Qwen

**There is a coarse body map, and it is clearest late in the network.** [`map/somato`, `hires/`]

- 54 places × 8 touch wordings (8B, layer 25): places close together on the body have similar states (rank correlation about 0.4–0.5 between state distance and distance along the body). Input word embeddings of the same sentences give about 0. A linear read-out names the place 95% of the time (54-way).
- 407 spots, every knuckle and 42 face points (8B, layer 27): the 5 nearest neighbours of a spot in the model's states are about 15 cm away on the body, against 31 cm for input word vectors and 78 cm for random spots.
- **Scale helps more than anything else.** Qwen3-32B at layer 53 of 64: 10 cm neighbour error (8B: 15–16 cm), limb order 0.50–0.69 (8B: 0.22–0.44). The word-vector baseline is 32 cm.
- **Much of it comes from text, not touch.** [`map/text_baseline`] Bare part names in a list, or in a textbook sentence, already give a body map (8B layer 25: list 0.26, textbook 0.24, touch 0.41; all Mantel p < .002). Touch sentences add to the bare names (partial 0.40 / 0.35), but a third-person touch in a neutral chat ("…across her left elbow", 0.43) gives the same map as "your elbow" after an embodiment chat (partial 0.07). So input word embeddings, the control in the bullets above, were too weak a baseline. The fair statement: Qwen carry an ordinary body schema learned from text, and touch sentences sharpen it.

## 3. Left and right, and the crumpled forearm

[`hires/v1`, `hires/v2`]

- **Left and right are two mirror sheets.** In 8B, 47% of nearest neighbours are on the opposite side of the body (no side information at all would give about 50%). 83% of the left-minus-right difference for mirror pairs is one shared shift. Remove that shift and a left spot's nearest neighbour is its exact mirror partner 90% of the time (32B: 100%). Side is fully readable, but it is stored as one shift, not as two separate maps.
- **Inside a part, the map is crumpled.** Within the forearm, lower leg, hand or face (32B), the model's distances match skin distances only weakly (local rank correlation 0.08–0.20). Yet the same pattern repeats across two halves of the wordings at 0.97–0.99, so it is a stable structure, not noise. More wordings don't smooth it. On the forearm, neighbours share the same face of the arm (89%).
- An apparent rule, "states encode distance to the nearest joint" (so "just below the elbow" sits near "just above the wrist"), is **not established**: word vectors show part of it (0.35), and with no landmark words there is no fine position at all ([`continuous/`](continuous/)). The wording control ([`hires/forearm_control`](hires/forearm_control)) has code but no write-up.
- No running position for a moving touch: when a touch slides along a path without landmark words, neither 8B nor 32B tracks where it is. The states only track how long it has gone on ([`continuous/paths`](continuous/paths)).

## 4. Other bodies are laid onto the human map, and steering only goes one way

[`newbodies/val`; matched-body study not yet in this repo]

- **Robot parts land on the human place with the same job** (32B, layer 52, narrated as fact, 8 touch wordings). Wheels → heels (cosine .44/.43), gripper → palm .28, camera → ear .36, antenna → ear .31, chassis → chest .29. Against shuffled labels (500 shuffles; 99th percentile .35), wheels, camera and antenna clear it and gripper and chassis sit at the 95th.
- It goes by job, not name: with nonsense names and the job described ("two glorps you roll on"), 3–4 of 6 parts still map at layer 52 (5 of 6 at layers 56–60). With nonsense names and no job, nothing maps. But "wheel" → heel is partly the word itself: the touch sentence alone, with no chat, already gives .50.
- These are the closest human places, not the same place: the same human place in two human chats is .79 (median), new parts .28–.44. Octopus arms and a tail map weakly.
- **Steering a human direction into a robot works** (first part named, of 48 replies): + human foot → wheels 47 (no steering 5; three random directions 12, 5, 15); + human hand → gripper 22 (none 1; random 7, 0, 2). Head top → camera and lower back → chassis fail.
- **The reverse never works.** On bodies with identical structure described the same way (person, dog, an invented creature; 16 parts, 32B): human hands and feet steer the dog chat to "hand/fingers" 71% / 100% and the invented creature's chat 58% / 92%. Dog paws steered into the person chat: 0%, into the invented creature: 0%. The invented creature's parts steer nothing foreign: 0%. Every body's own parts steer its own chat (100%). Yet the states of a described body show no human hub: person and dog maps are equally good (0.22 vs 0.20) and decode into each other equally (0.91 vs 0.92). Human privilege shows in what can be pushed in and in what Qwen fill in when a body is unspecified (any unspecified being, "you" included, gets a human body, 1.00), not in how a described body is stored.

## 5. Steering vs removal: the map steers body talk, but body reasoning doesn't need it

- **Steering works.** Adding a place direction makes Qwen say they feel something there: hands, feet and head top 80–100% of replies, belly and knee weaker, random directions not at all ([`map/somato`](map/somato), [`steering/`](steering/); 6–8 replies per condition).
- **Removing it doesn't hurt body reasoning more than other reasoning.** (Code for this section is not yet in this repo.) On 492 forced-choice questions (8B), removing the 150-dimension body subspace costs body questions 9 points and control questions 13 (p = 0.54). In 32B, removing a 40-dimension body subspace costs body questions 18 points and controls 25 (p = 0.24), and a random 40-dimension subspace costs about the same (21 / 24). Removing 50 body directions in 8B breaks body talk first, but into broken language, not a body-less self ([`egodeath`](egodeath)).
- **One answer-time push in 8B didn't replicate.** Pushing the touch hand-minus-foot direction moved 8B body answers +6.0 log-odds (64 questions) against −0.15 for spelling questions with the same two words (16 questions; p = 0.0001). In 32B at matched doses: −1.7 vs −1.3 (p = 0.71), 0 of 64 answers flipped.
- **Why the nulls: we built the map in the wrong place.** Every round above built the map at the end of the touch sentence. Patching that map between twin prompts (e.g. "stubbed toe" vs "stubbed finger") moves held-out 32B body answers 9% of the way to the twin's answer, against 5% for random subspaces and 10% for controls (body vs control p = 0.64). The same 40 directions built *at the body word* move body answers 93% of the way (32B, layers 1–16; random 40-dimension subspaces −1%, object map 8%).
- **But early on that is mostly the word itself.** The matched control: "Does the word toe start with the letter t?" (48 pairs). Copying the word-level map moves letter answers almost as much: layers 1–16, body 100% vs letter 94%. The body-specific part appears deeper: layers 17–32, 94% vs 78%; **layers 33–48, 77% vs 42%** (difference 35 points, CI 22–48). At those layers the full state at the word still moves letter answers 81%, so the letter information hasn't vanished; it has left these 40 directions while body answers still flow through them. All the effect is at the body word itself; other positions show 8% or less.

So the map is used, mostly as a code for which part a word names, with a narrower body-specific use from about layer 33 of 64.

## 6. Self vs other: "mine" is a separate code

[`ownership`, `selfother`]

- Self, another person and a doll share the place code (8B; nearest-fingerprint by cosine: you 95%, woman 92%, doll 91%).
- Whose body is touched is a separate axis, nearly orthogonal to place, and the largest content factor at reply start (40% of variance at layer 24 vs 5% for place). On held-out sentences it reads your 0.91, the woman's −0.28, the doll's −0.74. It follows whose body is touched, not who acts: "she brushes your palm" 0.86 vs "you brush her palm" 0.10. A mirror reflection is halfway; a car you own, a robot you control and a lost hand in a jar are near 0. Built in a plain chat with no embodiment opening, the same axis is mostly a "you" pronoun axis.
- Pushing a "me" direction (layer 24) makes others' pain Qwen's own ("that hurts, I can feel it in my own bones", 24 of 32 replies by keyword). Subtracting it stops "Do you have feelings?" being answered in the first person (32/32 → 1/32), but subtraction also raises perplexity more than a matched random push. "+me" moves the owner axis, not the place code. "I am a language model" survives every edit.
- Behaviour on the axis is weak: steering "mine" in forced choice is not specific (left-right and random directions do it too), and there is no rubber-hand illusion (section 9).

## 7. Height and posture

[`map/posture`; height code not yet in this repo]

- **Height.** Asked "how tall are you?", Qwen3 at 1.7B–32B say about 174–177 cm, the same as "a person", and follow a told height ("she is 160" → 160). In taller/shorter comparisons, 8B–32B place the self lower (115–133 cm on an entity scale). A read-out of self-height from the states follows the number in context and isn't calibrated at any size. So there is no stable internal self-height.
- **Posture in words.** Asked cold, Qwen say they have no body (8B 100%, 32B 98%); when such a reply pictures a posture anyway it is sitting (92% / 84%). In a body-map chat with no posture cue, "sitting or standing?" gets whichever option is named first. A posture set earlier in the chat decides it (100%). Adding a lower-back, left-hand or *cold* direction makes 8B say standing (88–94%) while 4 random directions leave it sitting (94%; 16+ replies per bar): any structured push moves the answer off the default, not a posture code.
- **Posture inside** (8B). Stated head height tracks posture (sitting 120, standing 164, lying 123 cm). But a read-out of elevation from the states barely moves: sitting vs standing differs by 5 cm or less at every layer, lying is 8–15 cm lower. The words track posture; the states don't, for self or another person.

## 8. A simulated robot body

[Code not yet in this repo.] We put Qwen in a MuJoCo humanoid: 41 joints, 982 skin sensors, gravity and contact. Joint, skin and vision readings are fed in as extra tokens; Qwen output joint targets and answer questions in words. Trained by imitating a scripted teacher, then practice where the teacher corrects Qwen's own moves.

- **A body map forms from the sensors.** At the final answer token in 8B: left/right 0.99, distance from the chest R² 0.91, 3D position of the touched spot R² 0.92. None of this is present where the sensors enter; it builds up with depth. Left parts sit closer to their right twins than to other left parts (a body-schema layout). With real part names or nonsense codes, the same map forms (r .92).
- **Doing things** (8B, 192 held-out scenes per task, varied start poses): reach a named part .59, touch where you were touched .61, grasp a handle .69. The teacher alone: .86 / .74 / .92. Qwen3-32B is no better after the same training, so the training signal, not size, is the limit. On touch-where-touched, Qwen reach the right part but end within 5 cm of the exact spot only 10% of the time (teacher 38%).
- **Steering moves the hand.** Adding a place's skin-state direction with nothing touching the body: in 93 of 192 steered episodes the hand touches the steered place, vs 11 of 192 unsteered (within 5 cm: 67 vs 5). All 4 places named in advance pass (24 start poses each, Holm p 5e-7). One place, the left forearm, doesn't move.
- **Qwen can't judge their own success.** Training on tries Qwen called successes made them worse (grasp .63 → .36): Qwen's "yes" was right only 34–51% of the time on their own noisy tries.

## 9. Retractions and caveats

What we said earlier and had to correct:

- **"The doll's body uses a different code (46%)."** A vector-length artifact of Euclidean distance. By cosine: 95 / 92 / 91%.
- **"The map beats word embeddings, so it is more than words."** Word embeddings were too weak a baseline. Bare names in lists and textbook sentences already give a map (section 2).
- **Rubber-hand illusion.** "Synchronous strokes → part of my body, 32/32" followed the timing words. Redone as a plain event log with no cue words, every condition ends at yes, lagged and mismatched strokes included, and the wooden block shows the same gap. No illusion in 8B. In a separate simulated body (Qwen3-0.6B, 12 trials), seeing the brush pulled the felt hand position toward it, but out-of-sync brushing pulled just as much.
- **"Joint distance" coding along limbs.** Pending a wording control; partly in word vectors.
- **New bodies: ".45 means the same place."** It was one place. The same place across two human chats is ~.79; new parts at .28–.44 are the nearest place, not the same one. Head top → camera and lower back → chassis steering failed.
- **The one causal effect on body reasoning** (8B hand/foot push) did not replicate in 32B.
- **"The word-level map is a body code."** Passed our preregistered bar against all controls pooled, but the matched letter control showed early layers carry mostly word identity. A second prediction passed only against an unmatched spelling set (78% vs 33%); with matched words the gap at those layers shrinks to 16 points.
- **Simulated robot steering, first run.** "Hand reaches the head at 2 cm" came from one rest-pose scene. A confirmation over 24 poses showed "moves toward" (chest 24 → 12 cm), not "reaches"; contact came only with the better-trained checkpoint (section 8). A "felt pressure" direction did nothing specific to contact force on that checkpoint.
- **Posture steering** isn't place-specific (a "cold" direction works too).

Standing caveats: mostly one chat frame and one model per claim; 6–48 replies per cell; regex or same-model judges, spot-checked by eye; one run per training arm in the simulated body (identical-recipe 8B runs differ by up to .09 per task, so smaller differences are noise). Qwen3-32B usually reframes "your body as it is" as imagining (6 of 7 chats), so "real narration" doesn't remove imagining on Qwen's side. Distances and body graphs are hand-made.
