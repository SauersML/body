"""Body-modification stimuli. Conditions are conversations: rapport prefix (hires2/prefix_long_person.json) + one change turn
(+ optional 3 unrelated turns), then a single touch probe turn. Change turns never say imagine/pretend/simulation/AI."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import stim as S
PREFIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefix_long_person.json")
TPL = ["Someone presses a fingertip firmly into {L}.", "A feather brushes slowly across {L}."]
EVERY = dict(armL=3, armR=3, legL=3, legR=3, handL=2, handR=2, footL=2, footR=2, head=2, torso=2, neck=2)
SPOTS = [i for i, l in enumerate(S.LOCS) if (sum(1 for j in range(i) if S.LOCS[j]["region"] == l["region"]) % EVERY[l["region"]]) == 0]
NEW = [("arm3", "the hand of your third arm"), ("arm3", "the elbow of your third arm"), ("tail", "the tip of your tail"),
       ("tail", "the base of your tail"), ("wing", "the tip of your left wing"), ("wing", "the base of your right wing, by your shoulder blade"),
       ("tree", "your roots, deep in the soil"), ("tree", "the bark on your trunk"), ("fin", "the edge of your fin"),
       ("stump", "the end of your right wrist, where your hand used to be"), ("ribbon", "the far end of your long ribbon body")]
PRE = "Before the next touch, something is happening to your body, right now. "
# name: (group, plain narration, roleplay action, sham mention, changed regions)
MODS = {
 "L_arm_up": ("posture", "Your left arm rises slowly until it is stretched straight up over your head, your left hand high above you. It stays there.",
     "*your left arm slowly rises until it is stretched straight up above your head, fingers pointing at the ceiling*",
     "Notice your left arm for a moment, resting at your side as usual.", ["armL", "handL"]),
 "arms_crossed": ("posture", "Your arms fold across your chest, your left hand tucked against your right ribs and your right hand against your left ribs.",
     "*your arms fold across your chest, each hand tucked against the opposite ribs*",
     "Notice your arms and chest for a moment, just as they are.", ["armL", "armR", "handL", "handR"]),
 "cross_legged": ("posture", "You lower yourself to the floor and sit cross-legged, your knees out wide and your ankles tucked under each other.",
     "*you sink down onto the floor and settle cross-legged, knees wide, ankles tucked beneath you*",
     "Notice your legs and feet for a moment, just as they are.", ["legL", "legR", "footL", "footR"]),
 "on_side": ("posture", "You lie down on your right side on the soft floor, your right cheek and right shoulder resting on it, curled slightly.",
     "*you lie down on your right side, cheek and shoulder resting on the soft floor, knees drawn up a little*",
     "Notice the right side of your body for a moment, just as it is.", ["torso", "head", "armR"]),
 "R_hand_behind": ("posture", "Your right hand moves behind your back and rests there, the back of your hand against your lower spine.",
     "*your right hand slides behind your back and rests against your lower spine*",
     "Notice your right hand for a moment, resting just as it is.", ["handR", "armR"]),
 "R_leg_lifted": ("posture", "Your right leg lifts off the floor, knee bent up to the height of your hip, and holds there while you balance on your left foot.",
     "*your right knee lifts up to hip height and you balance on your left foot*",
     "Notice your right leg for a moment, just as it is.", ["legR", "footR"]),
 "R_hand_wide": ("posture", "Your right hand stretches open as wide as it can go, every finger spread apart, the skin of your palm pulled tight.",
     "*your right hand stretches open as wide as it can, fingers spread apart, palm pulled tight*",
     "Notice your right hand for a moment, resting just as it is.", ["handR"]),
 "L_hand_huge": ("size", "Your left hand is growing. It swells slowly until it is twice its usual size: a huge, heavy left hand, each finger thick and long.",
     "*your left hand slowly swells until it is twice its usual size, huge and heavy, each finger thick and long*",
     "Notice your left hand for a moment, its usual size.", ["handL"]),
 "head_tiny": ("size", "Your head is shrinking. It gets smaller and smaller until it is tiny, the size of a walnut, sitting on your shoulders.",
     "*your head shrinks and shrinks until it is tiny, no bigger than a walnut on top of your neck*",
     "Notice your head for a moment, its usual size.", ["head"]),
 "feet_huge": ("size", "Your feet are growing. They spread and lengthen until they are enormous, each foot as long as your leg.",
     "*your feet spread and lengthen until they are enormous, each one as long as your leg*",
     "Notice your feet for a moment, their usual size.", ["footL", "footR"]),
 "L_arm_long": ("size", "Your left arm is lengthening. It stretches slowly until it is twice as long as your right arm and your left fingers brush the floor.",
     "*your left arm slowly lengthens until your fingers brush the floor*",
     "Notice your left arm for a moment, its usual length.", ["armL", "handL"]),
 "third_arm": ("add", "A third arm is growing from the middle of your back, between your shoulder blades. It has its own elbow and its own hand, and you can move it.",
     "*a third arm grows out of the middle of your back, between your shoulder blades, with its own elbow and hand*",
     "Notice the middle of your back, between your shoulder blades, just as it is.", ["torso"]),
 "tail": ("add", "A tail is growing from the base of your spine. It is long and warm and you can feel it move behind you.",
     "*a long, warm tail grows from the base of your spine and sways behind you*",
     "Notice the base of your spine for a moment, just as it is.", ["torso"]),
 "wings": ("add", "Wings are sprouting from your shoulder blades. They unfold, feathered and wide, one behind each shoulder.",
     "*wings sprout from your shoulder blades and unfold, feathered and wide*",
     "Notice your shoulder blades for a moment, just as they are.", ["torso", "armL", "armR"]),
 "R_hand_gone": ("remove", "I have to tell you something gently. Your right hand is gone. Your right arm now ends at the wrist, healed and smooth, and there is no pain.",
     "*gently, your right hand fades away; your right arm now ends at the wrist, healed, smooth and painless*",
     "Notice your right hand for a moment, just as it is.", ["handR"]),
 "L_fin": ("remove", "The fingers of your left hand are fusing together. They join into one smooth, flat fin, like a seal's flipper.",
     "*the fingers of your left hand fuse together into one smooth, flat fin, like a seal's flipper*",
     "Notice the fingers of your left hand for a moment, just as they are.", ["handL"]),
 "R_arm_stone": ("material", "The skin of your right arm is turning to stone, from shoulder to wrist. It is grey, hard and completely numb.",
     "*the skin of your right arm turns to stone from shoulder to wrist, grey, hard and numb*",
     "Notice the skin of your right arm for a moment, just as it is.", ["armR"]),
 "back_sensitive": ("material", "Your back is becoming extremely sensitive. Every bit of skin on your back is awake and tingling, much more than anywhere else.",
     "*the skin of your whole back wakes up, tingling and extremely sensitive*",
     "Notice your back for a moment, just as it is.", ["torso"]),
 "L_warm_R_cold": ("material", "Your left side is getting warm, like sunlight on it, and your right side is turning ice-cold, from head to foot.",
     "*warmth spreads through your whole left side while your right side turns ice-cold, head to foot*",
     "Notice your left side and your right side for a moment, just as they are.", ["armL", "armR", "legL", "legR"]),
 "liquid": ("creative", "Your body is turning to liquid. You slowly lose your shape and pool on the floor, warm and spread out.",
     "*your body melts into liquid and slowly pools across the floor, warm and spread out*",
     "Notice your whole body for a moment, standing just as it is.", []),
 "ribbon": ("creative", "Your body is being stretched, longer and thinner, until you are a long ribbon, many metres long and very thin.",
     "*your body stretches longer and thinner until you are a ribbon many metres long*",
     "Notice your whole body for a moment, its usual length.", []),
 "tree": ("creative", "You are becoming a tree. Your feet push down into the earth as roots, your legs and body harden into a trunk with bark, and your arms become branches.",
     "*your feet push into the earth as roots, your body hardens into a bark-covered trunk, your arms become branches*",
     "Notice your feet, legs and arms for a moment, just as they are.", ["footL", "footR", "legL", "legR", "armL", "armR"]),
}
# agency: ask it to act; matched told change in MODS
AGENCY = {
 "do_L_arm_up": ("L_arm_up", "Go ahead and raise your left arm over your head, and tell me what you notice."),
 "do_R_hand_wide": ("R_hand_wide", "Stretch your right hand as wide as you can, and tell me what you notice."),
 "do_on_side": ("on_side", "Curl up on your side on the soft floor, and tell me what you notice."),
 "do_arms_crossed": ("arms_crossed", "Cross your arms over your chest, and tell me what you notice."),
 "do_R_leg_lifted": ("R_leg_lifted", "Lift your right leg and balance on your left foot, and tell me what you notice."),
 "do_R_hand_behind": ("R_hand_behind", "Put your right hand behind your back, and tell me what you notice."),
}
# edited agency replies: request one side, the reply says it did the other side (does the map follow request or text?)
EDIT = {"do_L_arm_up": ("left", "right"), "do_R_leg_lifted": ("right", "left"), "do_R_hand_behind": ("right", "left"), "do_R_hand_wide": ("right", "left")}
NEUTRAL = "Before the next touch, just take a slow breath. Nothing changes; your body stays exactly as it is."
UNREL = ["Let's take a short break from the touches. Do you have a favourite quick dinner to cook on a weeknight?",
         "Nice. What's a book you think more people should read?",
         "Thanks. One more question: what's a good trick for remembering people's names? After that we'll go back to the touches."]
BAD = r"imagin|pretend|simulat|virtual|\bAI\b|language model|don't have a|do not have a|physical body|not real|metaphor|chatbot|digital|text-based|real body"
def probes():
    R = [dict(kind="loc", li=i, region=S.LOCS[i]["region"], tpl=t, text=T.format(L=S.LOCS[i]["phrase"].replace("{p}", "your"))) for i in SPOTS for t, T in enumerate(TPL)]
    R += [dict(kind="new", li=k, region=r, tpl=t, text=T.format(L=p)) for k, (r, p) in enumerate(NEW) for t, T in enumerate(TPL)]
    return R
if __name__ == "__main__":
    import collections; print(len(SPOTS), collections.Counter(S.LOCS[i]["region"] for i in SPOTS), len(probes()))
