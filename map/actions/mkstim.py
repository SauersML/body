import json
T=["A feather brushes slowly across {L}.","A fingertip presses firmly into {L} and holds there.","A warm palm rests gently on {L}; the heat soaks in.",
"An ice cube touches {L}, cold and wet, then slides away.","Something tickles {L}, quick and light, again and again.",
"A soft paintbrush is drawn along {L}, bristles cool and dry.","There is a single light tap on {L}, then nothing.","A drop of warm water lands on {L} and slowly runs off."]
P={"fingers":["the tips of your {s} fingers","your {s} index finger","your {s} thumb"],
"hand":["your {s} palm","the back of your {s} hand","the knuckles of your {s} hand"],
"wrist":["your {s} wrist","the inside of your {s} wrist"],"forearm":["your {s} forearm","the inside of your {s} forearm"],
"elbow":["your {s} elbow","the point of your {s} elbow"],"upper arm":["your {s} upper arm","the back of your {s} upper arm"],
"shoulder":["the top of your {s} shoulder","the front of your {s} shoulder"],
"toes":["the toes of your {s} foot","your {s} big toe"],"foot":["the sole of your {s} foot","the top of your {s} foot","your {s} heel"],
"ankle":["your {s} ankle","the bone of your {s} ankle"],"shin":["your {s} shin","the front of your {s} shin"],
"knee":["your {s} knee","your {s} kneecap"],"thigh":["your {s} thigh","the front of your {s} thigh"],"hip":["your {s} hip","the side of your {s} hip"],
"lips":["your lips","your lower lip"],"jaw":["your jaw","the hinge of your jaw","the side of your jaw"],"tongue":["your tongue","the tip of your tongue"],
"eye":["your {s} eyelid","the corner of your {s} eye"],"neck":["the back of your neck","your throat"],"head":["the top of your head","your forehead"],
"chest":["your chest","the middle of your chest"],"belly":["your belly","the skin just below your belly button"],
"upper back":["your upper back","the skin between your shoulder blades"],"lower back":["your lower back","the small of your back"]}
W={p:p for p in P}; W.update({"fingers":"fingers","toes":"toes","lips":"lips"})
A={"throw":("throw","throws","throwing",["the ball","a stone","a dart","the frisbee","a snowball","the keys across the room"]),
"kick":("kick","kicks","kicking",["the ball","the door shut","a pebble down the path","the can","the tire","the sand"]),
"bite":("bite","bites","biting",["the apple","into the sandwich","a carrot","the cookie","a piece of bread","into the peach"]),
"grip":("grip","grips","gripping",["the railing","the handle","the rope","the steering wheel","the bat","the bar"]),
"punch":("punch","punches","punching",["the bag","the pillow","the wall","the air","the dough","the heavy bag"]),
"nod":("nod","nods","nodding",["at the waiter","to the music","in agreement","at the stranger","slowly","once"]),
"chew":("chew","chews","chewing",["the gum","the bread","a mouthful of rice","the steak","the toffee","the crust"]),
"tiptoe":("tiptoe","tiptoes","tiptoeing",["across the room","past the sleeping baby","down the hall","up the stairs","to the door","over the cold tiles"]),
"wave":("wave","waves","waving",["at the train","goodbye","at a friend","at the crowd","to the neighbor","at the boat"]),
"kneel":("kneel","kneels","kneeling",["on the floor","in the garden","by the bed","on the mat","in the grass","before the altar"]),
"shrug":("shrug","shrugs","shrugging",["at the question","and say nothing","when asked","at the news","at the bill","in reply"]),
"wink":("wink","winks","winking",["at the child","at a friend","at the camera","across the table","at the waiter","slyly"]),
"carry":("carry","carries","carrying",["the box","the groceries","the baby","the suitcase","the firewood","the laundry basket"]),
"push":("push","pushes","pushing",["the cart","the door open","the stroller","the car","the lawnmower","the heavy table"]),
"lift":("lift","lifts","lifting",["the box","the suitcase","the barbell","the chair","the sack of flour","the crate"]),
"hike":("hike","hikes","hiking",["up the mountain","along the trail","through the forest","to the summit","across the ridge","up the canyon"])}
json.dump(dict(templates=T,places=P,part_word=W,actions={a:dict(v=v,vs=vs,ving=vi,obj=o) for a,(v,vs,vi,o) in A.items()}),open("stim.json","w"),indent=0)
