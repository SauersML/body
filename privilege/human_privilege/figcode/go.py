import json,subprocess,sys
from poses import *
ALL=[["r_dog.png",{"m":"dog.glb","c":"#7d8791","az":-110,"el":10}],["r_dog2.png",{"m":"dog.glb","c":"#7d8791","az":-140,"el":10}],["r_cre.png",{"m":"cre.glb","c":"#3b8c86","az":-75}],["r_cre2.png",{"m":"cre.glb","c":"#3b8c86","az":-110}],
   ["r_stand.png",{"m":"hum_rig.fbx","aim":stand,"az":-30}],
   ["r_tail.png",{"m":"hum_rig.fbx","aim":stand,"tail":tail,"az":-75}],
   ["r_fours.png",{"m":"hum_rig.fbx","aim":fours,"rot":[90,0,0],"az":-60,"el":10}]]
sel=sys.argv[1:]
J=[j for j in ALL if not sel or j[0] in sel]
subprocess.run(["$BODY_WORK/.venv/bin/python","shot.py",json.dumps(J)],env={"PLAYWRIGHT_BROWSERS_PATH":"$BODY_WORK","PATH":"/usr/bin:/bin"})
