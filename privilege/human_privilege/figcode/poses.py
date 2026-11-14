import json
U=lambda *v:list(v)
stand=[["LeftUpLeg","LeftLeg",U(0.03,-1,0)],["RightUpLeg","RightLeg",U(-0.03,-1,0)],["LeftLeg","LeftFoot",U(0.0,-1,-0.02)],["RightLeg","RightFoot",U(0,-1,-0.02)],["LeftArm","LeftForeArm",U(0.12,-1,0.02)],["RightArm","RightForeArm",U(-0.12,-1,0.02)],["LeftForeArm","LeftHand",U(0.06,-1,0.08)],["RightForeArm","RightHand",U(-0.06,-1,0.08)]]
tail={"pts":[[0,0.0,-0.06],[0,-0.06,-0.14],[0,-0.16,-0.22],[0,-0.22,-0.33],[0,-0.17,-0.44],[0,-0.08,-0.47]],"r0":0.028,"r1":0.008}
fours=[["Spine","Spine1",U(0,0.12,1)],["Spine1","Spine2",U(0,0.12,1)],["Spine2","Neck",U(0,0.15,1)],["Neck","Head",U(0,0.6,1)],["Head","HeadTop_End",U(0,0.7,1)],
 ["LeftArm","LeftForeArm",U(0.05,-1,0.05)],["RightArm","RightForeArm",U(-0.05,-1,0.05)],["LeftForeArm","LeftHand",U(0,-1,0)],["RightForeArm","RightHand",U(0,-1,0)],
 ["LeftHand","LeftHandMiddle1",U(0,-0.15,1)],["RightHand","RightHandMiddle1",U(0,-0.15,1)],
 ["LeftUpLeg","LeftLeg",U(0.03,-1,0)],["RightUpLeg","RightLeg",U(-0.03,-1,0)],["LeftLeg","LeftFoot",U(0,-0.05,-1)],["RightLeg","RightFoot",U(0,-0.05,-1)],
 ["LeftFoot","LeftToeBase",U(0,-0.15,-1)],["RightFoot","RightToeBase",U(0,-0.15,-1)]]
