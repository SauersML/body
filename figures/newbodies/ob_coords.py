# canonical front-view coordinates (x: + = body's left = viewer's right; y: 0 top of head, down positive), units ~ cm-ish
C={'crown':(0,0),'forehead':(0,6),'earL':(9,11),'earR':(-9,11),'cheekL':(5,14),'cheekR':(-5,14),'lips':(0,18),'tongue':(0,16),
'neckF':(0,25),'neckB':(0,27),'chest':(0,42),'upback':(0,46),'belly':(0,60),'lowback':(0,66)}
arm=[('shoulder',(20,32)),('upperarm',(25,45)),('elbow',(28,58)),('forearm',(30,68)),('wrist',(32,78)),('palm',(33,84)),('backhand',(36,84)),
     ('thumb',(28,89)),('index',(32,95)),('middle',(35,96)),('ring',(37.5,95)),('little',(40,92))]
leg=[('hip',(11,74)),('thigh',(11,90)),('knee',(11,106)),('shin',(11,122)),('ankle',(11,138)),('heel',(13,143)),('sole',(9,145)),('toes',(6,146))]
for s,sg in (('L',1),('R',-1)):
    for n,(x,y) in arm+leg: C[n+s]=(sg*x,y)
EDGES=[('crown','forehead'),('forehead','earL'),('forehead','earR'),('earL','cheekL'),('earR','cheekR'),('cheekL','lips'),('cheekR','lips'),('lips','tongue'),
('lips','neckF'),('neckF','neckB'),('neckF','chest'),('chest','upback'),('chest','belly'),('belly','lowback'),('upback','lowback')]
for s in 'LR':
    E=[('shoulder','upperarm'),('upperarm','elbow'),('elbow','forearm'),('forearm','wrist'),('wrist','palm'),('palm','backhand'),('palm','thumb'),('palm','index'),('palm','middle'),('palm','ring'),('palm','little'),
       ('hip','thigh'),('thigh','knee'),('knee','shin'),('shin','ankle'),('ankle','heel'),('heel','sole'),('sole','toes')]
    EDGES+= [(a+s,b+s) for a,b in E]+[('neckF','shoulder'+s),('belly','hip'+s)]
