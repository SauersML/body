import numpy as np, re
from chains import L, PH, SP, SR
def side_of(i): return 'L' if SR[i].endswith('L') or re.search(r'\bleft\b',PH[i]) else ('R' if SR[i].endswith('R') or re.search(r'\bright\b',PH[i]) else '')
def part(i):
    p, r = PH[i], SR[i]
    if r == 'head': return 'head'
    if r == 'neck': return 'neck'
    if r == 'torso':
        if any(w in p for w in ('collarbone','chest','breastbone','armpit','upper back','shoulder blade','between the shoulder','top of spine')): return 'chest'
        return 'belly'
    if r.startswith('arm'):
        if 'shoulder' in p and 'upper arm' not in p: return 'shoulder'
        if 'upper arm' in p: return 'upper arm'
        if 'elbow' in p and 'forearm' not in p: return 'elbow'
        return 'forearm'
    if r.startswith('hand'):
        if any(w in p for w in ('finger','thumb','fingertip')): return 'fingers'
        return 'hand'
    if r.startswith('leg'):
        if 'hip' in p and 'thigh' not in p or 'buttock' in p: return 'hip'
        if 'thigh' in p: return 'thigh'
        if 'knee' in p and ('shin' not in p and 'lower leg' not in p): return 'knee'
        return 'shin'
    if r.startswith('foot'):
        if 'toe' in p: return 'toes'
        return 'foot'
S = [side_of(i) for i in range(len(PH))]
MID = ['head','neck','chest','belly']
GROUPS = {}
for i in range(len(PH)):
    k = (part(i), '' if part(i) in MID else S[i]); GROUPS.setdefault(k, []).append(i)
ARMC = ['shoulder','upper arm','elbow','forearm','hand','fingers']; LEGC = ['hip','thigh','knee','shin','foot','toes']
