# Time warp for v5: original v4 scene time (63.3 s) -> new time (~47.8 s). Holds shrink more than transitions.
import numpy as np, json
from scipy.interpolate import PchipInterpolator
K=[(0,0),(3.6,3.0),(4.3,3.6),(8.5,7.2),(9.3,7.8),(13.4,11.2),(14.4,12.0),(17.6,14.6),(20,16.4),(27.3,21.8),(31.5,24.6),
   (34,26.4),(40.8,31.4),(43.8,33.6),(46,35.2),(48,36.6),(52.5,39.9),(55.5,42.2),(60.5,45.8),(63.3,47.8)]
g=PchipInterpolator(*zip(*K))        # orig -> new
T0,T1=63.3,47.8
if __name__=="__main__":
    o=np.linspace(0,T0,200001); nw=g(o); tn=np.arange(0,T1+1e-9,0.005)
    json.dump({"dt":0.005,"T":T1,"orig":np.interp(tn,nw,o).round(5).tolist()},open('web/warp.json','w'))
    print('ok',len(tn))
