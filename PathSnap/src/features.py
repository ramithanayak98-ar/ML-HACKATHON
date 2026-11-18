import numpy as np, pandas as pd
from src.core import arc_geometry
def build_features(track,nodes,arcs,cand):
    ax,ay,bx,by=arc_geometry(nodes,arcs)
    x,y,t=track.x.to_numpy(),track.y.to_numpy(),track.t.to_numpy()
    xp,yp=np.r_[x[0],x[:-1]],np.r_[y[0],y[:-1]]; xn,yn=np.r_[x[1:],x[-1]],np.r_[y[1:],y[-1]]
    tp,tn=np.r_[t[0],t[:-1]],np.r_[t[1:],t[-1]]
    heading=np.degrees(np.arctan2(xn-xp,yn-yp))%360
    speed=np.hypot(xn-xp,yn-yp)/np.maximum(tn-tp,1e-6)
    gap_prev=t-tp
    i=cand.point_idx.to_numpy(); a=cand.arc_id.to_numpy()
    arc_bear=np.degrees(np.arctan2(bx[a]-ax[a],by[a]-ay[a]))%360
    bd=np.abs(heading[i]-arc_bear)%360; bd=np.minimum(bd,360-bd)
    f=cand.copy()
    f["arc_len_m"]=np.hypot(bx-ax,by-ay)[a]
    f["speed_mps"]=speed[i]; f["gap_prev_s"]=gap_prev[i]
    f["bearing_diff_deg"]=bd; f["cos_bearing"]=np.cos(np.radians(bd))
    f["low_speed"]=(speed[i]<1.0).astype(int)
    g=f.groupby("point_idx")["dist_m"]
    f["n_cand"]=g.transform("size"); f["dist_minus_min"]=f.dist_m-g.transform("min")
    f["dist_ratio"]=f.dist_m/(g.transform("mean")+1e-6)
    return f
FEATURES=["dist_m","dist_rank","proj_t","arc_len_m","speed_mps","gap_prev_s","bearing_diff_deg","cos_bearing","low_speed","n_cand","dist_minus_min","dist_ratio"]
