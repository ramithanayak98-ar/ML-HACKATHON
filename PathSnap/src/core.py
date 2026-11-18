import numpy as np, pandas as pd
from scipy.spatial import cKDTree
R=6371000.0
def load_trip(folder, trip_id):
    p=f"{folder}/{trip_id}/{trip_id}"
    track=pd.read_csv(p+".track",sep=r"\s+",header=None,names=["lon","lat","t"])
    nodes=pd.read_csv(p+".nodes",sep=r"\s+",header=None,names=["lon","lat"])
    arcs=pd.read_csv(p+".arcs",sep=r"\s+",header=None,names=["u","v"])
    try: route=pd.read_csv(p+".route",header=None,names=["arc_id"])["arc_id"].to_numpy()
    except FileNotFoundError: route=None
    return track,nodes,arcs,route
def to_xy(lon,lat,lon0,lat0):
    x=np.radians(lon-lon0)*R*np.cos(np.radians(lat0)); y=np.radians(lat-lat0)*R
    return x,y
def project_trip(track,nodes):
    lon0,lat0=track.lon.mean(),track.lat.mean()
    track["x"],track["y"]=to_xy(track.lon,track.lat,lon0,lat0)
    nodes["x"],nodes["y"]=to_xy(nodes.lon,nodes.lat,lon0,lat0)
    return track,nodes
def arc_geometry(nodes,arcs):
    ax=nodes.x.to_numpy()[arcs.u]; ay=nodes.y.to_numpy()[arcs.u]
    bx=nodes.x.to_numpy()[arcs.v]; by=nodes.y.to_numpy()[arcs.v]
    return ax,ay,bx,by
def point_seg_dist(px,py,ax,ay,bx,by):
    dx,dy=bx-ax,by-ay; L2=dx*dx+dy*dy
    t=np.clip(((px-ax)*dx+(py-ay)*dy)/np.where(L2==0,1,L2),0,1)
    qx,qy=ax+t*dx,ay+t*dy
    return np.hypot(px-qx,py-qy),t
def label_points(track,nodes,arcs,route):
    ax,ay,bx,by=arc_geometry(nodes,arcs)
    rax,ray,rbx,rby=ax[route],ay[route],bx[route],by[route]
    n,m=len(track),len(route); px,py=track.x.to_numpy(),track.y.to_numpy()
    back=np.zeros((n,m),dtype=np.int32); prev=None
    for i in range(n):
        c,_=point_seg_dist(px[i],py[i],rax,ray,rbx,rby)
        if prev is None: D=c
        else:
            cm=np.minimum.accumulate(prev); arg=np.zeros(m,dtype=np.int32)
            # index of running min
            idx=np.arange(m); best=np.where(prev==cm,idx,0); arg=np.maximum.accumulate(best)
            D=c+cm; back[i]=arg
        prev=D
    j=int(np.argmin(prev)); out=np.empty(n,dtype=np.int64)
    for i in range(n-1,-1,-1):
        out[i]=route[j]
        if i>0: j=back[i,j]
    return out
def generate_candidates(track,nodes,arcs,radius=50.0,k_max=10,step=10.0):
    ax,ay,bx,by=arc_geometry(nodes,arcs); L=np.hypot(bx-ax,by-ay)
    nseg=np.maximum(1,np.ceil(L/step).astype(int))
    arc_rep=np.repeat(np.arange(len(arcs)),nseg+1)
    f=np.concatenate([np.linspace(0,1,k+1) for k in nseg])
    sx=ax[arc_rep]+f*(bx-ax)[arc_rep]; sy=ay[arc_rep]+f*(by-ay)[arc_rep]
    tree=cKDTree(np.c_[sx,sy]); rows=[]
    px,py=track.x.to_numpy(),track.y.to_numpy()
    for i,hits in enumerate(tree.query_ball_point(np.c_[px,py],r=radius+step)):
        cand=np.unique(arc_rep[hits])
        if len(cand)==0: continue
        d,t=point_seg_dist(px[i],py[i],ax[cand],ay[cand],bx[cand],by[cand])
        keep=d<=radius; cand,d,t=cand[keep],d[keep],t[keep]
        o=np.argsort(d)[:k_max]
        for r_,j in enumerate(o): rows.append((i,int(cand[j]),d[j],t[j],r_))
    return pd.DataFrame(rows,columns=["point_idx","arc_id","dist_m","proj_t","dist_rank"])
