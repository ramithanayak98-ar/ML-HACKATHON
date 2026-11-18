import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from src.core import arc_geometry
class RoadGraph:
    def __init__(s,nodes,arcs):
        ax,ay,bx,by=arc_geometry(nodes,arcs); s.L=np.hypot(bx-ax,by-ay)
        s.u=arcs.u.to_numpy(); s.v=arcs.v.to_numpy(); n=len(nodes)
        s.G=csr_matrix((s.L+1e-3,(s.u,s.v)),shape=(n,n)); s.cache={}
    def node_dist(s,src,limit):
        if src not in s.cache: s.cache[src]=dijkstra(s.G,indices=src,limit=limit)
        return s.cache[src]
    def route_dist(s,a,ta,b,tb,limit):
        if a==b and tb>=ta: return (tb-ta)*s.L[a]
        d=s.node_dist(s.v[a],limit)[s.u[b]]
        return (1-ta)*s.L[a]+d+tb*s.L[b]
def viterbi(track,cand,proba,graph,beta=10.0,limit=2000.0,eps=1e-6):
    cand=cand.assign(p=np.clip(proba,eps,1)); pts=sorted(cand.point_idx.unique())
    x,y=track.x.to_numpy(),track.y.to_numpy()
    groups={i:g for i,g in cand.groupby("point_idx")}
    result={}; prev=None
    def flush(chain):
        # chain: list of (point_idx, arcs, ta, backptr)
        j=int(np.argmax(chain[-1][4]))
        for k in range(len(chain)-1,-1,-1):
            i,arcs_,_,bp,_=chain[k]; result[i]=int(arcs_[j])
            if k>0: j=int(bp[j])
    chain=[]
    for i in pts:
        g=groups[i]; arcs_=g.arc_id.to_numpy(); ta=g.proj_t.to_numpy(); em=np.log(g.p.to_numpy())
        if not chain:
            chain.append((i,arcs_,ta,None,em)); continue
        pi,parcs,pta,_,pscore=chain[-1]
        gc=np.hypot(x[i]-x[pi],y[i]-y[pi])
        trans=np.full((len(parcs),len(arcs_)),-np.inf)
        for a in range(len(parcs)):
            for b in range(len(arcs_)):
                d=graph.route_dist(parcs[a],pta[a],arcs_[b],ta[b],limit)
                if np.isfinite(d): trans[a,b]=-abs(d-gc)/beta
        tot=pscore[:,None]+trans
        if not np.isfinite(tot).any():   # broken chain -> restart
            flush(chain); chain=[(i,arcs_,ta,None,em)]; continue
        bp=np.argmax(tot,axis=0); score=tot[bp,np.arange(len(arcs_))]+em
        chain.append((i,arcs_,ta,bp,score))
    if chain: flush(chain)
    return result
