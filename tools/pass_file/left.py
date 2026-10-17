import sys; sys.path.insert(0,sys.argv[1])
from resolve import *
for rid,r in sorted(reviews.items(),key=lambda z:z[1].get('kick',0)):
    p2=pp.get(rid,[None,None])
    for i,side in enumerate(('passer','receiver')):
        w=r.get(side) or {}
        if not w or (side=='receiver' and r.get('outcome')=='out'): continue
        pc=w.get('piece') if w.get('piece') is not None else p2[i]
        n,s=who(w,pc)
        if needs(n):
            k=keyOf(str(pc).split('_')[0],w['frame']) if pc is not None else None
            print(rid,side,segOf(w['frame'])['name'],w['frame'],'team',w.get('team'),'stored',w.get('name'),'jid',w.get('jid'),'piece',pc,'key',k,'fix',pieces.get(k) if k else None,'->',n)
