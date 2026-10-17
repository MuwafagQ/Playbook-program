import sys,csv,json; sys.path.insert(0,sys.argv[1])
from resolve import *
clock=coll('fx7/clock')
entries=sorted((e[0],e[1]) for v in clock.values() for e in v['entries'])
# two board readings disagree with all the others by 5 and 2.7 minutes (frame 0 before kick-off; frame 105121), so they are left out
entries=[e for e in entries if e not in ((0,297),(105121,3504))]
half_of=lambda g:1 if '_1H_' in g['name'] else 2
def board(f):
    g=segOf(f); h=half_of(g)
    same=[e for e in entries if half_of(segOf(e[0]))==h]
    if not same: return None
    e=min(same,key=lambda e:abs(e[0]-f))          # the nearest board reading in this half
    return max(e[1]+(f-e[0])/idx['fps'], 0 if h==1 else 45*60)
# match momentum: a mark {f, team, phase} holds until the next mark in the same half; the team without the ball is in Retreat
OPP={"Buildup":"Retreat","Transition":"Retreat","Finishing":"Retreat","Retreat":"Buildup"}
try: MARKS=sorted(coll('moments').values(),key=lambda m:m['f'])
except Exception: MARKS=[]
H2=min((g['seg_start'] for g in segs if '_2H_' in g['name']),default=10**12)
def moment(f):
    cur=None
    for m in MARKS:
        if m['f']>f: break
        if (m['f']>=H2)==(f>=H2): cur=m
    if not cur: return '',''
    p=cur['phase']; return (p,OPP[p]) if cur['team']==0 else (OPP[p],p)
mmss=lambda s:None if s is None else f"{int(s//60)}:{int(s%60):02d}"
out=[]
for rid,r in reviews.items():
    p2=pp.get(rid,[None,None]); row={'pass_id':rid}
    k=r.get('kick') or (r.get('passer') or {}).get('frame'); t=r.get('touch') or (r.get('receiver') or {}).get('frame')
    g=segOf(k); row.update(segment=g['name'],half=half_of(g),kick_frame=k,touch_frame=t,
        kick_clock=mmss(board(k)),touch_clock=mmss(board(t)) if t else None,
        kick_video_s=round(k/idx['fps'],2),outcome=r.get('outcome'),notes=r.get('notes') or '')
    row['momentum_A'],row['momentum_B']=moment(k)
    for i,side in enumerate(('passer','receiver')):
        w=r.get(side) or {}
        if side=='receiver' and r.get('outcome')=='out': w={}
        pc=w.get('piece') if w.get('piece') is not None else p2[i]
        n,s=who(w,pc) if w else (None,None)
        team=w.get('team'); row[side+'_team']={0:'A',1:'B'}.get(team,'')
        row[side+'_number']=int(n[1:]) if n and re.match(r'^[AB]\d+$',n) else ('GK' if n=='BGK' else '')
        row[side+'_label']=n or ''
        row[side+'_x']=round(w['x']) if w.get('x') is not None else ''; row[side+'_y']=round(w['y']) if w.get('y') is not None else ''
        row[side+'_name_source']=s or ''
        row[side+'_to_check']='yes' if w and needs(n) else ''
    out.append(row)
out.sort(key=lambda r:r['kick_frame'])
fn=sys.argv[2]
with open(fn,'w',newline='') as fh:
    wr=csv.DictWriter(fh,fieldnames=list(out[0].keys())); wr.writeheader(); wr.writerows(out)
print(len(out),'rows ->',fn); print(out[0]); print(out[400])
print('to check',sum(1 for r in out if r['passer_to_check'] or r['receiver_to_check']))
print('clock range', out[0]['kick_clock'], out[-1]['kick_clock'])
