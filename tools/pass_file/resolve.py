import json,glob,os,re,collections
D='/tmp/claude-0/-home-user-Playbook-program/945c82da-81db-5d70-b18b-7bd69bcf608b/scratchpad/'
def coll(d):
    return {os.path.basename(f)[:-5]:json.load(open(f)) for f in glob.glob(D+d+'/*.json')}
pieces=coll('fa/pieces'); names={k:v['entries'] for k,v in coll('fa/names').items()}
reviews=coll('fx8/reviews'); aliases={k:v['map'] for k,v in coll('fx7/aliases').items()}
ucuts={k:v['fs'] for k,v in coll('fx7/cuts').items()}
site=lambda f:json.load(open(D+'site/'+f))
pp=site('pass_pieces.json'); idx=site('review_index.json'); relay=site('relay_A.json')
cauto=site('cuts_auto.json'); cross=site('crossings.json')
segs=idx['segments']
def segOf(f):
    for g in segs:
        if g['seg_start']<=f<g['seg_start']+g['frames']: return g
SUBS=[("11_2H_45-50",{"A94":"A70","A65":"A27"}),("17_2H_75-80",{"A10":"A14"})]
TYPOS={"A66":"A65","A52":"A62","B8":"B21"}   # B8 and B21 are one player; B21 is the label used on almost every box
isTag=lambda n:bool(n) and bool(re.match(r'^A[A-Z]',n)) and n!="AGK"
def subName(n,f):
    for seg,m in SUBS:
        g=[x for x in segs if x['name']==seg][0]
        if f>=g['seg_start'] and n in m: n=m[n]
    return n
def alias(n,f):
    if n=="AGK": return "A97"
    if n in TYPOS: n=TYPOS[n]
    if n and re.match(r'^A\d+$',n): return subName(n,f)
    if not isTag(n): return n
    g=segOf(f); m=aliases.get(g['name']) if g else None
    return (m or {}).get(n,n)
# switches
cuts=collections.defaultdict(set); FROM={}
for p,v in cauto.items():
    for f,t in v: cuts[p].add(f)
for p,v in ucuts.items():
    for f in v: cuts[p].add(f)
for seg,cs in cross.items():
    for c in cs:
        if c[6]=="colour" and c[5] is not None and abs(c[5])>=0.5 and c[5]>0:
            p,q,_,_,f=c[:5]
            if c[7]: cuts[str(p)].add(f); FROM[f"{p}_{f}"]=(str(q),f)
            if c[8]: cuts[str(q)].add(f); FROM[f"{q}_{f}"]=(str(p),f)
cuts={p:sorted(v) for p,v in cuts.items()}
def keyOf(p,f):
    k=str(p)
    for x in cuts.get(str(p),[]):
        if x<=f: k=f"{p}_{x}"
        else: break
    return k
def nameAt(jid,frame):
    e=names.get(str(jid))
    if not e: return None
    g=segOf(frame); v=[x for x in e if g['seg_start']<=x[0]<g['seg_start']+g['frames']]
    if not v: return None
    n=v[0][1]
    for f,x in v:
        if f<=frame: n=x
    return n
P=relay['pieces']; ros=relay['names']
def rOwn(pid):
    L=pieces.get(str(pid))
    if L and L.get('name') in ros: return L['name']
    if L and (L.get('role') in('referee','unknown') or L.get('team')==1 or (L.get('name') or '').startswith('B')): return '—'
    return relay['seed'].get(str(pid))
votes=collections.defaultdict(collections.Counter)
for q,r in P.items():
    n=rOwn(q)
    if not n or n=='—' or r[6]<0: continue
    votes[r[6]][n]+=r[1]-r[0]+1
gname={}
for g,c in votes.items():
    (n,v),=c.most_common(1)
    if v/sum(c.values())>=0.7: gname[g]=n
named=[]
for q,r in P.items():
    own=rOwn(q); n=own or gname.get(r[6])
    if n and n!='—': named.append([q,n,r[0],r[1],not own])
clash=set(); by=collections.defaultdict(list)
for r in named: by[r[1]].append(r)
for n,v in by.items():
    v.sort(key=lambda r:r[2])
    for i in range(len(v)):
        for k in range(i+1,len(v)):
            if v[k][2]>v[i][3]: break
            if min(v[i][3],v[k][3])-v[k][2]>15:
                if v[i][4]: clash.add(v[i][0])
                if v[k][4]: clash.add(v[k][0])
def rName(pid):
    own=rOwn(pid)
    if own: return own
    r=P.get(str(pid))
    if not r or str(pid) in clash: return None
    return gname.get(r[6])
needs=lambda n:n!="BGK" and (not n or isTag(n) or not re.match(r'^[AB]\d+$',n))
def effName(base,f,jid=None,depth=0):
    key=keyOf(base,f); L=pieces.get(key) or {}
    if L.get('role')=='referee': return None
    if L.get('name'): return L['name']
    sf=FROM.get(key)
    if sf and depth<8:
        n=effName(sf[0],sf[1]-1,None,depth+1)
        if n: return n
    seed=relay['seed'].get(key)
    if seed: return seed
    if key==str(base) and jid is not None: return nameAt(jid,f)
    return None
def who(w,piece):
    """returns (name, source)"""
    if not w: return None,None
    if piece is not None and w.get('frame') is not None:
        base=str(piece) if str(piece).startswith('m-') else str(piece).split('_')[0]
        k=keyOf(base,w['frame']); L=pieces.get(k)
        if L and L.get('name'): return alias(L['name'],w['frame']),'fix'
        e=effName(base,w['frame'],w.get('jid'))
        if e and not needs(alias(e,w['frame'])): return alias(e,w['frame']),'swap'

        if not (L and L.get('role')):
            n=rName(k)
            if n and n!='—': return alias(n,w['frame']),'group'
    n=w.get('name') or (nameAt(w['jid'],w['frame']) if w.get('jid') is not None else None)
    return (alias(n,w['frame']) if w.get('frame') is not None else n),'pass'
