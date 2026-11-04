"""Independent checks of the PTP spec. Exact mixture-of-Gaussians FM velocity, no training."""
import numpy as np
rng = np.random.default_rng(0)

# ---------- 1. Spec's own embedded check + algebra ----------
def maximal_kernel(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    excess = np.maximum(q-p, 0); mass = excess.sum()
    K = np.zeros((len(p), len(p)))
    for i in range(len(p)):
        if p[i] == 0:
            K[i] = q; continue
        keep = min(1., q[i]/p[i]); K[i, i] = keep
        if mass > 1e-14: K[i] += (1-keep)*excess/mass
    return K
p = np.array([.5,.3,.2]); q = np.array([.2,.3,.5]); K = maximal_kernel(p,q)
assert np.allclose(p@K,q) and np.isclose(1-np.sum(p*np.diag(K)), .5*np.abs(p-q).sum())
print("1a maximal coupling embedded check: pass")

f  = lambda r: .5*np.log(1-r**2)+(1+r**2)/(2*(1-r**2))
fp = lambda r: r*(1+r**2)/(1-r**2)**2
rs = np.linspace(0.01,.95,50); h=1e-6
print("1b f'(rho) max abs err vs finite diff: %.2e" % np.max(np.abs((f(rs+h)-f(rs-h))/(2*h)-fp(rs))))

# KL number in sec 21.1: which ||anchor||^2 does it imply?
rho=.95; d=6; base=d*((1-rho**2)-1-np.log(1-rho**2))
print("1c KL=7.03385 implies ||anchor||^2 = %.3f (spec does not state the anchor)" % ((2*7.03385-base)/rho**2))
print("1d E[KL] = mutual info = -d/2 log(1-rho^2): per coord at .95 = %.3f nats; 12 tokens x 7 dims = %.1f nats"
      % (-.5*np.log(1-rho**2), -.5*np.log(1-rho**2)*84))

# RQ spline derivative
def rq(x,w,h_,d0,d1):
    xi=x/w; s=h_/w; den=s+(d0+d1-2*s)*xi*(1-xi)
    return h_*(s*xi**2+d0*xi*(1-xi))/den, s**2*(d1*xi**2+2*s*xi*(1-xi)+d0*(1-xi)**2)/den**2
e=0
for _ in range(1000):
    w,h_,d0,d1=np.exp(rng.normal(size=4)); x=rng.uniform(.01,.99)*w; g=1e-6*w
    e=max(e,abs((rq(x+g,w,h_,d0,d1)[0]-rq(x-g,w,h_,d0,d1)[0])/(2*g)-rq(x,w,h_,d0,d1)[1]))
print("1e RQ derivative max err: %.2e" % e)

# ---------- 2. Exact FM flow for a Gaussian mixture ----------
class ExactFM:
    """X_t=(1-t)Z+tA, A~sum pi_i N(m_i,S). Euler on the exact marginal velocity."""
    def __init__(s, means, S, steps=100):
        s.m=np.asarray(means); s.S=S; s.n=S.shape[0]; s.dt=1./steps
        s.ts=(np.arange(steps))/steps; s.cache=[]
        for t in s.ts:
            C=(1-t)**2*np.eye(s.n)+t**2*S; Ci=np.linalg.inv(C)
            s.cache.append((Ci, t*S@Ci-(1-t)*Ci))
    def __call__(s, z, logpi=None):
        x=z.copy(); k=len(s.m)
        lp=np.zeros(k) if logpi is None else logpi
        lp=np.broadcast_to(lp,(x.shape[0],k)) if lp.ndim==1 else lp
        for t,(Ci,G) in zip(s.ts,s.cache):
            r=[x-t*m for m in s.m]
            l=np.stack([lp[:,i]-.5*np.einsum('ij,jk,ik->i',r[i],Ci,r[i]) for i in range(k)],1)
            w=np.exp(l-l.max(1,keepdims=True)); w/=w.sum(1,keepdims=True)
            v=sum(w[:,[i]]*(s.m[i]+r[i]@G.T) for i in range(k))
            x=x+s.dt*v
        return x

H,L=16,4
def toeplitz_cov(sig,ell,n=H):
    i=np.arange(n); return sig**2*np.exp(-(i[:,None]-i[None])**2/(2*ell**2))+1e-6*np.eye(n)

# Sanity: Gaussian target -> A = m + S^{1/2} z (symmetric root)
S=toeplitz_cov(.3,3.); fm=ExactFM([np.zeros(H)],S,400)
z=rng.standard_normal((2000,H)); A=fm(z)
ev,U=np.linalg.eigh(S); root=U@np.diag(np.sqrt(ev))@U.T
print("2a unimodal: |A - S^{1/2} z| rmse=%.4f; A std=%.3f" % (np.sqrt(np.mean((A-z@root)**2)),A.std()))
print("2b S^{1/2} row 8 (locality, kernel around j=8):", np.round(root[8,4:13],3))
print("   mid-row share of squared mass within +-3 tokens: %.3f" % ((root[8,5:12]**2).sum()/(root[8]**2).sum()))

def step_carry(z,rho,mode):
    eps=rng.standard_normal(z.shape); n=z.copy()
    if mode=='shift':   # the spec: aligned, fresh tail
        n=np.zeros_like(z); n[:,:H-L]=rho*z[:,L:]+np.sqrt(1-rho**2)*eps[:,:H-L]; n[:,H-L:]=eps[:,H-L:]
    elif mode=='same':  # unaligned same-index ("fixed latent carry")
        n=rho*z+np.sqrt(1-rho**2)*eps
    elif mode=='cyc':   # aligned, executed tokens recycled into tail
        n=rho*np.roll(z,-L,axis=1)+np.sqrt(1-rho**2)*eps
    return n

profiles={'uniform':np.ones(H),'ramp':np.arange(1,H+1,dtype=float)}
Sw=toeplitz_cov(.15,3.)
print("\n3. Open-loop mode keep rate per replan (symmetric 2-mode, stationary state, N=4000, exact flow)")
print("   analytic for shift, uniform: 1/2+arcsin(.75 rho)/pi")
for name,prof in profiles.items():
    dv=prof/np.linalg.norm(prof)*np.sqrt(H)*1.0
    fm=ExactFM([dv,-dv],Sw,60)
    label=lambda a: np.sign(a@dv)
    for mode in ['shift','cyc','same']:
        row=[]
        for rho in [0,.5,.85,.95]:
            z=rng.standard_normal((4000,H)); a0=label(fm(z)); z2=step_carry(z,rho,mode); a1=label(fm(z2))
            row.append(np.mean(a0==a1))
        print("   %-8s %-6s rho=0/.5/.85/.95: "%(name,mode)+" ".join("%.3f"%r for r in row))
print("   analytic uniform shift: "+" ".join("%.3f"%(.5+np.arcsin(.75*r)/np.pi) for r in [0,.5,.85,.95]))

# Memory horizon: keep rate k replans later
print("\n4. Mode agreement after m replans at rho=.95 (uniform profile)")
dv=np.ones(H); fm=ExactFM([dv,-dv],Sw,60); label=lambda a: np.sign(a@dv)
for mode in ['shift','same']:
    z=rng.standard_normal((4000,H)); a0=label(fm(z)); out=[]
    for m in range(1,9):
        z=step_carry(z,.95,mode); out.append(np.mean(label(fm(z))==a0))
    print("   %-6s m=1..8: "%mode+" ".join("%.3f"%o for o in out))
# exact latent independence at m=H/L
z0=rng.standard_normal((200000,H)); z=z0
for m in range(4): z=step_carry(z,.95,'shift')
print("   shift: max |corr(z_k+4[i], z_k[j])| over all i,j = %.4f (MC noise ~%.4f)"%(np.abs((z.T@z0)/len(z)).max(),3/np.sqrt(len(z))))

# Within-mode jitter: overlap disagreement in action space, unimodal smooth law
print("\n5. Unimodal jitter: RMS(new plan - old plan) on physically overlapping tokens")
fm=ExactFM([np.zeros(H)],toeplitz_cov(.3,3.),60)
for mode in ['shift','same']:
    row=[]
    for rho in [0,.5,.85,.95]:
        z=rng.standard_normal((3000,H)); a0=fm(z); a1=fm(step_carry(z,rho,mode))
        row.append(np.sqrt(np.mean((a1[:,:H-L]-a0[:,L:])**2)))
    print("   %-6s rho=0/.5/.85/.95: "%mode+" ".join("%.3f"%r for r in row))
print("   analytic aligned: sqrt(2(1-rho))*.3: "+" ".join("%.3f"%(np.sqrt(2*(1-r))*.3) for r in [0,.5,.85,.95]))

# ---------- 6. Closed loop: state feedback vs latent memory ----------
# Lateral state y; per-replan the decoder's mode weight depends on y through beta.
# Chunk = per-step lateral velocity (+v or -v profile). Execute L, y += sum of executed.
print("\n6. Closed-loop dither from y=0: lateral velocity chunks, modes +/-v, log-odds = beta*y")
print("   metric: mean executed-direction reversals over 25 replans; steps until |y|>commit (cap 100)")
v=.05; dv=np.full(H,v); fmc=ExactFM([dv,-dv],toeplitz_cov(.01,3.),40)
def closed(beta,rho,mode,N=1500,R=25,commit=1.0):
    y=np.zeros(N); z=rng.standard_normal((N,H)); last=np.zeros(N); rev=np.zeros(N); tc=np.full(N,R*L,float)
    for k in range(R):
        if k>0: z=step_carry(z,rho,mode)
        lo=beta*y; lp=np.stack([lo/2,-lo/2],1)
        a=fmc(z,lp); ex=a[:,:L].sum(1); s=np.sign(ex)
        rev+=(s!=last)&(last!=0); last=s; y=y+ex
        hit=(np.abs(y)>commit)&(tc==R*L); tc[hit]=(k+1)*L
    return rev.mean(),tc.mean()
for beta in [0.,2.,8.]:
    for mode in ['shift','same']:
        s=" ".join("%.2f/%3.0f"%closed(beta,r,mode) for r in [0,.5,.85,.95])
        print("   beta=%-3g %-6s rho=0/.5/.85/.95 (rev/t_commit): %s"%(beta,mode,s))

# ---------- 7. Budget ----------
print("\n7. Selection noise in sec 11.1 under a TRUE NULL (every rho has the same p)")
for p0 in [.5,.8,.95]:
    d=rng.binomial(20,p0,(20000,12))/20
    print("   p=%.2f: E[best of 12 at n=20] - p = +%.3f; P(best beats rho=0 arm by >=10pp) = %.2f"
          %(p0,(d.max(1)-p0).mean(),((d.max(1)-d[:,0])>=.1).mean()))
from math import ceil
z=1.96+.84
print("   n per arm, unpaired, 80% power, alpha .05:")
for p0,dl in [(.8,.05),(.8,.1),(.5,.1),(.95,.03)]:
    pb=p0+dl/2
    print("     p=%.2f, delta=%.2f: n=%d"%(p0,dl,ceil(2*z*z*pb*(1-pb)/dl**2)))
