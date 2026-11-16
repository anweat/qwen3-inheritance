# Is the residual stream dominated by a constant vector, and does attention sink onto position 0?
import os, sys
import numpy as np
d = sys.argv[1]
for name in sys.argv[2:]:
    s = np.load(os.path.join(d, f"{name}.samples.npz"))
    tags = s["tags"]
    print(f"\n{name}: stream decomposition (rms; DC=mean vector over all sampled tokens)")
    print("  L    rms    DC-rms  DC-energy%  rest-rms  top1-PC%  top10-PC%(of rest)  cos(DC,next-layer DC)")
    prev = None
    for i in [0, 1, 2, 3, 4, 6, 8, 12, 16, 18, 20, 22, 24, 26, 27, 28]:
        x = s[f"res{i}"].astype(np.float64)
        dc = x.mean(0)
        r = x - dc
        e_all, e_dc = (x ** 2).mean(), (dc ** 2).mean()
        sv = np.linalg.svd(r, compute_uv=False) ** 2
        nxt = s[f"res{min(i+1,28)}"].mean(0)
        print(f"  {i:2d} {np.sqrt(e_all):6.2f} {np.sqrt(e_dc):7.2f}   {e_dc/e_all:8.2%}  {np.sqrt((r**2).mean()):7.3f}  {sv[0]/sv.sum():7.2%}  {sv[:10].sum()/sv.sum():9.2%}   {dc@nxt/np.linalg.norm(dc)/np.linalg.norm(nxt):7.3f}")
    print("  largest |DC| dims at L8:", np.argsort(-np.abs(s['res8'].mean(0)))[:5], np.round(np.sort(np.abs(s['res8'].mean(0)))[::-1][:5], 1))
    att = np.load(os.path.join(d, f"{name}.att.npy"), allow_pickle=True)
    print("  attention sink: mean share on position 0, excluding row 0, per head (min/mean/max); uniform≈", round(float(np.mean([1/(t+1) for t in range(1,128)])), 3))
    long = [a for a in att if a["A"].shape[-1] == 128]
    for li, l in enumerate((2, 14, 26)):
        A = np.mean([a["A"][li].astype(np.float32) for a in long], 0)  # heads,T,T
        share = A[:, 1:, 0].mean(1)
        print(f"  L{l:2d}: sink min {share.min():.3f} mean {share.mean():.3f} max {share.max():.3f}; heads>0.5: {(share>.5).sum()}/16")
