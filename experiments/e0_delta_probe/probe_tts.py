import numpy as np, probe_delta as P
H = {k: P.headers(r, f) for k, (r, f, _) in P.MODELS.items()}
def T(m, L, t):
    pre = P.MODELS[m][2]
    return P.fetch(H[m][f"{pre}layers.{L}.{t}.weight"], P.ROWS if not t.endswith('norm') else None, 0)
for m in ["instruct", "asr", "tts"]:
    nr, cs, ln = [], [], []
    for L in range(28):
        for t in P.MATS:
            W, B = T(m, L, t), T("base", L, t)
            nr.append(np.linalg.norm(W) / np.linalg.norm(B)); cs.append((W*B).sum()/np.linalg.norm(W)/np.linalg.norm(B))
        a, b = T(m, L, "input_layernorm"), T("base", L, "input_layernorm")
        ln.append(np.corrcoef(a, b)[0, 1])
    print(f"{m:8s} norm ratio mean {np.mean(nr):.3f} [{np.min(nr):.3f},{np.max(nr):.3f}]  cos mean {np.mean(cs):.3f} [{np.min(cs):.3f},{np.max(cs):.3f}]  layernorm corr mean {np.mean(ln):.3f}")
# does TTS match base under a per-layer shift? check TTS layer i vs base layer j cos for q_proj
M = np.zeros((28, 28))
for i in range(28):
    for j in range(28):
        W, B = T("tts", i, "self_attn.q_proj"), T("base", j, "self_attn.q_proj")
        M[i, j] = (W*B).sum()/np.linalg.norm(W)/np.linalg.norm(B)
print("tts q_proj best cross-layer cos per layer:", np.round(M.max(1), 3)[:10], "argmax", M.argmax(1)[:10])
