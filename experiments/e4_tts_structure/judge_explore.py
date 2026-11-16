# Exploratory, after seeing the preset panel result, so nothing here can qualify a judge.
# Charsiu has no language model, so centring on TTS references may do more harm than good on human voices: does the raw sign
# work better? And how much of the human shortfall is clips that every judge calls the other reading?
import numpy as np
import judge_panel as J

T, N = J.T, J.N
ST = {j: J.scores(j, "tts_out/TTS-0.6B_nonstream") for j in J.JUDGES}
SN = {j: J.scores(j, "tts_natural/TTS-0.6B_nonstream") for j in J.JUDGES}


def lo(s, clip, mode):
    if mode == "centred":
        return J.centred(s, clip)
    c = s.get(clip)
    return None if not c else c.get("lo")


for j in J.JUDGES:
    for mode in ("centred", "raw"):
        for kind in ("seg", "tone"):
            g, a, h = [], [], []
            for i, r in T.items():
                if r["calibrate"] and J.ctype(r["gold"], r["alt"]) == kind and i in ST[j]:
                    x, y = lo(ST[j][i], "C_g", mode), lo(ST[j][i], "C_a", mode)
                    if x is not None:
                        g.append(x > 0)
                    if y is not None:
                        a.append(y < 0)
            for i, r in N.items():
                if J.ctype(r["gold"], r["alt"]) == kind and i in SN[j]:
                    x = lo(SN[j][i], "H", mode)
                    if x is not None:
                        h.append(x > 0)
            if g and h:
                print(f"{j:10s} {mode:7s} {kind:4s} TTS says-g {np.mean(g):.3f} says-a {np.mean(a):.3f} | human {np.mean(h):.3f} (n {len(h)})")

all_wrong = []
for i, r in N.items():
    v = [J.centred(SN[j].get(i, {}), "H") for j in J.JUDGES]
    if all(x is not None and x <= 0 for x in v):
        all_wrong.append(i)
print(f"\nhuman clips every judge scores as the other reading: {len(all_wrong)}")
for i in all_wrong:
    r = N[i]
    print(f"  {r['char']} {r['gold']}/{r['alt']} {r['text'][:30]}")
print("human rate with those clips removed")
for j in J.JUDGES:
    for kind in ("seg", "tone"):
        h = [J.centred(SN[j].get(i, {}), "H") for i, r in N.items() if J.ctype(r["gold"], r["alt"]) == kind and i not in all_wrong]
        h = [x > 0 for x in h if x is not None]
        if h:
            print(f"  {j:10s} {kind:4s} {np.mean(h):.3f} (n {len(h)})")
