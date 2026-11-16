# Direct, proxy-free download from the mirror into plain folders. Resumable per chunk; weights verified by sha256.
import hashlib, json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ENDPOINT = "https://hf-mirror.com"
OUT = os.path.expanduser("~/.cache/qwen-all/models")
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CHUNK = 16 << 20
WANT = (".safetensors", ".json", ".txt", ".jinja")
exp = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected_tts.json")))


def read(url, a=None, b=None, tries=30):
    last = None
    for i in range(tries):
        try:
            h = {"Range": f"bytes={a}-{b}"} if a is not None else {}
            with opener.open(urllib.request.Request(url, headers=h), timeout=90) as r:
                data = r.read()
            if a is None or len(data) == b - a + 1:
                return data
            last = IOError("short read")
        except Exception as e:
            last = e
        time.sleep(min(30, 2 + i))
    raise last


def fetch_big(url, path, size, sha):
    if os.path.exists(path) and os.path.getsize(path) == size:
        return check(path, sha)
    part, done_path = path + ".part", path + ".done"
    if not os.path.exists(part):
        with open(part, "wb") as f:
            f.truncate(size)
    done = set(int(x) for x in open(done_path).read().split()) if os.path.exists(done_path) else set()
    chunks = [(i, i * CHUNK, min(size, (i + 1) * CHUNK) - 1) for i in range((size + CHUNK - 1) // CHUNK)]
    lock, t0, got = threading.Lock(), time.time(), [0]

    def work(c):
        i, a, b = c
        if i in done:
            return
        data = read(url, a, b)
        with open(part, "r+b") as f:
            f.seek(a)
            f.write(data)
        with lock:
            done.add(i)
            open(done_path, "a").write(f"{i}\n")
            got[0] += len(data)
            if len(done) % 25 == 0:
                print(f"  {os.path.basename(path)} {len(done)}/{len(chunks)} {got[0]/(time.time()-t0)/1e6:.1f} MB/s", flush=True)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(work, chunks))
    os.replace(part, path)
    os.remove(done_path)
    return check(path, sha)


def check(path, sha):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    good = h.hexdigest() == sha
    print(f"  {os.path.basename(path)} sha256 {'OK' if good else 'MISMATCH'}", flush=True)
    if not good:
        os.remove(path)
    return good


ok = True
for repo, e in exp.items():
    name = repo.split("/")[1]
    info = json.loads(read(f"{ENDPOINT}/api/models/{repo}?blobs=true"))
    print(repo, "revision match:", info["sha"] == e["sha"], flush=True)
    for s in info["siblings"]:
        f = s["rfilename"]
        if not f.endswith(WANT):
            continue
        path = os.path.join(OUT, name, f)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"{ENDPOINT}/{repo}/resolve/{e['sha']}/{f}"
        if f in e["files"]:
            ok &= fetch_big(url, path, s["size"], e["files"][f])
        elif not (os.path.exists(path) and os.path.getsize(path) == s.get("size")):
            open(path, "wb").write(read(url))
print("VERIFY", "OK" if ok else "FAILED", flush=True)
sys.exit(0 if ok else 1)
