# Resumable parallel range download with sha256 verification, for slow links.
# usage: fetch_parallel.py <repo> <filename> <sha256> <out_path> [workers]
import hashlib, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

repo, fname, sha, out = sys.argv[1:5]
workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8
CHUNK = 8 << 20
URLS = [f"https://huggingface.co/{repo}/resolve/main/{fname}", f"https://hf-mirror.com/{repo}/resolve/main/{fname}"]
direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
proxied = urllib.request.build_opener()
ROUTES = [(URLS[0], proxied), (URLS[1], direct)]


def size():
    for url, op in ROUTES:
        try:
            r = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            with op.open(r, timeout=30) as resp:
                return int(resp.headers["Content-Range"].split("/")[-1])
        except Exception as e:
            print("size failed", url, e, flush=True)
    raise SystemExit("no route")


total = size()
os.makedirs(os.path.dirname(out), exist_ok=True)
part, done_path = out + ".part", out + ".done"
if not os.path.exists(part):
    with open(part, "wb") as f:
        f.truncate(total)
chunks = [(i, i * CHUNK, min(total, (i + 1) * CHUNK) - 1) for i in range((total + CHUNK - 1) // CHUNK)]
done = set(int(x) for x in open(done_path).read().split()) if os.path.exists(done_path) else set()
lock = threading.Lock()
stats = {"bytes": 0, "t": time.time(), "routes": [0, 0]}
print(f"total {total/1e9:.2f} GB, {len(chunks)} chunks, {len(done)} already done", flush=True)


def work(c):
    i, a, b = c
    if i in done:
        return
    for attempt in range(40):
        ri = (i + attempt) % 2
        url, op = ROUTES[ri]
        try:
            r = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            with op.open(r, timeout=45) as resp:
                data = resp.read()
            if len(data) != b - a + 1:
                raise IOError("short read")
            with open(part, "r+b") as f:
                f.seek(a)
                f.write(data)
            with lock:
                done.add(i)
                open(done_path, "a").write(f"{i}\n")
                stats["bytes"] += len(data)
                stats["routes"][ri] += 1
                if len(done) % 20 == 0:
                    el = time.time() - stats["t"]
                    print(f"{len(done)}/{len(chunks)} chunks, {stats['bytes']/el/1e6:.2f} MB/s, routes hf/mirror {stats['routes']}", flush=True)
            return
        except Exception:
            time.sleep(1)
    raise IOError(f"chunk {i} failed")


with ThreadPoolExecutor(workers) as ex:
    list(ex.map(work, chunks))
h = hashlib.sha256()
with open(part, "rb") as f:
    for block in iter(lambda: f.read(1 << 24), b""):
        h.update(block)
if h.hexdigest() != sha:
    raise SystemExit(f"SHA MISMATCH {h.hexdigest()}")
os.replace(part, out)
os.remove(done_path)
print("VERIFIED", out, flush=True)
