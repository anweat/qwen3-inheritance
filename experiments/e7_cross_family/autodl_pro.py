# AutoDL 容器实例 Pro API client for E7 (docs: https://www.autodl.com/docs/instance_pro_api/).
# Token: put the developer token (console -> 账号 -> 设置 -> 开发者Token) in ~/.config/autodl/token (chmod 600),
# or in $AUTODL_TOKEN. This script never prints the token or the root password.
#   python autodl_pro.py list
#   python autodl_pro.py grab [--max-gb 32] [--tries 120] [--wait 30]   # create on the cheapest spec that has stock
#   python autodl_pro.py ssh <id>        # writes Host autodl-e7 into ~/.ssh/config (key login only)
#   python autodl_pro.py status|on|off|release <id>
# The API cannot power on in no-GPU mode (docs: "暂不支持API以无卡模式开机"); every create/on is billed with a GPU.
# Key login: the start command appends ~/.ssh/autodl_ed25519.pub to root's authorized_keys at every boot.
import json, os, re, sys, time, urllib.parse, urllib.request

HOST = "https://api.autodl.com"
KEY = os.path.expanduser("~/.ssh/autodl_ed25519")
# Spec IDs from the docs appendix, ordered by GPU memory (cost grows with memory).
SPECS = [("4090D", 24), ("v-32g-p", 32), ("5090-p", 32), ("v-48g", 48), ("v-48g-350w", 48), ("pro6000-p", 96), ("h800", 80)]
IMAGE = "base-image-l2t43iu6uk"  # cuda11.8 ubuntu20.04 torch2.0.0; E7 builds its own venv with uv
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance.json")


def token():
    t = os.environ.get("AUTODL_TOKEN")
    if not t:
        p = os.path.expanduser("~/.config/autodl/token")
        if not os.path.exists(p):
            sys.exit("no token: put it in ~/.config/autodl/token or $AUTODL_TOKEN")
        t = open(p).read().strip()
    return t


def call(method, path, body):
    if method == "GET":  # the GET endpoints reject a JSON body; send the fields as query parameters
        path, data = path + "?" + urllib.parse.urlencode(body), None
    else:
        data = json.dumps(body).encode()
    req = urllib.request.Request(HOST + path, data=data, method=method,
                                 headers={"Authorization": token(), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.loads(r.read())
    if out.get("code") != "Success":
        raise RuntimeError(f"{path}: {out.get('code')} {out.get('msg')}")
    return out["data"]


def start_command():
    pub = open(KEY + ".pub").read().strip()
    return (f"mkdir -p /root/.ssh && chmod 700 /root/.ssh && grep -qF '{pub}' /root/.ssh/authorized_keys 2>/dev/null "
            f"|| echo '{pub}' >> /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys")


def cmd_list():
    d = call("POST", "/api/v1/dev/instance/pro/list", {"page_index": 1, "page_size": 20})
    for x in d["list"]:
        print(x["uuid"], x["status"], x.get("start_mode"), x.get("gpu_spec_uuid"), x.get("region_name"), x.get("name"))


def cmd_grab(max_gb=32, tries=120, wait=30, disk=20):
    specs = [s for s, gb in SPECS if gb <= max_gb]
    for t in range(tries):
        for s in specs:
            try:
                iid = call("POST", "/api/v1/dev/instance/pro/create", {
                    "req_gpu_amount": 1, "expand_system_disk_by_gb": disk, "gpu_spec_uuid": s, "image_uuid": IMAGE,
                    "cuda_v_from": 124, "instance_name": "e7-cross-family", "start_command": start_command()})
            except Exception as e:  # out of stock and other refusals
                print(f"try {t} {s}: {e}", flush=True)
                continue
            json.dump(dict(id=iid, spec=s, created=time.strftime("%F %T")), open(STATE, "w"))
            print("created", iid, "on", s, flush=True)
            return iid
        time.sleep(wait)
    sys.exit("no stock within the try budget")


def cmd_ssh(iid):
    for _ in range(60):
        if call("GET", "/api/v1/dev/instance/pro/status", {"instance_uuid": iid}) == "running":
            break
        time.sleep(10)
    d = call("GET", "/api/v1/dev/instance/pro/snapshot", {"instance_uuid": iid})
    host, port = d["proxy_host"], d["ssh_port"]
    print(d.get("snapshot_gpu_alias_name"), "price", d.get("payg_price"), "(API units)", host, port)
    cfg = os.path.expanduser("~/.ssh/config")
    s = open(cfg).read() if os.path.exists(cfg) else ""
    s = re.sub(r"\n?Host autodl-e7\n(?:[ \t]+.*\n?)*", "\n", s)
    s = s.rstrip("\n") + f"\n\nHost autodl-e7\n    HostName {host}\n    Port {port}\n    User root\n    IdentityFile {KEY}\n    IdentitiesOnly yes\n    ServerAliveInterval 30\n"
    open(cfg, "w").write(s)
    print("ssh autodl-e7 ready")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    op = a[0]
    if op == "list":
        cmd_list()
    elif op == "grab":
        kw = dict(zip(a[1::2], a[2::2]))
        cmd_grab(int(kw.get("--max-gb", 32)), int(kw.get("--tries", 120)), int(kw.get("--wait", 30)))
    elif op == "ssh":
        cmd_ssh(a[1])
    elif op in ("status", "on", "off", "release"):
        path = dict(status="status", on="power_on", off="power_off", release="release")[op]
        body = {"instance_uuid": a[1]}
        if op == "on":
            body.update(payload="gpu", start_command=start_command())
        print(call("GET" if op == "status" else "POST", f"/api/v1/dev/instance/pro/{path}", body))
    else:
        sys.exit(f"unknown op {op}")
