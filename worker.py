# Corpus HQ free embeddings worker (Supabase/gte-small, same model as the Supabase edge runtime). Runs on GitHub Actions.
import os, time, requests, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
API = "https://lbvaosyfikkpvcwksiph.supabase.co/functions/v1/embed-free"
H = {"x-gte-secret": os.environ["GTE_SECRET"], "Content-Type": "application/json"}
SHARD = int(os.environ["SHARD"]); SHARDS = int(os.environ["SHARDS"]); STOP = time.time() + float(os.environ.get("MAX_SECONDS", "19800"))
tok = Tokenizer.from_file("model/tokenizer.json"); tok.enable_truncation(512); tok.enable_padding()
so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 2
sess = ort.InferenceSession("model/onnx/model.onnx", so, providers=["CPUExecutionProvider"]); names = [i.name for i in sess.get_inputs()]
def embed(texts):
    enc = tok.encode_batch(texts)
    ids = np.array([e.ids for e in enc], dtype=np.int64); mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in names: feed["token_type_ids"] = np.zeros_like(ids)
    out = sess.run(None, feed)[0]; m = mask[..., None].astype(np.float32); v = (out * m).sum(1) / m.sum(1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)
def call(body):
    for a in range(5):
        try:
            r = requests.post(API, headers=H, json=body, timeout=120)
            if r.ok: return r.json()
        except Exception: pass
        time.sleep(5 * (a + 1))
    raise RuntimeError("api failed")
done = 0; t0 = time.time(); empty = 0
while time.time() < STOP:
    table = "chunks"; rows = call({"mode": "claim", "table": "chunks", "limit": 64, "shard": SHARD, "shards": SHARDS})["rows"]
    if not rows: table = "wiki"; rows = call({"mode": "claim", "table": "wiki", "limit": 64, "shard": SHARD, "shards": SHARDS})["rows"]
    if not rows:
        empty += 1
        if empty >= 3: break
        time.sleep(30); continue
    empty = 0; out = []
    for i in range(0, len(rows), 16):
        part = rows[i:i + 16]; vs = embed([r["txt"] or " " for r in part])
        out += [{"id": r["id"], "v": [round(float(x), 6) for x in v]} for r, v in zip(part, vs)]
    call({"mode": "write", "table": table, "rows": out}); done += len(out)
    if done % 1024 < 64: print(f"shard {SHARD}: {done} done, {done / (time.time() - t0):.1f}/s", flush=True)
print(f"shard {SHARD} finished: {done} vectors", flush=True)
