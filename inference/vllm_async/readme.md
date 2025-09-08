# README — Qwen2.5-VL-32B (vLLM) Async Inference with 60 Frames/Prompt

**Status:** Successfully test-run on a single node with **4× NVIDIA A6000** (48 GB) using **`CONCURRENCY = 12`**.
The pipeline reads **pre-extracted frames** (≈60 JPGs per \~30 s segment at 2 FPS), sends **one prompt per feature** with **all 60 images attached**, and writes **raw model replies** to **JSONL** (no post-processing). asking Junhua Huang if you have more question

---

## 0) What this does (one-liner)

Walk each prepared segment folder, attach **60 images** per request to `Qwen/Qwen2.5-VL-32B-Instruct` via **vLLM** (OpenAI-style API), ask your **per-feature prompts**, and save **raw** answers to `answers.raw.jsonl` next to the frames.

---

## 1) Environment setup (conda first)

> We’ll create a clean conda env, install the **FFmpeg binary** via conda, then Python packages via pip.

```bash
# Create & activate a Python 3.10 environment (3.9–3.12 fine)
conda create -n qwen-vllm python=3.10 -y
conda activate qwen-vllm

# Install the FFmpeg *binary* (needed by many extractors/tools)
conda install -c conda-forge ffmpeg -y

# Python packages
pip install "vllm==0.10.1" -U
pip install openai -U

# If your scripts import the ffmpeg Python binding, install THIS:
pip install ffmpeg-python
# (Note: the package is named 'ffmpeg-python', not 'ffmpeg'.)
```

> If any extra package is reported missing later, just `pip install <name>` inside this env.

---

## 2) Prepare data (already done / reference)

If you still need to extract frames (≈60 JPGs/segment) and audio:

```bash
python extract.py \
  --input_dir  /mnt/SSD3/tengyou/seizure_videos/segments/all_dataset \
  --output_dir /mnt/SSD3/junhua/data
```

**Expected layout:**

```
/mnt/SSD3/junhua/data/
  <video_basename>/
    segment_000/
      frames/
        frame_001.jpg
        ...
        frame_060.jpg      # ≈30s × 2 FPS (cap to 60)
      audio.wav            # optional; Qwen2.5-VL ignores audio, it is prepare for the omni model
    segment_001/
      ...
```

> We do **not** decode video in the vLLM client; we read the already-prepared JPGs.

---

## 3) Start the vLLM server (4 GPUs, allow 60 images per prompt)

First grab your server IP (handy if client runs on the same machine):

```bash
export SERVER_IP=$(hostname -I | awk '{print $1}')
echo "SERVER_IP=$SERVER_IP"
# (print again any time)
hostname -I | awk '{print $1}'
```

Then launch vLLM:

```bash
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 60}' \
  --allowed-local-media-path /mnt/SSD3/junhua \
  --mm-processor-kwargs '{"min_pixels": 12544, "max_pixels": 401408}'
```

**What these mean (short & useful):**

* `--tensor-parallel-size 4` → split the 32B model **across 4 GPUs** (A6000×4).
* `--max-model-len 32768` → context window \~32k tokens (model default).
* `--limit-mm-per-prompt '{"image": 60}'` → allow **60 images** in one request.
  *(If JSON form errors in your vLLM build, use: `--limit-mm-per-prompt image=60`.)*
* `--allowed-local-media-path /mnt/SSD3/junhua` → let the server read `file://` images from that tree.
* `--mm-processor-kwargs` → Qwen’s image pre-processor caps per-image pixels, controlling **visual tokens**:

  * `min_pixels = 16×28×28 = 12544` (≈16 visual tokens/image floor).
  * `max_pixels = 401408` (≈**512 tokens/image**). With **60 images**, visual tokens ≈ `60×512 = 30,720` → fits under 32k with room for prompt + answer.
  * **Why needed?** vLLM accepts **multiple images**, not raw video. These caps make the server treat your 60 frames like the HF video pipeline (keeps token budget safe). Functionally equivalent in practice.


---

## 4) Run the async client

Point the client at your server:

```bash
export VLLM_BASE_URL="http://$SERVER_IP:8000/v1"
export VLLM_API_KEY="sk-local-secret"   # any string if auth isn’t enforced
```

From the directory where your async script lives:

```bash
python async_qwen2.5_32b.py
```

**What it does:**

* Walks `/mnt/SSD3/junhua/data/**/segment_*/frames/`
* For each **feature**, sends **60 images first** (fixed order) + the **prompt**
* Writes one line per feature to:

```
 /mnt/SSD3/junhua/data/<video>/segment_xxx/answers.raw.jsonl
```

Example line:

```json
{"feature":"head_turning","raw":"{\"answer\":\"no\",\"justification\":\"...\"}"}
```

* **No** JSON cleaning, no CSV. Pure raw outputs by design.

---

## 5) Recommended defaults & adjustable knobs

* **Concurrency (client):** `CONCURRENCY = 12` on 4×A6000 worked well.
  Try **8–16** for 60-image prompts. Increase gradually; watch VRAM and latency.

* **Max tokens (client):**
  `MAX_TOKENS = 256` is typically enough for *yes/no + short justification*.
  Used 512 for longer answer/ justification; you can drop to **256** to reduce decode time.
  Optionally allow **384** for a few longer prompts (e.g., `tonic`, `clonic`).

* **Images first:** always build messages with **all images first**, then the text prompt.
  This keeps the image prefix **identical** across features in a segment, enabling vLLM V1’s **prefix & preprocessor cache** → faster repeat prompts.

* **Optional warm-up:** send one trivial prompt per segment before fanning out the rest to “prime” caches.

* **Server memory headroom:** if you need more KV space, consider
  `--gpu-memory-utilization 0.90`–`0.92`.

---

## 6) Interpreting server logs

* **Prefix cache hit rate \~90%+** → good; means your image prefix is reused across prompts.
* **GPU KV cache usage near 0% between bursts** → normal; short generations release KV quickly.
* **Avg prompt throughput (tok/s)** → prefill speed (often very high with cache hits).
* **Avg generation throughput (tok/s)** → decode speed; lowering `MAX_TOKENS` improves tail latency.

---

## 7) Common pitfalls (and quick fixes)

* **“pip install ffmpeg” confusion**
  The *binary* comes from `conda install -c conda-forge ffmpeg`.
  The Python binding is `pip install ffmpeg-python`. Use both if needed.

* **`--limit-mm-per-prompt` syntax**
  Use **`'{"image": 60}'`** or **`image=60`** (version-dependent). Avoid `"images"`.

* **Local file access denied**
  Ensure your image paths start with `file://` and are under `--allowed-local-media-path`.

* **Context-length errors**
  Lower `max_pixels` (e.g., 401,408) or reduce images per request (e.g., 32).

* **OOMs**
  Lower `CONCURRENCY`, reduce `MAX_TOKENS`, or reduce `max_pixels`; optionally add `--gpu-memory-utilization 0.90`.

