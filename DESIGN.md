# tiny-swap-bench — design note (pre-scaffold)

This document is the **sign-off gate** before code lands. After you approve it (or request edits), the repo will be scaffolded to match.

---

## (a) Reference architecture — dimensions for ~20M **non-embedding** parameters

**Definition (fixed for all reporting):**  
**Non-embedding params** = every trainable tensor **except** token embedding `wte` (shape `(V, D)`) and positional encoding parameters **if any** (e.g. learned absolute `wpe` of shape `(T_max, D)`).  
Output head uses **tied weights** with `wte`, so no separate `lm_head`.  
A final **LayerNorm** (`ln_f`, shape `(D,)`) is included in non-embedding.

**Block parameterization (GPT-style, pre-LayerNorm in the reference):**  
Per layer `ℓ`:

- `ln_1`, `ln_2`: LayerNorm on `D` → **2×(2D) = 4D** (weight + bias each).
- Attention: fused QKV `(D → 3D)` + out proj `(D → D)` → **3D² + D² = 4D²**.
- MLP: up `(D → 4D)` + down `(4D → D)` → **4D² + 4D² = 8D²**.

So per layer: **12D² + 4D**.  
Across `L` layers: **L·(12D² + 4D)**.  
Plus final `ln_f`: **+2D**.

**Chosen reference (round numbers, ~20M):**

| Symbol | Value |
|--------|--------|
| `L` (layers) | **11** |
| `D` (model dim) | **384** |
| `H` (heads) | **6** → `D % H == 0`, head dim **64** |
| MLP expansion | **4×** → inner **1536** |
| `T` (sequence length) | **512** |
| Vocab `V` | **50257** (GPT-2 BPE) |

**Exact non-embedding count (reference, pre-LN + LayerNorm blocks + final LN):**

```
L * (12*D² + 4*D) + 2*D
= 11 * (12 * 384² + 4 * 384) + 768
= 11 * (1,769,472 + 1,536) + 768
= 11 * 1,771,008 + 768
= 19,481,088 + 768
= 19,481,856
```

**≈ 19.48M** non-embedding parameters (within the requested “~20M” band).

**Embedding side (reported separately, not in the 20M figure):**  
- `wte`: `50257 × 384` (tied to logits).

**Positional encoding params (by variant, excluded from “20M” if they are pure embedding tables):**

| PE variant | Extra trainable params |
|------------|-------------------------|
| Learned absolute | `T_max × D` (typically `512 × 384`) |
| RoPE | **0** (rotary inv_freq buffer can be non-trainable) |
| NoPE | **0** |
| ALiBi | **0** (fixed slopes) |

Norm / optimizer swaps change parameter counts slightly (e.g. RMSNorm uses one vector per norm vs LayerNorm’s two). The scaffold will **print exact counts per run** in the resolved config so comparisons stay honest.

---

## (b) Pluggable swap axes (v1)

**Principle:** One axis changes per experiment; everything else matches `configs/base.yaml` + the relevant override file. No registry pattern until ≥3 options need discovery; **explicit imports** + a small **string-to-constructor** map in one place (training entry) is acceptable.

### 1. Positional encoding (`src/model/pe_*.py` — one file per module)

- **Interface:** After token embedding, PE receives `x` with shape **`(B, T, D)`** and returns **`(B, T, D)`**.  
- **Learned absolute:** add `wpe` **`(T_max, D)`** to positions `0 … T-1` (slice broadcast).  
- **RoPE:** apply rotary to **Q/K** inside attention with head dim `D/H` (standard split-head layout). No `wpe`.  
- **NoPE:** identity on positions (model sees order only via causal mask + local structure).  
- **ALiBi:** bias added to attention logits **`(H, T, T)`** or applied via broadcast; no `wpe`.

**Wiring:** `ReferenceTransformer` (name TBD) takes `pe: Literal["learned-abs","rope","nope","alibi"]` (or enum) and builds the matching module. RoPE/ALiBi touch attention internals; learned/NoPE stay outside MHA.

### 2. Normalization (`src/model/norm_*.py` + block layout)

Variants:

| ID | Layout |
|----|--------|
| `pre_ln_ln` | Pre-LN, LayerNorm in block + final LN |
| `pre_ln_rms` | Pre-LN, RMSNorm in block + final RMSNorm |
| `post_ln_ln` | Post-LN ordering (norm placement per canonical post-LN GPT layout), LayerNorm |
| `pre_ln_rms_qk` | Pre-LN + RMSNorm + **QK-norm** (normalize Q/K projections per head before RoPE/attention; compatible with RoPE when both selected) |

**Pluggability:** A **`DecoderBlock`** (or composable `forward`) is constructed from `(norm_factory, attention_fn, mlp, use_qk_norm: bool, post_vs_pre flags)`. Shared tensor shapes are documented in-line.

### 3. Optimizer (`src/optim/{adamw,lion,muon}.py`)

- **`build_optimizer(name, model_parameters, cfg) -> torch.optim.Optimizer`** (or a thin wrapper).  
- **AdamW:** `torch.optim.AdamW` with config-matched hyperparameters.  
- **Lion:** implemented in-repo (paper equations), **no extra dependency**.  
- **Muon:** implemented in-repo from a **minimal, cited reference** (update rule on 2D weight matrices + standard rule on vectors/biases), **no extra dependency** unless you explicitly approve one later.

---

## (c) FLOP counting — analytical formula (training will report)

**Goal:** Per forward pass (per step), report **exact analytical FLOPs** as implemented, then aggregate to **FLOPs-to-epoch** and **FLOP-matched checkpoint eval**.

**Convention:** Count **real multiply-adds** as **one FLOP** (standard in many ML FLOP counts); matmul `(M×K)·(K×N)` → **2·M·N·K** FLOPs.

**Per layer, sequence length `T`, hidden `D`, heads `H`, head dim `d_h = D/H`, MLP inner `4D`:**

1. **LayerNorm / RMSNorm** (forward): ~**2D per token per norm** applied (dominated by variance/mean ops); included as a small explicit term so it does not vanish next to matmuls.

2. **Attention — projections**  
   - Fused QKV: `(T, D) × (D, 3D)` → **2·T·D·3D = 6·T·D²**  
   - Out proj: `(T, D) × (D, D)` → **2·T·D²**  
   Subtotal: **8·T·D²**

3. **Attention — logits & softmax-weighted values**  
   - `QK^T`: for each head, `(T, d_h) × (d_h, T)` → **2·T²·d_h` per head → **2·T²·D** total  
   - `PV`: **2·T²·D**  
   Subtotal: **4·T²·D** (ALiBi adds only cheap bias adds; RoPE rotates Q/K — same order as QKV; counted in projection/rope helper as small fixed terms)

4. **MLP**  
   - Up: **2·T·D·4D = 8·T·D²**  
   - Down: **8·T·D²**  
   Subtotal: **16·T·D²**

**Approximate dominant per-layer forward FLOPs:**

\[
\text{FLOPs}_{\text{layer}} \approx 8T D^{2} + 4T^{2} D + 16T D^{2} = 24\,T D^{2} + 4T^{2} D
\]

**Full model (forward):** `L × FLOPs_layer + embedding lookups (neglected or tiny) + lm_head if untied (here tied → no extra)` + final norm.

**Training step:** **3× forward FLOPs** (forward + backward ≈ 2× forward in common approximation) unless you prefer reporting **forward-only**; the code will document which convention is logged. *Recommendation:* log **forward FLOPs per token** and **training FLOPs = 3 × forward** for weight grads (activation checkpointing off for v1).

**“6 × N × D” check:** For linear-heavy stacks, total matmul FLOPs often scale like **O(N_params × sequence)** with problem-dependent constants; we will **not** rely on this as the primary definition. Instead, the **line-by-line analytical** sum above (plus any QK-norm / RoPE terms explicitly added) is the source of truth, with a **unit test** on a **2-layer toy config** hand-derived to **±2%**.

**Matched-FLOP eval:** Store cumulative FLOPs each step; at eval time, select checkpoint whose cumulative train FLOPs is **closest without exceeding** the budget FLOP count (config defines budget = FLOPs for **1B tokens** at the chosen batch).

---

## (d) Decisions for your sign-off

1. **Exact reference:** **L=11, D=384, H=6** giving **19,481,856** non-embedding params — OK, or do you want a slightly larger pair (e.g. **L=12, D=372**) to land closer to 20.0M?

2. **Non-embedding definition:** Confirm **exclude `wte` + any `wpe`**, include **`ln_f`**, **tied head** — OK?

3. **FLOP convention for “matched FLOPs”:** Use **3× forward FLOPs per training step** as the training budget counter — OK, or forward-only?

4. **Post-LN block:** Implement **post-LN GPT-2 style** (norm after sublayer, residual on branch) with the same residual stream width — OK?

5. **QK-norm scope:** Apply RMSNorm (or LN) to **Q and K** after projection, **per head**, before RoPE when PE is RoPE — OK?

6. **Validation split:** **1% hold-out** from TinyStories-v2, **deterministic** split (e.g. `hash(example_id) % 100 == 0` or dataset order with fixed seed) — OK?

7. **W&B:** Logged **if `WANDB_API_KEY` set**; otherwise JSONL + stdout only — OK?

8. **`experiments/baseline_reproducibility/run.py`:** Sequential **three runs** (seeds 0,1,2), each **full 1B-token budget** — expected **~75–105 min** on one H100; confirm you want this for pipeline validation (vs a smaller smoke flag for dev only).

---

## Next step (after your approval)

1. Environment check (`nvidia-smi`, `torch.cuda`).  
2. Scaffold `src/`, `configs/`, `experiments/`, `tests/`, `README.md`.  
3. CPU tests (including overfit-one-batch, PE/norm/optim smoke, FLOP counter ±2%, YAML load).  
4. Run **`experiments/baseline_reproducibility/run.py`** on GPU and paste the **results table**.

---

**Please reply with approval or edits** (especially items 1–8 in §d). No code will be generated until you give the go-ahead.
