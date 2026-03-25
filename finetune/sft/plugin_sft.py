"""
Custom SFT loss plugin for SeizureSemiologyBench — task-4 timestamp prediction.

Three ideas combined for task-4 (channel == "task-4"):
  Idea 1 — Digit-token weighting:  give extra CE weight to the 4 MM/SS digit
            positions in the response so errors in the actual time value are
            penalised harder than errors in the surrounding JSON skeleton.
  Idea 2 — Channel CE scaling:     scale task-4's overall CE contribution up
            relative to other tasks (task-4 responses are short; without scaling
            they contribute less gradient per sample).
  Idea 3 — Auxiliary soft-MAE:     compute a differentiable expected timestamp
            (seconds) from the logit distribution at each digit position via a
            soft-argmax, then add a Smooth-L1 term against the ground-truth
            seconds.  This gives explicit gradient signal about temporal distance,
            not just token-level accuracy.

Hyperparameters (tunable via env vars):
  TASK4_CE_SCALE   (default 2.0)   — idea 2: CE multiplier for task-4 samples
  TASK4_DIGIT_W    (default 2.0)   — idea 1: extra weight on digit tokens
  TASK4_MAE_LAMBDA (default 0.05)  — idea 3: weight of soft-MAE term
  TASK4_MAE_BETA   (default 10.0)  — smooth-L1 transition point (seconds)

Usage (in the swift sft bash script):
  --external_plugins /path/to/plugin_sft.py
  --loss_type task4_combined
  --enable_channel_loss True        (keeps per-channel loss logging in wandb)
"""

import os
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Tokenizer-derived constants — populated by _verify_tokenizer() on first call.
#
# Expected layout for Qwen2.5-Omni-7B (verified at runtime, not assumed):
#   digit '0' → token ID 15, '1' → 16, ..., '9' → 24  (contiguous block)
#   response `{"timestamp": "MM:SS"}` → exactly 10 tokens:
#     pos 0: '{"'   pos 1: 'timestamp'  pos 2: '":'  pos 3: ' "'
#     pos 4: M1     pos 5: M2           pos 6: ':'
#     pos 7: S1     pos 8: S2           pos 9: '"}'
# ---------------------------------------------------------------------------
_DIGIT_TOKEN_BASE  = None   # token ID of digit '0'  — set by verify
_DIGIT_IDS         = None   # list of 10 token IDs for '0'..'9'
_DIGIT_VALS        = None   # lazy tensor [0,1,...,9] on the right device
_RESP_TOKEN_LEN    = None   # must be 10
_M1_LABEL_OFFSET   = None   # label index of M1 relative to response start
_M2_LABEL_OFFSET   = None
_S1_LABEL_OFFSET   = None
_S2_LABEL_OFFSET   = None
_DIGIT_LOSS_OFFSETS = None  # = label offsets - 1 (causal-LM shift)

# Seconds contributed by each digit: M1×600 + M2×60 + S1×10 + S2×1
_DIGIT_SECOND_WEIGHTS = [600, 60, 10, 1]

_VERIFIED = False   # guard: run verification exactly once


def _verify_tokenizer(tokenizer):
    """
    Derive and validate all tokenizer-dependent constants using the live
    tokenizer.  Raises RuntimeError immediately if any assumption fails so
    training stops before computing a single wrong loss value.

    Checks:
      1. Each digit '0'–'9' tokenises to exactly one token.
      2. Those 10 token IDs are distinct and contiguous (base, base+1, …, base+9).
      3. The canonical task-4 response '{"timestamp": "MM:SS"}' tokenises to
         exactly 10 tokens with digit tokens at positions 4, 5, 7, 8.
    """
    global _DIGIT_TOKEN_BASE, _DIGIT_IDS, _RESP_TOKEN_LEN
    global _M1_LABEL_OFFSET, _M2_LABEL_OFFSET, _S1_LABEL_OFFSET, _S2_LABEL_OFFSET
    global _DIGIT_LOSS_OFFSETS, _VERIFIED

    # ── Check 1: each digit is a single token ────────────────────────────────
    digit_ids = []
    for d in range(10):
        ids = tokenizer.encode(str(d), add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"[plugin_sft] Digit '{d}' tokenises to {len(ids)} tokens {ids}, "
                f"expected exactly 1. The digit-weighting and soft-MAE loss "
                f"cannot work with multi-token digits. Check your tokenizer."
            )
        digit_ids.append(ids[0])

    # ── Check 2: token IDs are distinct and contiguous ───────────────────────
    if len(set(digit_ids)) != 10:
        raise RuntimeError(
            f"[plugin_sft] Digit token IDs are not all distinct: {digit_ids}. "
            f"Cannot build a reliable digit→value mapping."
        )
    base = min(digit_ids)
    expected = list(range(base, base + 10))
    if sorted(digit_ids) != expected:
        raise RuntimeError(
            f"[plugin_sft] Digit token IDs {sorted(digit_ids)} are not "
            f"contiguous starting from {base}. Soft-argmax requires a "
            f"contiguous block so digit value = token_id - base."
        )
    # Verify ordering: digit_ids[d] == base + d
    for d in range(10):
        if digit_ids[d] != base + d:
            raise RuntimeError(
                f"[plugin_sft] digit_ids[{d}]={digit_ids[d]} but expected "
                f"{base + d}. Digits are not ordered '0','1',...,'9'."
            )

    # ── Check 3: canonical response structure ─────────────────────────────────
    # Use a known timestamp to probe the token layout
    probe_ts  = "01:23"
    probe_str = f'{{"timestamp": "{probe_ts}"}}'
    probe_ids = tokenizer.encode(probe_str, add_special_tokens=False)

    if len(probe_ids) != 10:
        raise RuntimeError(
            f"[plugin_sft] '{probe_str}' tokenises to {len(probe_ids)} tokens "
            f"{[tokenizer.decode([t]) for t in probe_ids]}, expected 10. "
            f"The response structure has changed; update the offset constants."
        )

    # Expected digit values for probe_ts "01:23": M1=0,M2=1,S1=2,S2=3
    probe_digits = {'M1': 0, 'M2': 1, 'S1': 2, 'S2': 3}
    # Find which positions hold those digit tokens
    found = {}
    for pos, tid in enumerate(probe_ids):
        if tid in digit_ids:
            digit_val = tid - base
            for name, val in probe_digits.items():
                if digit_val == val and name not in found:
                    found[name] = pos
                    break

    required = {'M1', 'M2', 'S1', 'S2'}
    if found.keys() != required:
        raise RuntimeError(
            f"[plugin_sft] Could not locate all digit positions in probe "
            f"response '{probe_str}'. "
            f"Tokens: {[tokenizer.decode([t]) for t in probe_ids]}. "
            f"Found positions: {found}."
        )

    m1_off, m2_off = found['M1'], found['M2']
    s1_off, s2_off = found['S1'], found['S2']

    # Sanity: colon between MM and SS must sit between M2 and S1
    colon_id = tokenizer.encode(':', add_special_tokens=False)
    if len(colon_id) == 1 and probe_ids[m2_off + 1] != colon_id[0]:
        raise RuntimeError(
            f"[plugin_sft] Expected ':' token between M2 (pos {m2_off}) and "
            f"S1 (pos {s1_off}), but found token "
            f"'{tokenizer.decode([probe_ids[m2_off + 1]])}' at pos {m2_off+1}."
        )

    # ── Commit verified constants ─────────────────────────────────────────────
    _DIGIT_TOKEN_BASE   = base
    _DIGIT_IDS          = digit_ids
    _RESP_TOKEN_LEN     = len(probe_ids)
    _M1_LABEL_OFFSET    = m1_off
    _M2_LABEL_OFFSET    = m2_off
    _S1_LABEL_OFFSET    = s1_off
    _S2_LABEL_OFFSET    = s2_off
    _DIGIT_LOSS_OFFSETS = [m1_off - 1, m2_off - 1, s1_off - 1, s2_off - 1]

    _VERIFIED = True
    print(
        f"[plugin_sft] Tokenizer verified ✓\n"
        f"  digit base token ID : {_DIGIT_TOKEN_BASE}\n"
        f"  digit token IDs     : {_DIGIT_IDS}\n"
        f"  response token len  : {_RESP_TOKEN_LEN}\n"
        f"  label offsets M1/M2/S1/S2: "
        f"{_M1_LABEL_OFFSET}/{_M2_LABEL_OFFSET}/{_S1_LABEL_OFFSET}/{_S2_LABEL_OFFSET}\n"
        f"  loss offsets  M1/M2/S1/S2: {_DIGIT_LOSS_OFFSETS}"
    )


# ---------------------------------------------------------------------------
# Monkey-patch: make channels available inside compute_loss_func
# ---------------------------------------------------------------------------
# The swift trainer pops "channel" from inputs inside compute_loss() BEFORE
# calling compute_loss_func.  We peek at it first via a wrapper and store it
# on the trainer instance so our loss function can read trainer._current_channels.

def _install_channel_forwarding():
    from swift.trainers.trainers import Seq2SeqTrainer
    _orig = Seq2SeqTrainer.compute_loss

    def _patched(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Peek before the original pops it; doesn't consume the key
        self._current_channels = inputs.get('channel', None)
        return _orig(self, model, inputs,
                     return_outputs=return_outputs,
                     num_items_in_batch=num_items_in_batch)

    Seq2SeqTrainer.compute_loss = _patched


_install_channel_forwarding()


# ---------------------------------------------------------------------------
# Helper: per-sample format check
# ---------------------------------------------------------------------------
def _task4_format_ok(s_label, r, seq_len):
    """
    Return True only if this sample's label tokens at the four digit positions
    are all genuine digit tokens.  Called per-sample every step; if it returns
    False the MAE term is silently skipped for that sample (CE still applies).

    Checks:
      - The full 10-token response fits within seq_len (not truncated).
      - The tokens at M1, M2, S1, S2 label offsets are all in _DIGIT_IDS.
    """
    if r + _S2_LABEL_OFFSET >= seq_len:
        return False   # response was truncated
    for offset in (_M1_LABEL_OFFSET, _M2_LABEL_OFFSET, _S1_LABEL_OFFSET, _S2_LABEL_OFFSET):
        if s_label[r + offset].item() not in _DIGIT_IDS:
            return False   # unexpected token at a digit position
    return True


# ---------------------------------------------------------------------------
# Helper: soft digit expectation at a single logit position
# ---------------------------------------------------------------------------
def _soft_digit(logits_at_pos, digit_vals):
    """Return E[digit] under softmax over the 10 digit token positions."""
    digit_logits = logits_at_pos[_DIGIT_IDS]          # [10]
    probs = torch.softmax(digit_logits.float(), dim=-1)
    return (probs * digit_vals).sum()                  # scalar


# ---------------------------------------------------------------------------
# Main custom loss
# ---------------------------------------------------------------------------
def task4_combined_loss(outputs, labels, num_items_in_batch=None, trainer=None, **kwargs):
    """
    Combined CE + soft-MAE loss with channel-aware weighting.
    Registered as loss_type='task4_combined'.
    """
    from swift.trainers.utils import per_token_loss_func

    global _DIGIT_VALS, _VERIFIED

    # ── One-time tokenizer verification ──────────────────────────────────────
    # Runs on the very first training step using the live tokenizer so we fail
    # fast with a clear message rather than silently computing a wrong loss.
    if not _VERIFIED:
        if trainer is None:
            raise RuntimeError("[plugin_sft] trainer=None on first call; cannot verify tokenizer.")
        tokenizer = getattr(trainer, 'processing_class', None) \
                 or getattr(trainer, 'tokenizer', None)
        if tokenizer is None:
            raise RuntimeError("[plugin_sft] Cannot find tokenizer on trainer object.")
        _verify_tokenizer(tokenizer)   # raises RuntimeError if anything is wrong
    # ─────────────────────────────────────────────────────────────────────────

    device = outputs.logits.device

    # Lazy-init digit value tensor (must be on the same device as logits)
    if _DIGIT_VALS is None or _DIGIT_VALS.device != device:
        _DIGIT_VALS = torch.arange(10, dtype=torch.float32, device=device)

    # Read hyperparameters from env (allows tuning without code changes)
    ce_scale    = float(os.environ.get('TASK4_CE_SCALE',   '2.0'))
    digit_w     = float(os.environ.get('TASK4_DIGIT_W',    '2.0'))
    mae_lambda  = float(os.environ.get('TASK4_MAE_LAMBDA', '0.05'))
    mae_beta    = float(os.environ.get('TASK4_MAE_BETA',   '10.0'))

    # Per-token CE loss (flat, unreduced): shape [B * seq_len]
    # per_token_loss_func internally does roll(-1) so loss[p]=CE(logits[p],labels[p+1])
    token_loss = per_token_loss_func(outputs, labels)   # [B*seq]

    channels = getattr(trainer, '_current_channels', None)

    # ----- fallback: no channel info → plain CE -----
    if channels is None:
        n = num_items_in_batch if num_items_in_batch is not None \
            else (labels[:, 1:] != -100).sum()
        return token_loss.sum() / n

    B, seq_len = labels.shape

    # Valid-token mask (same roll as per_token_loss_func)
    valid_flat = (torch.roll(labels, shifts=-1, dims=-1).view(-1) != -100)  # [B*seq]

    total_ce   = torch.tensor(0.0, device=device, requires_grad=True) + 0.0
    total_w    = torch.tensor(0.0, device=device)
    mae_terms  = []

    for i, channel in enumerate(channels):
        start = i * seq_len
        end   = start + seq_len
        s_loss  = token_loss[start:end]        # [seq_len]
        s_valid = valid_flat[start:end]        # [seq_len]
        s_label = labels[i]                    # [seq_len]

        if not s_valid.any():
            continue

        if channel == 'task-4':
            # Find first non-masked position (start of assistant response)
            resp_positions = (s_label != -100).nonzero(as_tuple=False)
            if len(resp_positions) == 0:
                continue
            r = resp_positions[0, 0].item()

            # ---- Idea 1: build per-position weight vector ----
            weights = torch.ones(seq_len, dtype=torch.float32, device=device)
            for offset in _DIGIT_LOSS_OFFSETS:
                p = r + offset
                if 0 <= p < seq_len and s_valid[p]:
                    weights[p] = digit_w

            # Weighted CE for this sample
            w_masked = weights * s_valid.float()
            sample_ce = (s_loss * w_masked).sum()
            sample_n  = w_masked.sum()

            # ---- Idea 2: scale task-4 CE contribution ----
            total_ce = total_ce + ce_scale * sample_ce
            total_w  = total_w  + ce_scale * sample_n

            # ---- Idea 3: soft-MAE on timestamp digits ----
            # Only applied when the label tokens at the digit positions are
            # genuine digit tokens.  Silently skipped otherwise (CE still runs).
            if _task4_format_ok(s_label, r, seq_len):
                s_logits = outputs.logits[i]  # [seq_len, vocab]
                dv = _DIGIT_VALS

                # Logit position for predicting each digit = label_offset - 1
                exp_m1 = _soft_digit(s_logits[r + _M1_LABEL_OFFSET - 1], dv)
                exp_m2 = _soft_digit(s_logits[r + _M2_LABEL_OFFSET - 1], dv)
                exp_s1 = _soft_digit(s_logits[r + _S1_LABEL_OFFSET - 1], dv)
                exp_s2 = _soft_digit(s_logits[r + _S2_LABEL_OFFSET - 1], dv)

                exp_sec = (exp_m1 * _DIGIT_SECOND_WEIGHTS[0]
                           + exp_m2 * _DIGIT_SECOND_WEIGHTS[1]
                           + exp_s1 * _DIGIT_SECOND_WEIGHTS[2]
                           + exp_s2 * _DIGIT_SECOND_WEIGHTS[3])

                # GT seconds from label token IDs  (token_id - base = digit value)
                gt_m1 = s_label[r + _M1_LABEL_OFFSET].item() - _DIGIT_TOKEN_BASE
                gt_m2 = s_label[r + _M2_LABEL_OFFSET].item() - _DIGIT_TOKEN_BASE
                gt_s1 = s_label[r + _S1_LABEL_OFFSET].item() - _DIGIT_TOKEN_BASE
                gt_s2 = s_label[r + _S2_LABEL_OFFSET].item() - _DIGIT_TOKEN_BASE
                gt_sec = float(gt_m1 * _DIGIT_SECOND_WEIGHTS[0]
                               + gt_m2 * _DIGIT_SECOND_WEIGHTS[1]
                               + gt_s1 * _DIGIT_SECOND_WEIGHTS[2]
                               + gt_s2 * _DIGIT_SECOND_WEIGHTS[3])

                gt_tensor = torch.tensor(gt_sec, dtype=torch.float32, device=device)
                mae_terms.append(
                    F.smooth_l1_loss(exp_sec, gt_tensor, beta=mae_beta, reduction='mean')
                )

        else:
            # Non-task-4: standard CE, weight = 1
            total_ce = total_ce + (s_loss * s_valid.float()).sum()
            total_w  = total_w  + s_valid.float().sum()

    # Normalise CE
    denom = total_w if total_w > 0 else torch.tensor(1.0, device=device)
    loss = total_ce / denom

    # Add MAE term
    if mae_terms:
        mae_loss = torch.stack(mae_terms).mean()
        loss = loss + mae_lambda * mae_loss

    return loss


# ---------------------------------------------------------------------------
# Register in swift's loss_mapping so --loss_type task4_combined works
# ---------------------------------------------------------------------------
from swift.plugin import loss_mapping
loss_mapping['task4_combined'] = task4_combined_loss
