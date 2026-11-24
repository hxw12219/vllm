# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch
import torch.nn.functional as F

from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor

# Upper bound on the topk kernel's per-iteration gather width.
_MAX_TOPK_BLOCK = 1024


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk_ids_stride,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        # NOTE(woosuk): Make sure that logits and all following operations use FP32.
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    for j in range(0, topk, TOPK_BLOCK_SIZE):
        k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
        k_mask = k_offset < topk
        topk_ids = tl.load(
            topk_ids_ptr + req_idx * topk_ids_stride + k_offset, mask=k_mask, other=0
        )
        logits = tl.load(row_ptr + topk_ids, mask=k_mask)
        logits = logits.to(tl.float32)
        o = logits - max_val - lse
        tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits >= x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    # NOTE(woosuk): To save GPU memory, we do not materialize the full
    # [batch_size, vocab_size] logprobs tensor. The kernel computes
    # max + logsumexp per row and only emits logprobs at `token_ids`.
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    # Cap the kernel's per-iteration width so very large num_logprobs requests
    # stream the gather in bounded-size chunks, avoiding excessive mem use.
    topk_block_size = min(triton.next_power_of_2(num_logprobs), _MAX_TOPK_BLOCK)
    _topk_log_softmax_kernel[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        token_ids.stride(0),
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=1024,  # type: ignore
        TOPK_BLOCK_SIZE=topk_block_size,
    )
    return logprobs


# Vocab slice per program in the fused path. A row's split depends only on the
# vocab size, so its results do not depend on the rest of the batch.
_FUSED_BLOCK_SIZE = 4096
# Each top-k entry costs one max/argmin pass over the slice. Up to k=8 the
# fused path beat torch.topk at every batch size up to 4096 rows on B300; at
# k=20 it fell behind from about 512 rows, so larger k uses torch.topk.
_FUSED_MAX_TOPK = 8


@triton.jit
def _fused_topk_partial_kernel(
    logits_ptr,
    logits_stride,
    sampled_ptr,
    max_ptr,
    sum_ptr,
    count_ptr,
    top_val_ptr,
    top_idx_ptr,
    vocab_size,
    num_blocks,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < vocab_size
    row_ptr = logits_ptr + row * logits_stride
    x = tl.load(row_ptr + offs, mask=valid, other=float("-inf")).to(tl.float32)
    sampled = tl.load(row_ptr + tl.load(sampled_ptr + row)).to(tl.float32)

    out = row * num_blocks + block
    block_max = tl.max(x, axis=0)
    # Zero only -inf entries, so a NaN logit still poisons the row's sum.
    e = tl.where(x == float("-inf"), 0.0, tl.exp(x - block_max))
    tl.store(max_ptr + out, block_max)
    tl.store(sum_ptr + out, tl.sum(e, axis=0))
    tl.store(count_ptr + out, tl.sum((valid & (x >= sampled)).to(tl.int32), axis=0))

    # Top-k of this slice, ties broken by the lowest token id.
    ids = tl.where(valid, offs, vocab_size)
    for k in tl.static_range(TOPK):
        best = tl.max(x, axis=0)
        idx = tl.min(tl.where(x == best, ids, vocab_size), axis=0)
        tl.store(top_val_ptr + out * TOPK + k, best)
        tl.store(top_idx_ptr + out * TOPK + k, idx)
        taken = ids == idx
        x = tl.where(taken, float("-inf"), x)
        ids = tl.where(taken, vocab_size, ids)


@triton.jit
def _fused_topk_merge_kernel(
    logits_ptr,
    logits_stride,
    sampled_ptr,
    max_ptr,
    sum_ptr,
    count_ptr,
    top_val_ptr,
    top_idx_ptr,
    out_ids_ptr,
    out_logprobs_ptr,
    out_ranks_ptr,
    vocab_size,
    num_blocks,
    TOPK: tl.constexpr,
    BLOCKS_P2: tl.constexpr,
    CANDIDATES_P2: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.arange(0, BLOCKS_P2)
    in_row = blocks < num_blocks
    base = row * num_blocks
    block_max = tl.load(max_ptr + base + blocks, mask=in_row, other=float("-inf"))
    block_sum = tl.load(sum_ptr + base + blocks, mask=in_row, other=0.0)
    row_max = tl.max(block_max, axis=0)
    scaled = tl.where(
        block_max == float("-inf"), 0.0, block_sum * tl.exp(block_max - row_max)
    )
    lse = tl.log(tl.sum(scaled, axis=0))
    rank = tl.sum(tl.load(count_ptr + base + blocks, mask=in_row, other=0), axis=0)
    tl.store(out_ranks_ptr + row, rank.to(tl.int64))

    token = tl.load(sampled_ptr + row)
    logit = tl.load(logits_ptr + row * logits_stride + token).to(tl.float32)
    out = row * (TOPK + 1)
    tl.store(out_ids_ptr + out, token.to(tl.int64))
    tl.store(out_logprobs_ptr + out, logit - row_max - lse)
    if TOPK > 0:
        cands = tl.arange(0, CANDIDATES_P2)
        in_cands = cands < num_blocks * TOPK
        vals = tl.load(
            top_val_ptr + base * TOPK + cands, mask=in_cands, other=float("-inf")
        )
        ids = tl.load(
            top_idx_ptr + base * TOPK + cands, mask=in_cands, other=vocab_size
        )
        for k in tl.static_range(TOPK):
            best = tl.max(vals, axis=0)
            idx = tl.min(tl.where(vals == best, ids, vocab_size), axis=0)
            tl.store(out_ids_ptr + out + 1 + k, idx.to(tl.int64))
            tl.store(out_logprobs_ptr + out + 1 + k, best - row_max - lse)
            taken = ids == idx
            vals = tl.where(taken, float("-inf"), vals)
            ids = tl.where(taken, vocab_size, ids)


def _fused_topk_scores(
    logits: torch.Tensor, num_logprobs: int, sampled_token_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Token ids, logprobs and sampled-token ranks from one read of the logits.

    Each program reduces one vocab slice of a row to its max, exp-sum, count of
    logits at or above the sampled token's, and top-k; a second kernel merges
    a row's slices. Column 0 is the sampled token, then the top-k in
    descending order.
    """
    num_tokens, vocab_size = logits.shape
    device = logits.device
    sampled_token_ids = sampled_token_ids.contiguous()
    num_blocks = triton.cdiv(vocab_size, _FUSED_BLOCK_SIZE)
    k = max(num_logprobs, 1)
    block_max = torch.empty(num_tokens, num_blocks, dtype=torch.float32, device=device)
    block_sum = torch.empty_like(block_max)
    block_count = torch.empty(num_tokens, num_blocks, dtype=torch.int32, device=device)
    top_val = torch.empty(num_tokens, num_blocks, k, dtype=torch.float32, device=device)
    top_idx = torch.empty(num_tokens, num_blocks, k, dtype=torch.int32, device=device)
    token_ids = torch.empty(
        num_tokens, num_logprobs + 1, dtype=torch.int64, device=device
    )
    logprobs = torch.empty(
        num_tokens, num_logprobs + 1, dtype=torch.float32, device=device
    )
    ranks = torch.empty(num_tokens, dtype=torch.int64, device=device)
    if num_tokens > 0:
        _fused_topk_partial_kernel[(num_tokens, num_blocks)](
            logits,
            logits.stride(0),
            sampled_token_ids,
            block_max,
            block_sum,
            block_count,
            top_val,
            top_idx,
            vocab_size,
            num_blocks,
            TOPK=num_logprobs,
            BLOCK_SIZE=_FUSED_BLOCK_SIZE,
            num_warps=8,
        )
        _fused_topk_merge_kernel[(num_tokens,)](
            logits,
            logits.stride(0),
            sampled_token_ids,
            block_max,
            block_sum,
            block_count,
            top_val,
            top_idx,
            token_ids,
            logprobs,
            ranks,
            vocab_size,
            num_blocks,
            TOPK=num_logprobs,
            BLOCKS_P2=triton.next_power_of_2(num_blocks),
            CANDIDATES_P2=triton.next_power_of_2(num_blocks * k),
            num_warps=4,
        )
    if num_logprobs == 0:
        # Match the unfused path, which returns the sampled ids themselves.
        token_ids = sampled_token_ids.unsqueeze(-1)
    return token_ids, logprobs, ranks


def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | torch.Tensor | None = None,
    logprob_token_ids_state: "LogprobTokenIdsState | None" = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
    logits_mode: bool = False,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape
    is_tensor = isinstance(cu_num_logits, torch.Tensor)

    if (
        max_per_req_token_ids == 0
        and not logits_mode
        and num_logprobs <= _FUSED_MAX_TOPK
    ):
        logprob_token_ids, scores, token_ranks = _fused_topk_scores(
            logits, num_logprobs, sampled_token_ids
        )
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=scores,
            selected_token_ranks=token_ranks,
            cu_num_generated_tokens=None if is_tensor else cu_num_logits,
            cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
        )

    if max_per_req_token_ids == 0:
        # Fast path: no request asked for custom logprob_token_ids.
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
            logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
        if logits_mode:
            scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids)
    else:
        # Some requests specified logprob_token_ids. Build the [batch_size,
        # 1 + max_cols] token_ids matrix and validity mask on the GPU via a
        # single triton kernel, overriding the topk columns with per-request
        # tokens where applicable.
        assert logprob_token_ids_state is not None
        assert expanded_idx_mapping is not None

        if num_logprobs > 0:
            topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices
            topk_token_ids = topk_token_ids.to(torch.int32)
        else:
            # This tensor just used as an int32 pointer, data not accessed.
            topk_token_ids = logprob_token_ids_state.token_ids.gpu

        num_cols = max(num_logprobs, max_per_req_token_ids)
        logprob_token_ids = sampled_token_ids.new_zeros((batch_size, 1 + num_cols))
        valid_mask = torch.zeros_like(logprob_token_ids, dtype=torch.bool)
        _fill_logprob_token_ids_kernel[(batch_size,)](
            logprob_token_ids,
            logprob_token_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled_token_ids,
            topk_token_ids,
            topk_token_ids.stride(0),
            expanded_idx_mapping,
            logprob_token_ids_state.num_token_ids.gpu,
            logprob_token_ids_state.token_ids.gpu,
            logprob_token_ids_state.token_ids.gpu.stride(0),
            NUM_TOPK=num_logprobs,
            PADDED_COLS=triton.next_power_of_2(num_cols),
        )
        if logits_mode:
            scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids)
        scores = scores.masked_fill(~valid_mask, float("-inf"))

    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
    _ranks_kernel[(batch_size,)](
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        BLOCK_SIZE=8192,  # type: ignore
    )
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=scores,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=None if is_tensor else cu_num_logits,
        cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
    )


@triton.jit
def _fill_logprob_token_ids_kernel(
    # [batch_size, 1 + num_cols]
    out_token_ids_ptr,
    out_token_ids_stride,
    # [batch_size, 1 + num_cols]
    out_valid_mask_ptr,
    out_valid_mask_stride,
    sampled_token_ids_ptr,  # [batch_size]
    topk_indices_ptr,  # [batch_size, NUM_TOPK] (unused when NUM_TOPK == 0)
    topk_indices_stride,
    expanded_idx_mapping_ptr,  # [batch_size] -> req_state_idx
    num_per_req_token_ids_ptr,  # [max_num_reqs]
    per_req_token_ids_ptr,  # [max_num_reqs, MAX_LOGPROB_TOKEN_IDS]
    per_req_token_ids_stride,
    NUM_TOPK: tl.constexpr,
    PADDED_COLS: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    # Column 0: always the sampled token, always valid.
    sampled = tl.load(sampled_token_ids_ptr + batch_idx)
    tl.store(out_token_ids_ptr + batch_idx * out_token_ids_stride, sampled)
    tl.store(out_valid_mask_ptr + batch_idx * out_valid_mask_stride, 1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + batch_idx)
    num_custom = tl.load(num_per_req_token_ids_ptr + req_state_idx)

    col = tl.arange(0, PADDED_COLS)
    tid_base = out_token_ids_ptr + batch_idx * out_token_ids_stride + 1
    mask_base = out_valid_mask_ptr + batch_idx * out_valid_mask_stride + 1

    if num_custom > 0:
        # Override topk with per-request custom tokens.
        src = per_req_token_ids_ptr + req_state_idx * per_req_token_ids_stride
        valid = col < num_custom
    else:
        # Fill with topk indices (no-op when NUM_TOPK == 0).
        src = topk_indices_ptr + batch_idx * topk_indices_stride
        valid = col < NUM_TOPK

    tokens = tl.load(src + col, mask=valid, other=0).to(tl.int64)
    tl.store(tid_base + col, tokens, mask=valid)
    tl.store(mask_base + col, tl.full([PADDED_COLS], 1, tl.int1), mask=valid)


class LogprobTokenIdsState:
    """Per-request override of which token ids' logprobs to return.

    See `SamplingParams.logprob_token_ids`.
    """

    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.num_token_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.token_ids = StagedWriteTensor(
            (max_num_reqs, MAX_LOGPROB_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        token_ids = sampling_params.logprob_token_ids
        if not token_ids:
            self.num_token_ids.np[req_idx] = 0
            return
        n = len(token_ids)
        if n > MAX_LOGPROB_TOKEN_IDS:
            raise ValueError(
                f"Too many logprob_token_ids: {n}. The max is {MAX_LOGPROB_TOKEN_IDS}."
            )
        self.num_token_ids.np[req_idx] = n
        self.token_ids.stage_write(req_idx, 0, token_ids)

    def apply_staged_writes(self) -> None:
        self.num_token_ids.copy_to_uva()
        self.token_ids.apply_write()

    def max_num_token_ids(self, idx_mapping_np: np.ndarray) -> int:
        return int(self.num_token_ids.np[idx_mapping_np].max(initial=0))


def concat_logprob_stashes(
    parts: list[LogprobsTensors], cu_num_generated_tokens: list[int]
) -> LogprobsTensors:
    """Join the logprobs stashed for the requests committing this step.

    Each stash is as wide as the widest logprobs request in the batch at the
    step that request converged, so stashes from different steps can differ
    in width. Pad the narrow ones the way compute_topk_scores pads a mixed
    batch: token id 0 at -inf, which the output processor never reports.
    """
    width = max(p.logprob_token_ids.shape[1] for p in parts)

    def pad(t: torch.Tensor, value: float) -> torch.Tensor:
        return F.pad(t, (0, width - t.shape[1]), value=value)

    return LogprobsTensors(
        logprob_token_ids=torch.cat([pad(p.logprob_token_ids, 0) for p in parts]),
        logprobs=torch.cat([pad(p.logprobs, float("-inf")) for p in parts]),
        selected_token_ranks=torch.cat([p.selected_token_ranks for p in parts]),
        cu_num_generated_tokens=cu_num_generated_tokens,
    )


def masked_stash_write(
    stash: LogprobsTensors,
    slots: torch.Tensor,
    keep: torch.Tensor,
    new: LogprobsTensors,
) -> None:
    """Write `new` ([n * W] rows) into the per-slot stash where `keep`.

    The PyTorch counterpart of DiffusionGemma's `stash_converged_logprobs`.
    """
    n = slots.shape[0]
    pad = stash.logprobs.shape[-1] - new.logprobs.shape[1]
    ids = F.pad(new.logprob_token_ids.to(torch.int64), (0, pad))
    lps = F.pad(new.logprobs, (0, pad), value=float("-inf"))
    ranks = new.selected_token_ranks.to(torch.int64).view(n, -1)
    W = ranks.shape[1]
    for buf, rows in (
        (stash.logprob_token_ids, ids.view(n, W, -1)),
        (stash.logprobs, lps.view(n, W, -1)),
        (stash.selected_token_ranks, ranks),
    ):
        mask = keep.view(n, *[1] * (rows.dim() - 1))
        buf[slots, :W] = torch.where(mask, rows, buf[slots, :W])
