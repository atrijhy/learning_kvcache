"""Reference attention for prefix and state competition."""

import math

import torch
import torch.nn.functional as F


def gdn_state_rows_attention(readout, query, prefix_key, prefix_value, temperature, bias, *, clamp):
    """Add the prefix contribution before the gated RMSNorm.

    Queries and state readouts are [batch, heads, tokens, dim]. The prefix
    tensors are broadcastable [batch, heads, prefix tokens, dim].
    """
    state_logits = readout.float() * temperature.float()[None, :, None, None]
    state_logits = state_logits + bias.float()[None, :, None, None]
    state_log_mass = torch.logsumexp(state_logits, dim=-1, keepdim=True).clamp(*clamp)
    prefix_logits = F.normalize(query.float(), dim=-1) @ F.normalize(
        prefix_key.float(), dim=-1
    ).transpose(-1, -2)
    prefix_logits = prefix_logits * math.sqrt(query.shape[-1])
    weights = torch.softmax(torch.cat((state_log_mass, prefix_logits), dim=-1), dim=-1)
    return readout + (weights[..., 1:] @ prefix_value.float()).to(readout.dtype)


def full_attention_with_prefix(query, key, value, prefix_key, prefix_value, prefix_bias,
                               *, context_mask=None, dropout=0.0, scale=None):
    """Joint causal attention over negative-position prefix and user K/V."""
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[-2]
    groups = heads // key.shape[1]
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        prefix_key = prefix_key.repeat_interleave(groups, dim=1)
        prefix_value = prefix_value.repeat_interleave(groups, dim=1)

    if context_mask is None:
        query_positions = torch.arange(key_length - query_length, key_length, device=query.device)
        key_positions = torch.arange(key_length, device=query.device)
        context_mask = key_positions <= query_positions[:, None]
    if context_mask.dtype == torch.bool:
        context_mask = query.new_zeros(context_mask.shape).masked_fill(~context_mask, -torch.inf)
    context_mask = context_mask.to(dtype=query.dtype).expand(batch, heads, query_length, key_length)
    prefix_length = prefix_key.shape[-2]
    prefix_mask = prefix_bias.to(query.dtype).view(1, heads, 1, 1)
    prefix_mask = prefix_mask.expand(batch, heads, query_length, prefix_length)
    return F.scaled_dot_product_attention(
        query,
        torch.cat((prefix_key.expand(batch, -1, -1, head_dim), key), dim=-2),
        torch.cat((prefix_value.expand(batch, -1, -1, head_dim), value), dim=-2),
        attn_mask=torch.cat((prefix_mask, context_mask), dim=-1),
        dropout_p=dropout,
        scale=scale,
    )
