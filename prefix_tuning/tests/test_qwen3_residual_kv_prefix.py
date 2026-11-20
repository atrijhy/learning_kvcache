"""Architecture-level regression checks for the full-attention memory adapter."""
import torch
from pathlib import Path
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM
from prefix_tuning.virtual_prefix.modeling_qwen3_residual_kv_prefix import (
    install_residual_kv_prefix, prefix_state_dict, load_prefix_state_dict,
)


def test_neutral_initialization_gradients_causality_and_reload():
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=96,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, max_position_embeddings=128)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        baseline = model(ids, use_cache=False).logits.clone()
    params = install_residual_kv_prefix(model, 4)
    with torch.no_grad():
        torch.testing.assert_close(model(ids, use_cache=False).logits, baseline, atol=0, rtol=0)
    frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW(params.values(), lr=0.01, weight_decay=0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    for step in range(2):
        optimizer.zero_grad()
        model(ids, labels=ids, use_cache=False).loss.backward()
        for name, p in params.items():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            if step or name.endswith("prefix_value_tokens"):
                assert p.grad.abs().max() > 0
        optimizer.step()
    for name, p in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p, frozen[name], atol=0, rtol=0)
            assert p.grad is None
    model.eval()
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits.clone()
        changed = ids.clone()
        changed[:, 7:] = (changed[:, 7:]+1) % 128
        torch.testing.assert_close(model(changed, use_cache=False).logits[:, :7], expected[:, :7])
        # The residual memory must not enter the request's native KV cache.
        cache = model(ids[:, :7], use_cache=True).past_key_values
        assert cache.get_seq_length() == 7
        incremental = model(ids[:, 7:], past_key_values=cache, use_cache=True).logits
        torch.testing.assert_close(incremental, expected[:, 7:], atol=1e-5, rtol=1e-5)
    state = prefix_state_dict(model)
    with torch.no_grad():
        for p in params.values():
            p.zero_()
    load_prefix_state_dict(model, state)
    with torch.no_grad():
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected, atol=0, rtol=0)


def test_sparse_logits_match_masked_causal_loss():
    from prefix_tuning.virtual_prefix.train_qwen3_prefix_sft import supervised_loss
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=96,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16)
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.randint(0, 128, (1, 12))
    labels = ids.clone()
    labels[:, :8] = -100
    with torch.no_grad():
        dense = model(ids, labels=labels, use_cache=False).loss
        sparse_sum, count = supervised_loss(model, {
            "input_ids": ids[0].tolist(), "labels": labels[0].tolist()}, "cpu")
    assert count == 4
    torch.testing.assert_close(sparse_sum/count, dense)


def test_tool_results_and_history_reasoning_are_masked():
    from transformers import AutoTokenizer
    from prefix_tuning.virtual_prefix.prepare_envscaler_sft import tokenize_turn
    path = Path("/home/liyu.468/models/Qwen3-4B")
    if not path.is_dir():
        pytest.skip("Requires the local Qwen3 tokenizer; never downloads in tests")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    row = tokenize_turn(tokenizer, {
        "task_info": {"task_id": "test"}, "traj_type": "non_conversation", "tools": [],
        "messages": [
            {"role": "system", "content": "Follow the tools."},
            {"role": "user", "content": "Cancel order 123."},
            {"role": "assistant", "reasoning_content": "HISTORY_REASONING_SENTINEL", "content": "",
             "tool_calls": [{"function": {"name": "get_order", "arguments": {"id": 123}}}]},
            {"role": "tool", "content": "TOOL_RESULT_SENTINEL: pending"},
            {"role": "assistant", "reasoning_content": "CURRENT_REASONING_SENTINEL", "content": "",
             "tool_calls": [{"function": {"name": "cancel_order", "arguments": {"id": 123}}}]},
        ]}, 4)
    prompt = tokenizer.decode(row["input_ids"][:row["prompt_tokens"]])
    target = tokenizer.decode(row["labels"][row["prompt_tokens"]:])
    assert "HISTORY_REASONING_SENTINEL" not in prompt
    assert "TOOL_RESULT_SENTINEL" in prompt
    assert "CURRENT_REASONING_SENTINEL" in target and "cancel_order" in target
    assert "TOOL_RESULT_SENTINEL" not in target
    assert all(x == -100 for x in row["labels"][:row["prompt_tokens"]])
