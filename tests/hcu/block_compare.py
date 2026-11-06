"""H3 output/gradient parity against the pinned, unmodified upstream source."""
import importlib.util
import json
from pathlib import Path
from types import MethodType

import torch

from diffsynth.models import minimax_h3_dit as h3
from diffsynth.models import minimax_h3_tensor_ops as tensor_ops
from diffsynth_hcu.attention import get_attention_info
from diffsynth_hcu.ops import get_lightop_info


def reference_module():
    spec = importlib.util.spec_from_file_location(
        "diffsynth.models._hcu_upstream_reference", Path(__file__).with_name("reference_h3.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_blocks(device="cpu", dtype=torch.float32, *, compiled=False):
    if device != "cpu" and torch.version.hip is None:
        raise RuntimeError("HCU block optimization validation requires a HIP torch build")
    reference = reference_module()
    previous_compiled = dict(tensor_ops._COMPILED)
    previous_calls = dict(tensor_ops._CALLS)
    previous_options = tensor_ops._OPTIONS
    if compiled:
        tensor_ops.configure_minimax_h3_tensor_compile()
    try:
        for layout in ("standard", "comfy"):
            before_attention = get_attention_info()["calls"]
            before_lightop = get_lightop_info()["kernel_calls"]
            before_compile = dict(tensor_ops._CALLS)
            torch.manual_seed(19)
            # Keep the adapter's validated HCU test shape and tolerances.
            width, heads, head_dim, ffn, time_dim, tokens = (512, 4, 128, 1024, 128, 64) if device != "cpu" else (32, 2, 16, 64, 8, 7)
            options = (width, heads, head_dim, ffn, time_dim, 6 * width * 3, 1e-5, 1e-5)
            expected_block = reference.MiniMaxH3DiTBlock(*options).to(device, dtype)
            actual_block = h3.MiniMaxH3DiTBlock(*options).to(device, dtype)
            actual_block.load_state_dict(expected_block.state_dict(), strict=True)
            assert dict((n, p.shape) for n, p in actual_block.named_parameters()) == dict((n, p.shape) for n, p in expected_block.named_parameters())
            if layout == "comfy":
                def comfy_forward(module, x, *, rope_freqs, cu_seqlens, max_seqlen=None):
                    q, k, v = module.qkv_proj(x).view(x.shape[0], 3, heads, head_dim).unbind(1)
                    q, k = module.q_norm(q), module.k_norm(k)
                    if rope_freqs is not None:
                        q, k = reference._apply_rope(q, rope_freqs), reference._apply_rope(k, rope_freqs)
                    out = reference._sdpa_varlen_attention(q, k, v, cu_seqlens, module.softmax_scale)
                    return module.out_proj(out.reshape(x.shape[0], heads * head_dim))
                expected_block.attn.forward = MethodType(comfy_forward, expected_block.attn)
                actual_block.attn.qkv_layout = "comfy"
            first_stop = 17 if tokens == 64 else 3
            inputs = dict(
                x=torch.randn(tokens, width, device=device, dtype=dtype, requires_grad=True),
                t_emb=torch.randn(2, time_dim, device=device, dtype=dtype, requires_grad=True),
                combined_indices=torch.arange(tokens, device=device) % 6,
                rope_freqs=torch.randn(tokens, head_dim, device=device),
                cu_seqlens=torch.tensor([0, first_stop, tokens], dtype=torch.int32, device=device),
                max_seqlen=tokens - first_stop,
            )
            actual_inputs = {key: value.detach().clone().requires_grad_(value.requires_grad) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
            expected = expected_block(**inputs)
            actual = actual_block(**actual_inputs)
            expected.float().square().mean().backward()
            actual.float().square().mean().backward()
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.02)
            for key in ("x", "t_emb"):
                torch.testing.assert_close(actual_inputs[key].grad, inputs[key].grad, rtol=0.05, atol=0.0002, msg=key)
            for name, param in actual_block.named_parameters():
                expected_param = dict(expected_block.named_parameters())[name]
                torch.testing.assert_close(param.grad, expected_param.grad, rtol=0.05, atol=0.0002, msg=name)
            if device != "cpu":
                after_attention = get_attention_info()["calls"]
                assert after_attention.get("flash_attention_2", 0) > before_attention.get("flash_attention_2", 0), "H3 did not hit packed FA2"
                after_lightop = get_lightop_info()["kernel_calls"]
                for symbol in ("RMSNORM_ADD", "SWIGLU:bias_autograd"):
                    assert after_lightop.get(symbol, 0) > before_lightop.get(symbol, 0), f"H3 did not hit lightop {symbol}"
                if compiled:
                    assert tensor_ops._CALLS.get("rope", 0) > before_compile.get("rope", 0), "RoPE compile was not dispatched"
            print("PASS block parity: " + json.dumps(dict(
                device=device, dtype=str(dtype), layout=layout, compiled=compiled,
                max_abs_output_error=(actual.float() - expected.float()).abs().max().item(),
                attention=get_attention_info(), lightop=get_lightop_info(),
                tensor_compile=tensor_ops.get_minimax_h3_tensor_compile_info(),
            )), flush=True)
    finally:
        tensor_ops._COMPILED.clear()
        tensor_ops._COMPILED.update(previous_compiled)
        tensor_ops._CALLS.clear()
        tensor_ops._CALLS.update(previous_calls)
        tensor_ops._OPTIONS = previous_options
