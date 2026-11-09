import unittest

import torch
import gc
import importlib
import sys
import types
import weakref
from unittest.mock import patch

from diffsynth_hcu.attention import packed_varlen_attention
from diffsynth_hcu.fa_cache_impl import _SCOPE, FACheckpointCache
from diffsynth_hcu.rng import sample_noise_like, sample_timestep_id
from diffsynth.models.minimax_h3_dit import MiniMaxH3DiTBlock


class ThinRuntimeTests(unittest.TestCase):
    def test_checkpoint_callable_releases_original_inputs(self):
        # Reuse the adapter regression: checkpoint retains its callback after
        # the original activation's saved-tensor lifetime has ended.
        gradient = importlib.import_module("diffsynth.core.gradient.gradient_checkpoint")
        retained = []
        def checkpoint_call(function, *values):
            retained.append(function)
            with torch.no_grad():
                return function(*values).detach()
        fake_ds = types.SimpleNamespace(checkpointing=types.SimpleNamespace(
            is_configured=lambda: True, checkpoint=checkpoint_call))
        value = torch.tensor([3.0], requires_grad=True)
        reference = weakref.ref(value)
        with patch.dict(sys.modules, {"deepspeed": fake_ds}):
            actual = gradient.gradient_checkpoint_forward(
                lambda x, *, scale: x * scale, True, False, value, scale=2
            )
        torch.testing.assert_close(actual, torch.tensor([6.0]))
        del value
        gc.collect()
        self.assertIsNone(reference())
        replacement = torch.tensor([5.0], requires_grad=True)
        retained[0](replacement, 2).sum().backward()
        torch.testing.assert_close(replacement.grad, torch.tensor([2.0]))

    def test_block_output_and_gradients_match_upstream(self):
        from block_compare import compare_blocks
        compare_blocks()

    def test_small_h3_model_forward_and_controlnet_vdn_construction(self):
        from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
        from diffsynth.models.minimax_h3_controlnet import MiniMaxH3ControlNet
        from diffsynth.models.minimax_h3_dit_vdn import MiniMaxH3VDNBranch
        model = MiniMaxH3DiT(
            num_layers=1, token_refiner_num_layers=1, hidden_size=32,
            num_attention_heads=2, attention_head_dim=16, ffn_hidden_size=64,
            text_dim=32, time_embed_hidden_size=32, time_embed_dim=8,
            adaln_out_features=576, final_adaln_out_features=64, rope_inv_freq_len=2,
        )
        positions = lambda ids: {"position_ids": torch.tensor(ids)}
        video, audio = model(
            x=torch.randn(1, 7, 96), audio_x=torch.randn(1, 7, 32),
            img_position_ids=torch.zeros(1, 7, 3), unique_timesteps=torch.tensor([0.5]),
            inverse_indices=torch.zeros(7, dtype=torch.long), update_mask=torch.ones(3),
            token_tags=torch.tensor([0, 0, 0, 1, 1, 2, 2]), prompt_embeds=torch.randn(2, 32),
            img_pos_info=positions([0, 1, 2]), audio_pos_info=positions([3, 4]),
            text_pos_info=positions([5, 6]), img_pos_for_infer_output_info=positions([0, 1, 2]),
            packed_seq_params={"cu_seqlens_q": torch.tensor([0, 7], dtype=torch.int32), "max_seqlen_q": 7},
            refiner_packed_seq_params={"cu_seqlens_q": torch.tensor([0, 2], dtype=torch.int32), "max_seqlen_q": 2},
        )
        self.assertEqual(tuple(video.shape), (3, 96))
        self.assertEqual(tuple(audio.shape), (2, 32))
        (video.square().mean() + audio.square().mean()).backward()
        control = MiniMaxH3ControlNet(control_layers=(0,), hidden_size=32, num_attention_heads=2,
            attention_head_dim=16, ffn_hidden_size=64, time_embed_dim=8, adaln_out_features=576)
        self.assertEqual(len(control.blocks), 1)
        vdn = MiniMaxH3VDNBranch(num_layers=1, hidden_size=32, num_attention_heads=2,
            attention_head_dim=16, linear_head_dim=16, short_conv=())
        self.assertEqual(len(vdn.blocks), 1)

    def test_fa_cache_disables_on_cpu_and_independent_offload(self):
        from diffsynth_hcu.fa_cache import configure_fa_cache
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        model.blocks[0]._hcu_fa_cache_mode = "gpu"
        with patch.object(torch.version, "hip", None):
            self.assertEqual(configure_fa_cache(model, 1), 0)
        self.assertFalse(hasattr(model.blocks[0], "_hcu_fa_cache_mode"))
        with patch.object(torch.version, "hip", "test"):
            self.assertEqual(configure_fa_cache(model, 1, checkpoint_offload=True), 0)

    def test_seeded_rng_is_stable_per_step_and_stream(self):
        context = (42, 0, 3, 0)
        first = sample_noise_like(torch.empty(3, 4), context, stream=1)
        second = sample_noise_like(torch.empty(3, 4), context, stream=1)
        other_stream = sample_noise_like(torch.empty(3, 4), context, stream=2)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other_stream))
        self.assertEqual(
            sample_timestep_id(0, 100, context).item(),
            sample_timestep_id(0, 100, context).item(),
        )

    def test_rng_without_context_preserves_native_sequence(self):
        torch.manual_seed(71)
        expected_timestep = torch.randint(0, 100, (1,))
        expected_noise = torch.randn_like(torch.empty(3, 4))
        torch.manual_seed(71)
        torch.testing.assert_close(sample_timestep_id(0, 100), expected_timestep, rtol=0, atol=0)
        torch.testing.assert_close(sample_noise_like(torch.empty(3, 4)), expected_noise, rtol=0, atol=0)

    def test_lora_installer_preserves_hooks_parameters_and_mutable_scaling(self):
        # Reuse the adapter's parity/hook/restoration contract with a minimal
        # PEFT-shaped Linear double; target PEFT execution remains a separate gate.
        from diffsynth_hcu.lora import LoRAAdapter
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base_layer = torch.nn.Linear(8, 16, bias=False)
                self.base_layer._hcu_fc1_swiglu = True
                self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(8, 2, bias=False)})
                self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 16, bias=False)})
                self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Identity()})
                self.active_adapters = ["default"]
                self.disable_adapters = self.merged = False
                self.scaling = {"default": 1.0}

            def _check_forward_args(self, x):
                pass

            def forward(self, x):
                return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x)) * self.scaling["default"]

        model = torch.nn.Sequential(Linear(), torch.nn.SiLU(), torch.nn.Linear(16, 8))
        x = torch.randn(4, 8, requires_grad=True)
        expected = model(x)
        expected.sum().backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
        input_gradient = x.grad.clone()
        ids = {name: id(p) for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        x.grad = None
        hits = []
        handle = model[0].register_forward_hook(lambda *args: hits.append(1))
        fake_layer = types.ModuleType("peft.tuners.lora.layer")
        fake_layer.Linear = Linear
        with patch.dict(sys.modules, {"peft.tuners.lora.layer": fake_layer}):
            adapter = LoRAAdapter(model, aiter="off")
        try:
            actual = model(x)
            actual.sum().backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, input_gradient, rtol=0, atol=0)
            for name, param in model.named_parameters():
                torch.testing.assert_close(param.grad, gradients[name], rtol=0, atol=0)
            self.assertEqual(ids, {name: id(p) for name, p in model.named_parameters()})
            self.assertEqual(hits, [1])
            self.assertEqual(adapter.calls["unit_scale_fast_path"], 1)
            model[0].scaling["default"] = 0.5
            torch.testing.assert_close(model[0](x), model[0]._hcu_original_lora_forward(x), rtol=0, atol=0)
            self.assertEqual(adapter.calls["original_forward"], 1)
        finally:
            adapter.close()
            handle.remove()
        self.assertNotIn("forward", vars(model[0]))
        self.assertFalse(hasattr(model[0], "_hcu_lora_adapter"))

    def test_packed_attention_cpu_fallback_has_gradients(self):
        q = torch.randn(5, 2, 4, requires_grad=True)
        k = torch.randn(5, 2, 4, requires_grad=True)
        v = torch.randn(5, 2, 4, requires_grad=True)
        out = packed_varlen_attention(
            q, k, v, torch.tensor([0, 2, 5], dtype=torch.int32), softmax_scale=0.5
        )
        self.assertEqual(tuple(out.shape), (5, 2, 4))
        out.square().mean().backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)

    def test_fa_scope_resets_after_exception(self):
        cache = FACheckpointCache(mode="gpu")
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            with cache.scope():
                pass
        self.assertIsNone(_SCOPE.get())
        with self.assertRaisesRegex(ValueError, "sentinel"):
            with FACheckpointCache(mode="gpu").scope():
                raise ValueError("sentinel")
        self.assertIsNone(_SCOPE.get())

    def test_fa_replay_preserves_gradients_across_interleaved_checkpoints(self):
        # Reuse the adapter's FakeFA arithmetic to exercise the real replay
        # autograd nodes and cache ownership without claiming native FA proof.
        import diffsynth_hcu.fa_cache_impl as fa
        from torch.utils.checkpoint import checkpoint
        def attention(q, k, v, scale):
            probs = torch.softmax(torch.einsum("ihd,jhd->hij", q, k) * scale, -1)
            return torch.einsum("hij,jhd->ihd", probs, v)
        class FakeFA:
            forward_calls = backward_calls = 0
            def _wrapped_flash_attn_varlen_forward(self, q, k, v, *args, **kwargs):
                self.forward_calls += 1
                return attention(q, k, v, args[10]), torch.zeros(1), None, torch.zeros(2, dtype=torch.int64)
            def _wrapped_flash_attn_varlen_backward(self, dout, q, k, v, out, lse, dq, dk, dv,
                    cuq, cuk, maxq, maxk, dropout, scale, *args, **kwargs):
                self.backward_calls += 1
                with torch.enable_grad():
                    inputs = [value.detach().requires_grad_() for value in (q, k, v)]
                    gradients = torch.autograd.grad(attention(*inputs, scale), inputs, dout)
                for destination, gradient in zip((dq, dk, dv), gradients):
                    destination.copy_(gradient)
        backend = FakeFA()
        weight = torch.randn(4, 24, requires_grad=True)
        reference_weight = weight.detach().clone().requires_grad_()
        xs = [torch.randn(5, 4, requires_grad=True) for _ in range(2)]
        reference_xs = [value.detach().clone().requires_grad_() for value in xs]
        cu = torch.tensor([0, 5], dtype=torch.int32)
        outputs, caches = [], []
        def block(x):
            q, k, v = (x @ weight).view(5, 3, 2, 4).unbind(1)
            cache, replay = fa._SCOPE.get()
            cache.seen += 1
            return fa._CachedFA.apply(q, k, v, cu, 5, 0.5, cache, replay).square()
        with patch.object(fa, "_backend", return_value=backend):
            for x in xs:
                cache = fa.FACheckpointCache()
                caches.append(cache)
                outputs.append(checkpoint(cache.wrap(block), x, use_reentrant=True))
            sum(value.sum() for value in outputs).backward()
        expected = [attention(*(x @ reference_weight).view(5, 3, 2, 4).unbind(1), 0.5).square() for x in reference_xs]
        sum(value.sum() for value in expected).backward()
        torch.testing.assert_close(weight.grad, reference_weight.grad)
        for actual, ref in zip(xs, reference_xs):
            torch.testing.assert_close(actual.grad, ref.grad)
        self.assertEqual((backend.forward_calls, backend.backward_calls), (2, 2))
        self.assertTrue(all(cache.calls == 2 and cache.gpu is None for cache in caches))
        self.assertIsNone(fa._SCOPE.get())

    def test_h3_block_cpu_forward_backward(self):
        block = MiniMaxH3DiTBlock(32, 2, 16, 64, 8, 6 * 32 * 3, 1e-5, 1e-5)
        x = torch.randn(5, 32, requires_grad=True)
        t_emb = torch.randn(1, 8, requires_grad=True)
        out = block(
            x,
            t_emb=t_emb,
            combined_indices=torch.tensor([0, 1, 2, 0, 1]),
            rope_freqs=torch.randn(5, 16),
            cu_seqlens=torch.tensor([0, 5], dtype=torch.int32),
            max_seqlen=5,
        )
        self.assertEqual(tuple(out.shape), (5, 32))
        out.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(t_emb.grad)


if __name__ == "__main__":
    unittest.main()
