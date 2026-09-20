"""Isolated PEFT integration tests; mocked dispatch is not GPU kernel validation."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch, Mock

import torch
import torch.utils.checkpoint
from peft import LoraConfig, inject_adapter_in_model

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "_minimax_h3_test"
package = ModuleType(PACKAGE)
package.__path__ = []
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + ".minimax_h3_aiter", ROOT / "diffsynth/models/minimax_h3_aiter.py")
ops = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ops
spec.loader.exec_module(ops)
dit = ModuleType(PACKAGE + ".minimax_h3_dit")
dit.__dict__.update(nn=torch.nn, minimax_h3_fc1_swiglu=ops.minimax_h3_fc1_swiglu)
tree = ast.parse((ROOT / "diffsynth/models/minimax_h3_dit.py").read_text(encoding="utf-8"))
nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3MLP"]
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<MiniMaxH3MLP>", "exec"), dit.__dict__)
sys.modules[dit.__name__] = dit


def create_model(dtype=torch.float32):
    model = dit.MiniMaxH3MLP(8, 16).to(dtype=dtype)
    model = inject_adapter_in_model(LoraConfig(r=2, lora_alpha=2, target_modules=["fc1", "fc2"]), model)
    with torch.no_grad():
        for layer in (model.fc1, model.fc2):
            layer.lora_B["default"].weight.normal_(0, 0.1)
    return model.to(dtype=dtype)


def reference(model, x):
    hidden = model.fc1._h3_original_lora_forward(x) if hasattr(model.fc1, "_h3_original_lora_forward") else model.fc1(x)
    return model.fc2(ops._native_swiglu(hidden))


class AiterIntegrationTest(unittest.TestCase):
    def test_missing_backend_does_not_patch(self):
        model = create_model()
        forward = model.fc1.forward
        with patch.object(ops, "_load_aiter_add_swiglu", return_value=None):
            self.assertEqual(ops.enable_minimax_h3_aiter_add_swiglu(model), 0)
        self.assertEqual(model.fc1.forward, forward)

    def test_import_error_is_silent_fallback(self):
        import builtins
        original = builtins.__import__
        def failing(name, *args, **kwargs):
            if name.startswith("aiter"):
                raise OSError("optional library unavailable")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=failing):
            self.assertIsNone(ops._load_aiter_add_swiglu())

    def test_cpu_fallback_and_only_fc1_is_patched(self):
        model = create_model()
        fc2_forward = model.fc2.forward
        kernel = Mock(side_effect=AssertionError("CPU must not dispatch"))
        with patch.object(ops, "_load_aiter_add_swiglu", return_value=kernel):
            self.assertEqual(ops.enable_minimax_h3_aiter_add_swiglu(model), 1)
            self.assertEqual(ops.enable_minimax_h3_aiter_add_swiglu(model), 0)
        self.assertEqual(model.fc2.forward, fc2_forward)
        x = torch.randn(3, 8, requires_grad=True)
        y = model(x)
        expected = reference(model, x)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        params = [x] + [p for p in model.parameters() if p.requires_grad]
        for a, b in zip(torch.autograd.grad(y.sum(), params), torch.autograd.grad(expected.sum(), params)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        kernel.assert_not_called()

    def test_mocked_fusion_gradients_hooks_and_checkpoint(self):
        model = create_model(torch.bfloat16)
        kernel = Mock(side_effect=lambda a, b: ops._native_swiglu(a + b))
        with patch.object(ops, "_load_aiter_add_swiglu", return_value=kernel):
            ops.enable_minimax_h3_aiter_add_swiglu(model)
        hook_calls = []
        handles = [module.register_forward_hook(lambda *args: hook_calls.append(True)) for module in (model.fc1, model.fc1.base_layer, model.fc1.lora_A["default"], model.fc1.lora_B["default"])]
        x = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
        # This mocks eligibility only; all tensors and arithmetic remain on CPU.
        with patch.object(torch.Tensor, "is_cuda", property(lambda _: True)):
            y = torch.utils.checkpoint.checkpoint(model, x, use_reentrant=True)
            y.float().square().sum().backward()
            self.assertGreaterEqual(kernel.call_count, 2)
            self.assertIsInstance(model.fc1(x), torch.Tensor)
        self.assertGreaterEqual(len(hook_calls), 8)
        self.assertIsNone(ops._SPLIT_FC1.get())
        actual_grads = [x.grad.clone()] + [p.grad.clone() for p in model.parameters() if p.requires_grad]
        model.zero_grad()
        x.grad = None
        expected = reference(model, x)
        expected.float().square().sum().backward()
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        for a, b in zip(actual_grads, [x.grad] + [p.grad for p in model.parameters() if p.requires_grad]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for handle in handles:
            handle.remove()

    def test_mutable_adapter_state_and_mixed_dtype_fallback(self):
        for state in ("scaled", "disabled", "merged", "mixed", "multiple"):
            with self.subTest(state=state):
                model = create_model(torch.bfloat16)
                kernel = Mock(side_effect=AssertionError("unsupported state must not dispatch"))
                with patch.object(ops, "_load_aiter_add_swiglu", return_value=kernel):
                    ops.enable_minimax_h3_aiter_add_swiglu(model)
                if state == "scaled":
                    model.fc1.scaling["default"] = 0.5
                elif state == "disabled":
                    model.fc1.enable_adapters(False)
                elif state == "merged":
                    model.fc1.merge()
                elif state == "mixed":
                    model.fc1.lora_A["default"].float()
                    model.fc1.lora_B["default"].float()
                else:
                    model.fc1.update_layer("second", r=2, lora_alpha=2, lora_dropout=0, init_lora_weights=True, use_rslora=False)
                    model.fc1.set_adapter(["default", "second"])
                x = torch.randn(3, 8, dtype=torch.bfloat16)
                with patch.object(torch.Tensor, "is_cuda", property(lambda _: True)):
                    torch.testing.assert_close(model(x), reference(model, x), rtol=0, atol=0)
                kernel.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "requires GPU and AITER training operator")
    def test_real_gpu_kernel(self):
        kernel = ops._load_aiter_add_swiglu()
        if kernel is None:
            self.skipTest("AITER Add+SwiGLU unavailable")
        for dtype in (torch.float16, torch.bfloat16):
            model = create_model(dtype).cuda()
            original = copy.deepcopy(model)
            counted = Mock(side_effect=kernel)
            with patch.object(ops, "_load_aiter_add_swiglu", return_value=counted):
                self.assertEqual(ops.enable_minimax_h3_aiter_add_swiglu(model), 1)
            x = torch.randn(7, 8, device="cuda", dtype=dtype, requires_grad=True)
            ref_x = x.detach().clone().requires_grad_()
            y, ref = model(x), original(ref_x)
            y.float().square().sum().backward()
            ref.float().square().sum().backward()
            counted.assert_called_once()
            torch.testing.assert_close(y, ref, rtol=0.03, atol=0.01)
            torch.testing.assert_close(x.grad, ref_x.grad, rtol=0.03, atol=0.01)
            for a, b in zip(model.parameters(), original.parameters()):
                if a.requires_grad:
                    torch.testing.assert_close(a.grad, b.grad, rtol=0.03, atol=0.01)


if __name__ == "__main__":
    unittest.main()
