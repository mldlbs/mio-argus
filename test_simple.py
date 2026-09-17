import asyncio
import base64
import io
import warnings

import pytest
import torch
from PIL import Image

import api_server


class FakeTokenizer:
    def __call__(self, instruction, max_length, padding, truncation, return_tensors):
        return {
            "input_ids": torch.zeros(1, max_length, dtype=torch.long),
            "attention_mask": torch.ones(1, max_length, dtype=torch.long),
        }


class FakeModel:
    def __call__(self, image_tensor, input_ids, attention_mask):
        return {
            "action_logits": torch.tensor([[-10.0, -10.0, -10.0, -10.0, -10.0, 0.0]]),
            "coord_pred": torch.tensor([[[0.1, 0.2]]]),
            "scroll_logits": torch.tensor([[0.0, 0.0, 10.0]]),
        }


def test_predict_returns_action_name():
    api_server.model = FakeModel()
    api_server.tokenizer = FakeTokenizer()
    api_server.device = torch.device("cpu")

    image = Image.new("RGB", (224, 224), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    request = api_server.PredictRequest(
        image_base64=base64.b64encode(buffer.getvalue()).decode(),
        instruction="click",
    )

    result = asyncio.run(api_server.predict(request))

    assert result.action == "press"
    assert result.coord == pytest.approx([0.1, 0.2])
    assert result.scroll == "none"
    assert result.all_probs == pytest.approx({
        "click": 0.0000,
        "type": 0.0000,
        "scroll": 0.0000,
        "move": 0.0000,
        "hotkey": 0.0000,
        "press": 1.0,
    }, abs=3e-4)


def test_lora_linear_state_dict_keys_and_forward():
    import torch.nn.functional as F
    from mio_argus import LoRALinear

    linear = LoRALinear(4, 3, rank=2)
    x = torch.randn(2, 4)
    expected = F.linear(x, linear.weight, linear.bias) + (
        x @ linear.lora_A.t() @ linear.lora_B.t()
    ) * linear.lora_scaling

    assert set(linear.state_dict()) == {
        "weight",
        "bias",
        "lora_A",
        "lora_B",
    }
    assert torch.allclose(linear(x), expected)


def test_apply_lora_is_idempotent_and_targeted():
    import torch
    import torch.nn as nn
    from mio_argus import LoRALinear, apply_lora

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)
            self.other = nn.Linear(4, 4, bias=False)

    model = TinyModel()
    apply_lora(model, rank=3)
    apply_lora(model, rank=3)

    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.other, nn.Linear)
    assert not hasattr(model.other, "lora_A")
    assert model.q_proj.lora_A.shape == (3, 4)
    assert model.q_proj.lora_B.shape == (4, 3)


def test_v3_constructor_parameters_are_effective(monkeypatch):
    import torch.nn as nn
    import mio_argus as model_module

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)

        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def forward(self, **kwargs):
            return None

    class TinyFusion:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(model_module, "ViTModel", TinyEncoder)
    monkeypatch.setattr(model_module, "GPT2Model", TinyEncoder)
    monkeypatch.setattr(model_module, "CrossAttentionFusion", TinyFusion)

    model = model_module.ComputerUseModelV3(
        num_actions=4,
        freeze_vit=False,
        freeze_gpt2=False,
        coord_loss_weight=2.0,
        scroll_loss_weight=0.25,
        lora_rank=5,
    )

    assert model.num_actions == 4
    assert model.action_head[-1].out_features == 4
    assert model.coord_loss_weight == 2.0
    assert model.scroll_loss_weight == 0.25
    assert model.lora_rank == 5
    assert isinstance(model.vit.q_proj, model_module.LoRALinear)
    assert model.vit.q_proj.lora_A.shape == (5, 4)
    assert all(p.requires_grad for p in model.vit.parameters())
    assert all(p.requires_grad for p in model.gpt2.parameters())


def test_compute_loss_accepts_scalar_coord_mask():
    import torch.nn.functional as F

    from mio_argus import ComputerUseModelV3

    model = ComputerUseModelV3.__new__(ComputerUseModelV3)
    model.coord_loss_weight = 1.0
    model.scroll_loss_weight = 1.0

    outputs = {
        "action_logits": torch.tensor([[-2.0, 0.0, -2.0]]),
        "coord_pred": torch.tensor([[[0.2, 0.3]]]),
        "scroll_logits": torch.tensor([[0.0, 0.0, 2.0]]),
    }
    targets = {
        "action": torch.tensor([1]),
        "coords": torch.tensor([[0.1, 0.2]]),
        "scroll": torch.tensor([2]),
        "coord_mask": torch.tensor(True),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        loss = model.compute_loss(outputs, targets)

    expected = (
        F.cross_entropy(outputs["action_logits"], targets["action"])
        + F.huber_loss(
            outputs["coord_pred"].reshape(-1),
            targets["coords"].reshape(-1),
            delta=0.1,
        )
        + F.cross_entropy(outputs["scroll_logits"], targets["scroll"])
    )
    assert torch.allclose(loss, expected)


def test_apply_lora_preserves_legacy_linear_state_dict_keys():
    import torch.nn as nn
    from mio_argus import LoRALinear, apply_lora

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 3, bias=True)
            self.proj.lora_A = torch.nn.Parameter(torch.randn(2, 4) * 0.01)
            self.proj.lora_B = torch.nn.Parameter(torch.zeros(3, 2))

    model = TinyModel()
    apply_lora(model, target_names=('proj',), rank=2)

    assert isinstance(model.proj, LoRALinear)
    assert set(model.proj.state_dict()) == {"weight", "bias", "lora_A", "lora_B"}
