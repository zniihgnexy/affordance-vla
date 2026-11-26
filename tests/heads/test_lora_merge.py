import pytest
import torch
from torch import nn

from affordance_vla.heads.lora import add_lora
from affordance_vla.heads.lora_merge import merge_lora_state


def test_merged_weights_reproduce_the_adapted_layer():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
    add_lora(model, rank=2, alpha=4.0, match=lambda name: name == "0")
    for module in model.modules():
        if hasattr(module, "up"):
            nn.init.normal_(module.up.weight)  # a trained adapter, not the zero start
    x = torch.randn(5, 8)
    expected = model(x)

    merged = merge_lora_state(model.state_dict(), alpha=4.0, rank=2)
    assert set(merged) == {"0.weight", "0.bias", "2.weight", "2.bias"}
    plain = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
    plain.load_state_dict(merged)
    assert torch.allclose(plain(x), expected, atol=1e-5)


def test_an_adapter_without_its_base_weight_is_an_error():
    state = {"0.base.weight": torch.zeros(2, 2), "0.down.weight": torch.zeros(1, 2)}
    with pytest.raises(KeyError):
        merge_lora_state(state, alpha=2.0, rank=1)
