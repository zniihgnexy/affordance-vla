"""Tests for the low-rank adapters added to frozen linear layers."""
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from affordance_vla.heads.lora import LoRALinear, add_lora  # noqa: E402


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.to_out = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.0))
        self.norm = nn.LayerNorm(8)


class Tower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


def test_wrapped_layer_equals_the_base_layer_before_training():
    # Arrange
    torch.manual_seed(0)
    base, x = nn.Linear(8, 4), torch.randn(3, 8)

    # Act
    wrapped = LoRALinear(base, rank=2, alpha=2.0)

    # Assert
    torch.testing.assert_close(wrapped(x), base(x))


def test_wrapped_layer_adds_the_low_rank_update():
    # Arrange
    torch.manual_seed(0)
    base, x = nn.Linear(8, 4), torch.randn(3, 8)
    wrapped = LoRALinear(base, rank=2, alpha=4.0)
    nn.init.normal_(wrapped.up.weight)

    # Act
    out = wrapped(x)

    # Assert: base + (alpha / rank) · up(down(x))
    torch.testing.assert_close(out, base(x) + 2.0 * wrapped.up(wrapped.down(x)))


def test_wrapped_layer_freezes_the_base_and_trains_only_the_update():
    # Act
    wrapped = LoRALinear(nn.Linear(8, 4), rank=2, alpha=2.0)

    # Assert
    trainable = {name for name, parameter in wrapped.named_parameters() if parameter.requires_grad}
    assert trainable == {"down.weight", "up.weight"}


def test_wrapped_layer_follows_the_dtype_of_a_base_layer_in_another_precision():
    # Arrange: frozen weights in another precision than the adapter, as with a bf16 checkpoint
    base, x = nn.Linear(8, 4).double(), torch.randn(3, 8, dtype=torch.float64)
    wrapped = LoRALinear(base, rank=2, alpha=2.0)
    nn.init.normal_(wrapped.up.weight)

    # Act
    out = wrapped(x)

    # Assert
    assert out.dtype == torch.float64 and not torch.allclose(out, base(x))


def test_add_lora_wraps_the_matching_linear_layers_only():
    # Arrange
    tower = Tower()

    # Act
    replaced = add_lora(tower, rank=2, alpha=2.0, match=lambda name: name.endswith("to_q"))

    # Assert
    assert replaced == ["blocks.0.to_q", "blocks.1.to_q"]
    assert isinstance(tower.blocks[0].to_q, LoRALinear) and isinstance(tower.blocks[0].to_out[0], nn.Linear)


def test_add_lora_reaches_layers_inside_containers():
    # Arrange
    tower = Tower()

    # Act
    replaced = add_lora(tower, rank=2, alpha=2.0, match=lambda name: ".to_out." in name)

    # Assert
    assert replaced == ["blocks.0.to_out.0", "blocks.1.to_out.0"]
    assert isinstance(tower.blocks[1].to_out[0], LoRALinear)


def test_add_lora_rejects_a_pattern_that_matches_nothing():
    # Act / Assert
    with pytest.raises(ValueError, match="no linear layer"):
        add_lora(Tower(), rank=2, alpha=2.0, match=lambda name: name.endswith("to_k"))


def test_add_lora_does_not_wrap_a_layer_twice():
    # Arrange
    tower = Tower()
    add_lora(tower, rank=2, alpha=2.0, match=lambda name: "to_q" in name)

    # Act / Assert: the frozen layer inside a wrapper also carries "to_q" in its name
    with pytest.raises(ValueError, match="no linear layer"):
        add_lora(tower, rank=2, alpha=2.0, match=lambda name: "to_q" in name)
    assert isinstance(tower.blocks[0].to_q.base, nn.Linear)
