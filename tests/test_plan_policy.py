"""Tests for conditioning a starVLA flow-matching action head on a plan (toy stand-in for the head)."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from affordance3d.plan_input import PLAN_DIM  # noqa: E402
from affordance3d.plan_policy import (  # noqa: E402
    PlanConditionedHead,
    PlanEncoder,
    dit_lora_match,
    head_velocity,
    sample_actions,
)

WIDTH, DIM, HORIZON, TOKENS = 16, 7, 4, 5


class ToyActionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(DIM, WIDTH)
        self.time = nn.Embedding(1000, WIDTH)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.layer(actions) + self.time(timesteps)[:, None, :]


class ToyDiT(nn.Module):
    """Every token reads the conditioning states and nothing else, like the all-cross-attention DiT."""

    def __init__(self, interleave: bool = False) -> None:
        super().__init__()
        self._interleave_self_attention = interleave
        self.mix = nn.Linear(WIDTH, WIDTH)

    def forward(self, hidden_states, encoder_hidden_states, timestep, encoder_attention_mask=None,
                return_pre_output=False):
        context = sum(state.float().mean(dim=1, keepdim=True) for state in encoder_hidden_states)
        return torch.tanh(self.mix(hidden_states) + context)


class ToyHead(nn.Module):
    """The attributes of starVLA's LayerwiseFlowmatchingActionHead that the plan code relies on."""

    def __init__(self, interleave: bool = False) -> None:
        super().__init__()
        self.action_encoder = ToyActionEncoder()
        self.position_embedding = nn.Embedding(32, WIDTH)
        self.model = ToyDiT(interleave)
        self.action_decoder = nn.Linear(WIDTH, DIM)
        self.config = SimpleNamespace(add_pos_embed=True)
        self.action_horizon, self.action_dim = HORIZON, DIM
        self.num_inference_timesteps, self.num_timestep_buckets = 4, 1000
        self.input_embedding_dim = WIDTH


def make_inputs(batch: int = 2):
    torch.manual_seed(0)
    states = [torch.randn(batch, TOKENS, WIDTH) for _ in range(2)]
    return torch.randn(batch, HORIZON, DIM), torch.tensor([100, 700][:batch]), states


def plain_velocity(head: ToyHead, noisy, timesteps, states) -> torch.Tensor:
    features = head.action_encoder(noisy, timesteps) + head.position_embedding(torch.arange(HORIZON))[None]
    return head.action_decoder(head.model(features, states, timesteps))


def test_plan_encoder_starts_at_zero_so_the_head_is_unchanged():
    # Arrange
    encoder = PlanEncoder(width=WIDTH)

    # Act
    embedding = encoder(torch.randn(3, PLAN_DIM))

    # Assert
    assert embedding.shape == (3, WIDTH) and not embedding.any()


def test_velocity_with_a_zero_plan_equals_the_plain_head():
    # Arrange
    head = ToyHead()
    noisy, timesteps, states = make_inputs()

    # Act
    velocity = head_velocity(head, noisy, timesteps, states, None, torch.zeros(2, WIDTH))

    # Assert
    assert velocity.shape == (2, HORIZON, DIM)
    torch.testing.assert_close(velocity, plain_velocity(head, noisy, timesteps, states))


def test_velocity_depends_on_the_plan_embedding():
    # Arrange
    head = ToyHead()
    noisy, timesteps, states = make_inputs()

    # Act
    velocity = head_velocity(head, noisy, timesteps, states, None, torch.ones(2, WIDTH))

    # Assert
    assert not torch.allclose(velocity, plain_velocity(head, noisy, timesteps, states))


def test_velocity_refuses_a_head_whose_tokens_attend_to_each_other():
    # Arrange: with self-attention the 32 learned tokens matter and may not be dropped
    head = ToyHead(interleave=True)
    noisy, timesteps, states = make_inputs()

    # Act / Assert
    with pytest.raises(ValueError, match="self-attention"):
        head_velocity(head, noisy, timesteps, states, None, torch.zeros(2, WIDTH))


def test_sample_actions_integrates_the_velocity_from_the_given_noise():
    # Arrange
    head = ToyHead()
    _, _, states = make_inputs()
    noise = torch.randn(2, HORIZON, DIM)

    # Act
    actions = sample_actions(head, states, None, torch.zeros(2, WIDTH), noise)

    # Assert: four Euler steps of size 1/4 at time buckets 0, 250, 500, 750
    expected = noise
    for bucket in (0, 250, 500, 750):
        expected = expected + plain_velocity(head, expected, torch.full((2,), bucket), states) / 4
    torch.testing.assert_close(actions, expected)


def test_adapter_state_holds_only_what_is_trained_and_restores_the_outputs():
    # Arrange: train-able parts get non-trivial values in one model ...
    torch.manual_seed(0)
    head = ToyHead()
    base_state = {name: value.clone() for name, value in head.state_dict().items()}
    model = PlanConditionedHead(head, lora_rank=2, lora_alpha=2.0, lora_match=lambda name: name.endswith("mix"))
    for parameter in model.parameters():
        if parameter.requires_grad:
            nn.init.normal_(parameter, std=0.5)
    noisy, timesteps, states = make_inputs()
    plan = torch.randn(2, PLAN_DIM)
    expected = model.velocity(noisy, timesteps, states, None, plan)

    # Act: ... and are loaded into a second model built from the same frozen head
    fresh_head = ToyHead()
    fresh_head.load_state_dict(base_state)
    fresh = PlanConditionedHead(fresh_head, lora_rank=2, lora_alpha=2.0, lora_match=lambda name: name.endswith("mix"))
    fresh.load_adapter(model.adapter_state())

    # Assert
    names = set(model.adapter_state())
    assert all(name.startswith(("plan_encoder.", "head.action_decoder.")) or name.endswith(("down.weight", "up.weight"))
               for name in names)
    assert any(name.endswith("up.weight") for name in names)
    assert not any(name.startswith("head.action_encoder.") for name in names)
    torch.testing.assert_close(fresh.velocity(noisy, timesteps, states, None, plan), expected)


def test_load_adapter_rejects_a_state_with_missing_or_unknown_entries():
    # Arrange
    model = PlanConditionedHead(ToyHead(), lora_rank=2, lora_alpha=2.0, lora_match=lambda name: name.endswith("mix"))
    state = model.adapter_state()
    state.pop(next(iter(state)))

    # Act / Assert
    with pytest.raises(ValueError, match="adapter state does not fit"):
        model.load_adapter(state)


def test_dit_lora_match_selects_attention_and_feed_forward_layers_of_the_blocks():
    # Arrange
    wanted = ["model.transformer_blocks.0.attn1.to_q", "model.transformer_blocks.35.attn1.to_out.0",
              "model.transformer_blocks.3.ff.net.0.proj", "model.transformer_blocks.3.ff.net.2"]
    unwanted = ["model.transformer_blocks.0.norm1.linear", "model.proj_out_1", "action_decoder.layer1",
                "action_encoder.layer1", "model.timestep_encoder.timestep_embedder.linear_1"]

    # Act / Assert
    assert all(dit_lora_match(name) for name in wanted)
    assert not any(dit_lora_match(name) for name in unwanted)


def test_velocity_refuses_a_head_that_does_not_say_how_its_tokens_attend():
    # Arrange: without the flag the code cannot know whether the learned tokens may be left out
    head = ToyHead()
    del head.model._interleave_self_attention
    noisy, timesteps, states = make_inputs()

    # Act / Assert
    with pytest.raises(ValueError, match="cannot tell"):
        head_velocity(head, noisy, timesteps, states, None, torch.zeros(2, WIDTH))


def test_plan_conditioned_head_samples_with_its_own_plan_encoder():
    # Arrange
    torch.manual_seed(0)
    model = PlanConditionedHead(ToyHead(), lora_rank=2, lora_alpha=2.0, lora_match=lambda name: name.endswith("mix"))
    nn.init.normal_(model.plan_encoder.net[-1].weight, std=0.5)
    _, _, states = make_inputs()
    noise, plan = torch.randn(2, HORIZON, DIM), torch.randn(2, PLAN_DIM)

    # Act
    actions = model.sample(states, None, plan, noise)

    # Assert
    torch.testing.assert_close(actions, sample_actions(model.head, states, None, model.plan_encoder(plan), noise))
    assert not torch.allclose(actions, sample_actions(model.head, states, None, torch.zeros(2, WIDTH), noise))
