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


class ToyBlock(nn.Module):
    """One block of the toy deep DiT: reads its input and the conditioning states."""

    def __init__(self) -> None:
        super().__init__()
        self.mix = nn.Linear(WIDTH, WIDTH)

    def forward(self, hidden_states, encoder_hidden_states=None, temb=None):
        context = encoder_hidden_states.float().mean(dim=1, keepdim=True)
        return hidden_states + torch.tanh(self.mix(hidden_states) + context)


class ToyDeepDiT(nn.Module):
    """Three blocks called one after the other with the hidden states first, like starVLA's DiT."""

    def __init__(self) -> None:
        super().__init__()
        self._interleave_self_attention = False
        self.transformer_blocks = nn.ModuleList(ToyBlock() for _ in range(3))

    def forward(self, hidden_states, encoder_hidden_states, timestep, encoder_attention_mask=None,
                return_pre_output=False):
        for index, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states[index % 2], temb=None)
        return hidden_states


def deep_head() -> ToyHead:
    head = ToyHead()
    head.model = ToyDeepDiT()
    return head


def deep_model(head: ToyHead | None = None, deep: bool = True) -> PlanConditionedHead:
    """A plan head on the toy deep DiT; the low-rank updates start at zero and change nothing."""
    return PlanConditionedHead(head or deep_head(), lora_rank=2, lora_alpha=2.0,
                               lora_match=lambda name: name.endswith("mix"), deep=deep)


def open_plan(model: PlanConditionedHead) -> None:
    """Give the plan a non-zero embedding, so that whatever reads it has something to read."""
    with torch.no_grad():
        model.plan_encoder.net[-1].bias.fill_(0.5)


def test_a_deep_plan_head_starts_as_the_released_head():
    # Arrange: the plan's embedding is non-zero, so only the zero maps keep it out of the later blocks
    torch.manual_seed(0)
    head = deep_head()
    noisy, timesteps, states = make_inputs()
    model = deep_model(head)
    open_plan(model)
    expected = head_velocity(head, noisy, timesteps, states, None, torch.full((2, WIDTH), 0.5))

    # Act
    velocity = model.velocity(noisy, timesteps, states, None, torch.randn(2, PLAN_DIM))

    # Assert
    assert not any(parameter.any() for parameter in model.deep_plan.parameters())
    torch.testing.assert_close(velocity, expected)


@pytest.mark.parametrize("index", [0, 1])
def test_each_map_of_a_deep_plan_lets_the_plan_into_its_block(index: int):
    # Arrange: the toy DiT has three blocks; map 0 feeds block 1 and map 1 feeds block 2
    torch.manual_seed(0)
    model = deep_model()
    open_plan(model)
    noisy, timesteps, states = make_inputs()
    plan = torch.randn(2, PLAN_DIM)
    with torch.no_grad():
        before = model.velocity(noisy, timesteps, states, None, plan)
        model.deep_plan.maps[index].weight.normal_()

        # Act
        after = model.velocity(noisy, timesteps, states, None, plan)

    # Assert
    assert len(model.deep_plan.maps) == 2  # block 0 receives the plan with the action tokens
    assert not torch.allclose(after, before)


def test_the_first_map_of_a_deep_plan_changes_what_the_second_block_is_given():
    # Arrange: record the input of every block with and without the first map open
    torch.manual_seed(0)
    model = deep_model()
    open_plan(model)
    noisy, timesteps, states = make_inputs()
    plan = torch.randn(2, PLAN_DIM)
    seen: list[list[torch.Tensor]] = []
    for block in model.head.model.transformer_blocks:
        block.register_forward_hook(lambda _block, args, _output: seen[-1].append(args[0].detach().clone()))

    # Act
    with torch.no_grad():
        seen.append([])
        model.velocity(noisy, timesteps, states, None, plan)
        model.deep_plan.maps[0].weight.normal_()
        seen.append([])
        model.velocity(noisy, timesteps, states, None, plan)

    # Assert: block 0 is given the same input, block 1 a different one
    torch.testing.assert_close(seen[1][0], seen[0][0])
    assert not torch.allclose(seen[1][1], seen[0][1])


def test_sampling_with_a_deep_plan_reads_the_plan_at_every_step():
    # Arrange
    torch.manual_seed(0)
    model = deep_model()
    open_plan(model)
    _, _, states = make_inputs()
    noise, plan = torch.randn(2, HORIZON, DIM), torch.randn(2, PLAN_DIM)
    with torch.no_grad():
        before = model.sample(states, None, plan, noise)
        model.deep_plan.maps[-1].weight.normal_()

        # Act
        after = model.sample(states, None, plan, noise)

    # Assert
    assert not torch.allclose(after, before)
    assert model.deep_plan._embedding is None


def test_gradients_reach_every_map_and_the_plan_encoder():
    # Arrange
    torch.manual_seed(0)
    model = deep_model()
    open_plan(model)
    noisy, timesteps, states = make_inputs()
    with torch.no_grad():
        model.deep_plan.maps[-1].weight.normal_()

    # Act
    model.velocity(noisy, timesteps, states, None, torch.randn(2, PLAN_DIM)).sum().backward()

    # Assert
    for layer in model.deep_plan.maps:
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
    assert model.plan_encoder.net[-1].weight.grad.abs().sum() > 0


def test_the_plan_is_taken_back_from_the_blocks_after_a_pass_and_after_an_error():
    # Arrange
    model = deep_model()
    noisy, timesteps, states = make_inputs()

    # Act
    with torch.no_grad():
        model.velocity(noisy, timesteps, states, None, torch.randn(2, PLAN_DIM))
    after_pass = model.deep_plan._embedding
    with pytest.raises(RuntimeError, match="stop"), model.deep_plan.given(torch.ones(2, WIDTH)):
        raise RuntimeError("stop")

    # Assert
    assert after_pass is None
    assert model.deep_plan._embedding is None


def test_a_head_used_without_its_plan_wrapper_is_the_released_head_even_with_open_maps():
    # Arrange: the released head's output is taken before the wrapper adds its hooks
    torch.manual_seed(0)
    head = deep_head()
    noisy, timesteps, states = make_inputs()
    expected = plain_velocity(head, noisy, timesteps, states)
    model = deep_model(head)
    open_plan(model)
    with torch.no_grad():
        model.deep_plan.maps[-1].weight.normal_()
        model.velocity(noisy, timesteps, states, None, torch.randn(2, PLAN_DIM))

        # Act: the head is called directly, without a plan
        plain = plain_velocity(head, noisy, timesteps, states)

    # Assert
    torch.testing.assert_close(plain, expected)


def test_the_adapter_of_a_deep_head_restores_its_maps_in_a_fresh_head():
    # Arrange: a deep head with trained-looking values, and a second one around the same frozen weights
    torch.manual_seed(0)
    head = deep_head()
    base_state = {name: value.clone() for name, value in head.state_dict().items()}
    model = deep_model(head)
    for parameter in model.parameters():
        if parameter.requires_grad:
            nn.init.normal_(parameter, std=0.5)
    noisy, timesteps, states = make_inputs()
    plan = torch.randn(2, PLAN_DIM)
    expected = model.velocity(noisy, timesteps, states, None, plan)
    fresh_head = deep_head()
    fresh_head.load_state_dict(base_state)
    fresh = deep_model(fresh_head)

    # Act
    fresh.load_adapter(model.adapter_state())

    # Assert
    assert any(name.startswith("deep_plan.maps.") for name in model.adapter_state())
    torch.testing.assert_close(fresh.velocity(noisy, timesteps, states, None, plan), expected)


def test_a_deep_adapter_does_not_fit_a_head_without_the_deep_plan_and_the_reverse():
    # Arrange
    deep, shallow = deep_model(), deep_model(deep=False)

    # Act / Assert
    assert shallow.deep_plan is None
    with pytest.raises(ValueError, match="adapter state does not fit"):
        shallow.load_adapter(deep.adapter_state())
    with pytest.raises(ValueError, match="adapter state does not fit"):
        deep.load_adapter(shallow.adapter_state())


def test_a_block_given_its_hidden_states_by_keyword_is_refused():
    # Arrange: the maps can only reach hidden states passed as the first positional argument
    model = deep_model()
    block = model.head.model.transformer_blocks[1]

    # Act / Assert
    with pytest.raises(RuntimeError, match="positional"), model.deep_plan.given(torch.ones(2, WIDTH)):
        block(hidden_states=torch.zeros(2, HORIZON, WIDTH), encoder_hidden_states=torch.zeros(2, TOKENS, WIDTH))
