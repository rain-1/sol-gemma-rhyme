import torch

from rhyme_interp.analysis import attention_head_ablation, layer_update_ablation


class FakeAttention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.dense = torch.nn.Identity()
        self.width = width

    def forward(self, x):
        return self.dense(x)


class FakeLayer(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = FakeAttention(width)

    def forward(self, x):
        return x + self.attention(x)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt_neox = torch.nn.Module()
        self.gpt_neox.layers = torch.nn.ModuleList([FakeLayer(4)])
        self.config = type("Config", (), {"num_attention_heads": 2, "hidden_size": 4})()


def test_layer_update_ablation_only_changes_final_position():
    model = FakeModel()
    x = torch.ones(1, 2, 4)
    with layer_update_ablation(model, 0):
        result = model.gpt_neox.layers[0](x)
    assert torch.equal(result[:, 0], torch.full((1, 4), 2.0))
    assert torch.equal(result[:, 1], torch.ones(1, 4))


def test_head_ablation_zeroes_requested_final_slice():
    model = FakeModel()
    x = torch.ones(1, 2, 4)
    with attention_head_ablation(model, 0, 1):
        result = model.gpt_neox.layers[0].attention(x)
    assert torch.equal(result[0, 0], torch.ones(4))
    assert torch.equal(result[0, 1], torch.tensor([1.0, 1.0, 0.0, 0.0]))

