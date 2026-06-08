import torch

from attention_spectrum_replay.config import SpectralConfig
from attention_spectrum_replay.spectral import SpectralEncoder


def test_descriptor_dim_and_translation_invariance():
    cfg = SpectralConfig(radial_bins=8, angular_bins=8, selected_last_layers=4)
    enc = SpectralEncoder(cfg)
    attn = torch.rand(2, 4, 3, 5, 4, 4)
    attn = attn / attn.sum(dim=(-1, -2), keepdim=True)
    shifted = torch.roll(attn, shifts=(1, 1), dims=(-1, -2))
    mask = torch.ones(2, 5)
    a = enc(attn, mask)
    b = enc(shifted, mask)
    assert a.descriptor.shape[-1] == 54
    assert (a.descriptor - b.descriptor).abs().max().item() < 1e-5
