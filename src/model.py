"""
Fourier Neural Operator (FNO) for climate downscaling.

Architecture reference:
  Li, Z. et al. (2021). Fourier Neural Operator for Partial Differential Equations.
  ICLR 2021. https://arxiv.org/abs/2010.08895

The FNO replaces the standard convolution in each layer with a spectral convolution
that operates in the Fourier domain, enabling the model to learn non-local integral
operators and to be resolution-invariant.

Each FNO layer computes:
  v_{t+1}(x) = σ( W v_t(x)  +  F^{-1}[ R(ξ) · F(v_t)(ξ) ] )

where:
  - W      is a pointwise linear transform (local branch)
  - R(ξ)   is a learnable complex weight tensor (spectral branch)
  - F / F^{-1} are the 2-D FFT / iFFT

We keep only the lowest `n_modes` Fourier modes in each spatial dimension,
which acts as an implicit low-pass filter and makes the operator
resolution-invariant within the truncated frequency band.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2-D Fourier layer: applies a learnable complex multiplication in the
    Fourier domain, keeping only the lowest `n_modes_h × n_modes_w` modes.

    Parameters
    ----------
    in_channels, out_channels : int
        Number of input / output feature channels.
    n_modes_h, n_modes_w : int
        Number of Fourier modes retained in the height / width dimension.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 n_modes_h: int, n_modes_w: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_modes_h    = n_modes_h
        self.n_modes_w    = n_modes_w

        # Learnable complex weights: shape (in_ch, out_ch, modes_h, modes_w)
        # stored as real and imaginary parts separately for compatibility
        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes_h, n_modes_w))
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes_h, n_modes_w))

    def _complex_mul2d(self, x_ft: torch.Tensor,
                       w_real: torch.Tensor,
                       w_imag: torch.Tensor) -> torch.Tensor:
        """
        Batched complex matrix-vector product for each Fourier mode.

        x_ft : (B, in_ch, H, W)  – complex tensor
        w    : (in_ch, out_ch, H, W)  – real/imag weight tensors
        """
        # (B, in_ch, H, W) × (in_ch, out_ch, H, W) → (B, out_ch, H, W)
        real = (torch.einsum("bixy,ioxy->boxy", x_ft.real, w_real)
              - torch.einsum("bixy,ioxy->boxy", x_ft.imag, w_imag))
        imag = (torch.einsum("bixy,ioxy->boxy", x_ft.real, w_imag)
              + torch.einsum("bixy,ioxy->boxy", x_ft.imag, w_real))
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_channels, H, W)
        returns : (B, out_channels, H, W)
        """
        B, _, H, W = x.shape

        # Forward 2-D real FFT  →  (B, in_ch, H, W//2+1)
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Allocate output spectrum (complex zeros)
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # Multiply the retained low-frequency modes
        mh = self.n_modes_h
        mw = self.n_modes_w
        out_ft[:, :, :mh, :mw] = self._complex_mul2d(
            x_ft[:, :, :mh, :mw],
            self.weight_real, self.weight_imag)

        # Inverse FFT back to spatial domain  →  (B, out_ch, H, W)
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    """
    Single FNO layer: spectral branch + pointwise (local) branch, followed
    by a GELU activation.
    """

    def __init__(self, channels: int, n_modes_h: int, n_modes_w: int):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, n_modes_h, n_modes_w)
        self.local    = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm     = nn.InstanceNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.local(x)))


class FNO2d(nn.Module):
    """
    Full 2-D Fourier Neural Operator for climate downscaling.

    Architecture
    ------------
    1. Lifting layer  : 1×1 conv  in_channels → hidden_channels
    2. N × FNO blocks : spectral + local branch with GELU
    3. Projection     : 1×1 conv  hidden_channels → 128 → out_channels

    The model optionally accepts time conditioning (day-of-year, hour) via
    a small MLP whose output is added to the lifted features, following the
    approach used in ClimateDiffuse.

    Parameters
    ----------
    in_channels : int
        Number of input channels (2 for T2m+TP, 4 if DEM channels are added).
    out_channels : int
        Number of output channels (always 2: T2m residual + TP residual).
    hidden_channels : int
        Width of the hidden representation (default 64).
    n_modes_h, n_modes_w : int
        Number of retained Fourier modes per spatial dimension (default 16).
    n_layers : int
        Number of FNO blocks (default 4).
    use_time_conditioning : bool
        Whether to add day-of-year / hour embeddings (default True).
    """

    def __init__(
        self,
        in_channels:  int = 2,
        out_channels: int = 2,
        hidden_channels: int = 64,
        n_modes_h: int = 16,
        n_modes_w: int = 16,
        n_layers:  int = 4,
        use_time_conditioning: bool = True,
    ):
        super().__init__()
        self.use_time_conditioning = use_time_conditioning

        # ── Lifting (project input channels → hidden) ────────────────────────
        self.lift = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        # ── Optional time conditioning MLP ───────────────────────────────────
        if use_time_conditioning:
            self.time_mlp = nn.Sequential(
                nn.Linear(2, 64),
                nn.GELU(),
                nn.Linear(64, hidden_channels),
            )

        # ── FNO blocks ────────────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            FNOBlock(hidden_channels, n_modes_h, n_modes_w)
            for _ in range(n_layers)
        ])

        # ── Projection (hidden → out_channels) ───────────────────────────────
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_channels, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(
        self,
        x:    torch.Tensor,
        doy:  torch.Tensor | None = None,
        hour: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, in_channels, H, W)  – normalised coarse input
        doy  : (B,)  – normalised day-of-year ∈ [0, 1]
        hour : (B,)  – normalised hour ∈ [0, 1]

        Returns
        -------
        (B, out_channels, H, W) – predicted normalised residual
        """
        # Lift to hidden space
        x = self.lift(x)                                   # (B, C, H, W)

        # Add time embedding (broadcast over spatial dims)
        if self.use_time_conditioning and doy is not None and hour is not None:
            t = torch.stack([doy, hour], dim=1)            # (B, 2)
            t_emb = self.time_mlp(t)                       # (B, C)
            x = x + t_emb[:, :, None, None]               # broadcast to (B, C, H, W)

        # FNO blocks
        for block in self.blocks:
            x = block(x)

        # Project to output channels
        return self.proj(x)                                # (B, out_ch, H, W)


# ── Convenience factory ───────────────────────────────────────────────────────

def build_fno(in_channels: int = 2, **kwargs) -> FNO2d:
    """Return a default FNO2d model. Pass kwargs to override defaults."""
    return FNO2d(in_channels=in_channels, **kwargs)
