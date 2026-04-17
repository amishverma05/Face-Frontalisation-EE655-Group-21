"""
models/stylegan2_wrapper.py
────────────────────────────
Wrapper around the official rosinality style-GAN2 generator
(stylegan2_model.py), patched to use pure-Python / PyTorch fallbacks
for the custom CUDA ops (upfirdn2d, fused_leaky_relu) so it works
on Windows without a C++ compiler.

All synthesis weights are loaded from ./checkpoints/ffhq-256-config-e.pt
and frozen.  The public interface is:

    G = StyleGAN2Generator('checkpoints/ffhq-256-config-e.pt').to(device).eval()
    imgs = G(w_plus, input_is_latent=True)   # (B, 3, 256, 256)  in [-1, 1]
"""

import sys, os, math, importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Inject pure-Python op shims BEFORE importing stylegan2_model
#           so that the `from op import …` inside that file hits our shims.
# ─────────────────────────────────────────────────────────────────────────────

class _UpFirDn2dNative:
    """Pure-PyTorch upfirdn2d (matches rosinality upfirdn2d_native exactly)."""
    @staticmethod
    def apply(input, kernel, up, down, pad):
        if isinstance(up, int):   up   = (up, up)
        if isinstance(down, int): down = (down, down)
        if len(pad) == 2:         pad  = (pad[0], pad[1], pad[0], pad[1])
        return _upfirdn2d_pt(input, kernel, *up, *down, *pad)


def _upfirdn2d_pt(input, kernel, up_x, up_y, down_x, down_y,
                  pad_x0, pad_x1, pad_y0, pad_y1):
    """Pure-PyTorch equivalent of rosinality's upfirdn2d_native."""
    _, channel, in_h, in_w = input.shape
    x = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = x.shape
    kernel_h, kernel_w = kernel.shape

    x = x.view(-1, in_h, 1, in_w, 1, minor)
    x = F.pad(x, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    x = x.view(-1, in_h * up_y, in_w * up_x, minor)

    x = F.pad(x, [0, 0,
                   max(pad_x0, 0), max(pad_x1, 0),
                   max(pad_y0, 0), max(pad_y1, 0)])
    x = x[:, max(-pad_y0, 0): x.shape[1] - max(-pad_y1, 0) or None,
              max(-pad_x0, 0): x.shape[2] - max(-pad_x1, 0) or None, :]

    x = x.permute(0, 3, 1, 2)
    x = x.reshape(-1, 1,
                   in_h * up_y + pad_y0 + pad_y1,
                   in_w * up_x + pad_x0 + pad_x1)
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    x = F.conv2d(x, w)
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
    x = x.reshape(-1, minor, out_h * down_y, out_w * down_x)
    x = x.permute(0, 2, 3, 1)
    x = x[:, ::down_y, ::down_x, :]
    return x.reshape(-1, channel, out_h, out_w)


def _fused_leaky_relu_pt(input, bias=None, negative_slope=0.2, scale=math.sqrt(2)):
    """Pure-PyTorch fused_leaky_relu (CPU + CUDA safe)."""
    if bias is not None:
        rest = [1] * (input.ndim - bias.ndim - 1)
        input = input + bias.view(1, bias.shape[0], *rest)
    return F.leaky_relu(input, negative_slope=negative_slope) * scale


class _FusedLeakyReLU(nn.Module):
    """Pure-PyTorch FusedLeakyReLU with learnable bias — matches rosinality."""
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=math.sqrt(2)):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x):
        return _fused_leaky_relu_pt(x, self.bias, self.negative_slope, self.scale)


# Build the fake 'op' module that stylegan2_model.py imports
import types
_op_module = types.ModuleType('op')
_op_module.FusedLeakyReLU   = _FusedLeakyReLU
_op_module.fused_leaky_relu = _fused_leaky_relu_pt
_op_module.upfirdn2d        = lambda inp, k, up=1, down=1, pad=(0,0): \
                                   _upfirdn2d_pt(inp, k,
                                       *(up, up) if isinstance(up, int) else up,
                                       *(down, down) if isinstance(down, int) else down,
                                       *(pad[0], pad[1], pad[0], pad[1]) if len(pad)==2 else pad)

# Fake conv2d_gradfix — just delegates to F.conv2d / F.conv_transpose2d
_conv_module = types.ModuleType('conv2d_gradfix')
_conv_module.conv2d           = lambda inp, w, **kw: F.conv2d(inp, w, **kw)
_conv_module.conv_transpose2d = lambda inp, w, **kw: F.conv_transpose2d(inp, w, **kw)
_op_module.conv2d_gradfix     = _conv_module

sys.modules['op']              = _op_module
sys.modules['conv2d_gradfix']  = _conv_module

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Import the real rosinality Generator from stylegan2_model.py
# ─────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

spec = importlib.util.spec_from_file_location(
    'stylegan2_model',
    os.path.join(_HERE, 'stylegan2_model.py')
)
_sg2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_sg2)
_RosiGenerator = _sg2.Generator   # the real rosinality Generator class


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Public wrapper
# ─────────────────────────────────────────────────────────────────────────────

class StyleGAN2Generator(nn.Module):
    """
    Thin wrapper around the rosinality Generator for 256×256 FFHQ.

    Parameters
    ----------
    ckpt_path : str | None
        Path to ffhq-256-config-e.pt.  If None, weights are random.
    style_dim : int
        W-latent dimension (512).
    """

    def __init__(self, ckpt_path: str = None, style_dim: int = 512):
        super().__init__()
        self.style_dim = style_dim

        # ffhq-256-config-e: channel_multiplier=1 (64ch at 256px — confirmed from checkpoint)
        self._G = _RosiGenerator(
            size=256,
            style_dim=style_dim,
            n_mlp=8,
            channel_multiplier=1,
        )

        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

        # Expose n_latent from inner generator
        self.synthesis = self._G   # alias so train.py can do G.synthesis.n_latent
        # n_latent is already set by rosinality:  log_size*2 - 2 = 14
        assert hasattr(self._G, 'n_latent'), "n_latent missing from rosinality Generator"

        # Freeze everything
        for p in self.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('g_ema', ckpt)

        # Checkpoint key structure:
        #   mapping.mapping.N.{weight,bias}  → style.N.{weight,bias}
        #   synthesis.conv1.*                → conv1.*
        #   synthesis.convs.*                → convs.*
        #   synthesis.to_rgb1.*              → to_rgb1.*
        #   synthesis.to_rgbs.*              → to_rgbs.*
        #   synthesis.input.*                → input.*
        remapped = {}
        for k, v in state.items():
            if k.startswith('mapping.mapping.'):
                new_k = 'style.' + k[len('mapping.mapping.'):]
            elif k.startswith('synthesis.'):
                new_k = k[len('synthesis.'):]        # strip 'synthesis.' prefix
            else:
                new_k = k

            # The checkpoint stores:
            #   convN.bias  → model expects  convN.activate.bias   (FusedLeakyReLU param)
            #   convN.noise → model expects  convN.noise.weight     (NoiseInjection param)
            # Apply these renames for conv1 and convs.*
            import re
            new_k = re.sub(r'^(conv1|convs\.\d+)\.bias$',   r'\1.activate.bias',  new_k)
            new_k = re.sub(r'^(conv1|convs\.\d+)\.noise$',  r'\1.noise.weight',   new_k)

            # Only activate.bias (FusedLeakyReLU) is stored as 4D (1,C,1,1) in checkpoint
            # but the model stores it as 1D (C,). All other biases are correctly 4D in both.
            if new_k.endswith('.activate.bias') and v.ndim == 4:
                v = v.squeeze()   # (1, C, 1, 1) -> (C,)

            remapped[new_k] = v

        missing, unexpected = self._G.load_state_dict(remapped, strict=False)
        loaded = len(state) - len(unexpected)
        print(f'[StyleGAN2] Loaded {loaded}/{len(state)} keys from {path}')
        if missing:
            print(f'  Missing  : {missing[:8]}')
        if unexpected:
            print(f'  Unexpected: {unexpected[:5]}')
        if loaded < int(0.70 * len(state)):
            raise RuntimeError(
                f'[StyleGAN2] Only {loaded}/{len(state)} keys loaded — '
                'checkpoint architecture mismatch!'
            )

    # ------------------------------------------------------------------ #
    @property
    def n_latent(self):
        return self._G.n_latent

    # ------------------------------------------------------------------ #
    @property
    def mapping(self):
        """Expose mapping network so setup_stylegan.py can call G.mapping(z)."""
        class _MappingAdapter:
            def __init__(self, style_fn):
                self._f = style_fn
            def __call__(self, z):
                return self._f(z)
        return _MappingAdapter(self._G.style)

    # ------------------------------------------------------------------ #
    def forward(self, w_plus, input_is_latent: bool = True):
        """
        Parameters
        ----------
        w_plus : (B, n_latent, 512) if input_is_latent=True
                 (B, 512)           if input_is_latent=False  (raw z)
        Returns
        -------
        (B, 3, 256, 256) in [-1, 1]  (NO tanh — rosinality returns raw)
        """
        if input_is_latent:
            # rosinality expects a LIST of style tensors or a single W+ (B, n_latent, 512)
            imgs, _ = self._G(
                [w_plus],
                input_is_latent=True,
                randomize_noise=False,
            )
        else:
            imgs, _ = self._G(
                [w_plus],
                input_is_latent=False,
                randomize_noise=False,
            )
        return imgs
