"""
convert_pkl_to_pt.py
--------------------
Converts a NVlabs TF StyleGAN2 .pkl into the .pt format for our wrapper.
Runs WITHOUT TensorFlow installed.

Usage:
  python convert_pkl_to_pt.py
  python convert_pkl_to_pt.py --pkl checkpoints/ffhq-256-config-e-003810.pkl
"""

import argparse
import math
import os
import pickle
import sys
import types

import numpy as np
import torch

PKL_PATH = 'checkpoints/ffhq-256-config-e-003810.pkl'
OUT_PATH = 'checkpoints/ffhq-256-config-e.pt'


# ─────────────────────────────────────────────────────────────────────────────
# Minimal stubs: only what pickle needs to reconstruct the object graph.
# We must NOT stub 'tensorflow' itself — that breaks torchvision later.
# ─────────────────────────────────────────────────────────────────────────────

class _DummyNetwork:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): self.__dict__.update(s)


class _DummyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Intercept NVlabs network classes
        if name in ('Network', 'Generator', 'Discriminator'):
            return _DummyNetwork
        # For dnnlib/tflib modules, install a stub and return a generic class
        if module.startswith('dnnlib') or module.startswith('tflib'):
            if module not in sys.modules:
                sys.modules[module] = types.ModuleType(module)
            class _Stub:
                def __new__(cls, *a, **k): return object.__new__(cls)
                def __init__(self, *a, **k): pass
            return _Stub
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            class _Stub:
                def __new__(cls, *a, **k): return object.__new__(cls)
                def __init__(self, *a, **k): pass
            return _Stub


def load_pkl(path: str):
    # Pre-install only dnnlib stubs (NOT tensorflow — that breaks torchvision)
    for mod in ['dnnlib', 'dnnlib.tflib', 'dnnlib.tflib.network', 'dnnlib.util']:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    print(f'[1] Loading {path}  ({os.path.getsize(path)/1e6:.1f} MB) ...')
    with open(path, 'rb') as f:
        data = _DummyUnpickler(f).load()
    print(f'    type={type(data).__name__}  len={len(data) if hasattr(data,"__len__") else "?"}')
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Build name->array from the nested component.variables lists
# Each item in variables is a (name:str, array:np.ndarray) tuple
# ─────────────────────────────────────────────────────────────────────────────

def build_vars(g_ema) -> dict:
    out = {}

    def walk(net, prefix=''):
        variables = getattr(net, 'variables', [])
        for item in variables:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                name, arr = item[0], item[1]
                if isinstance(name, str) and isinstance(arr, np.ndarray):
                    out[name] = arr

        for sub in getattr(net, 'components', {}).values():
            walk(sub)

    walk(g_ema)
    print(f'[2] Collected {len(out)} weight arrays.')
    # print first few to help diagnose key naming
    for i, (k, v) in enumerate(sorted(out.items())):
        if i >= 15: print(f'    ... ({len(out)-15} more)'); break
        print(f'    {k:55s}  {str(v.shape):20s}  {v.dtype}')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Map TF names -> rosinality state_dict keys
# The PKL has keys WITHOUT the "G_synthesis/" prefix, e.g. "4x4/Const/const"
# ─────────────────────────────────────────────────────────────────────────────

def _t(arr, perm=None):
    if perm: arr = arr.transpose(perm)
    return torch.from_numpy(arr.copy())


def convert(vars_dict: dict, size: int = 256, n_mlp: int = 8) -> dict:
    """Map TF variable names to our wrapper's state dict keys.
    
    Wrapper key structure:
      mapping.mapping.{1..8}.weight / .bias     (MappingNetwork)
      synthesis.input.input                      (ConstantInput)
      synthesis.conv1.conv.weight / .noise / .bias / .conv.modulation.*
      synthesis.convs.{i}.conv.weight / ...     (StyledConv)
      synthesis.to_rgb1.*                        (ToRGB)
      synthesis.to_rgbs.{i}.*                   (ToRGB)
      synthesis.noises.noise_{i}                (noise buffers)
    """
    sd = {}
    log_size = int(math.log(size, 2))  # 8 for 256px

    sample_keys = sorted(vars_dict.keys())
    has_G_synth = any(k.startswith('G_synthesis/') for k in sample_keys)
    has_G_map   = any(k.startswith('G_mapping/')   for k in sample_keys)
    synth_prefix = 'G_synthesis/' if has_G_synth else ''
    map_prefix   = 'G_mapping/'   if has_G_map   else ''
    print(f'    Key style: synth_prefix="{synth_prefix}", map_prefix="{map_prefix}"')

    def get(name):
        return vars_dict.get(name)

    # -- Mapping network: wrapper uses 'mapping.mapping.{i+1}.weight' --
    for i in range(n_mlp):
        w = get(f'{map_prefix}Dense{i}/weight')
        b = get(f'{map_prefix}Dense{i}/bias')
        if w is not None: sd[f'mapping.mapping.{i+1}.weight'] = _t(w, (1,0))
        if b is not None: sd[f'mapping.mapping.{i+1}.bias']   = _t(b)

    # -- Synthesis constant input: wrapper uses 'synthesis.input.input' --
    c = get(f'{synth_prefix}4x4/Const/const')
    if c is not None: sd['synthesis.input.input'] = _t(c)

    def add_torgb(tf_name, ros_name):
        """ros_name like 'synthesis.to_rgb1' or 'synthesis.to_rgbs.0'"""
        w  = get(f'{synth_prefix}{tf_name}/weight')
        mw = get(f'{synth_prefix}{tf_name}/mod_weight')
        mb = get(f'{synth_prefix}{tf_name}/mod_bias')
        b  = get(f'{synth_prefix}{tf_name}/bias')
        # TF ToRGB weight is (kH, kW, in, out) -> torch (out, in, kH, kW) -> unsqueeze -> (1, out, in, kH, kW)
        if w  is not None: sd[f'{ros_name}.conv.weight']             = _t(w, (3,2,0,1)).unsqueeze(0)
        if mw is not None: sd[f'{ros_name}.conv.modulation.weight']  = _t(mw, (1,0))
        if mb is not None: sd[f'{ros_name}.conv.modulation.bias']    = _t(mb) + 1
        if b  is not None: sd[f'{ros_name}.bias']                    = _t(b).reshape(1,-1,1,1)

    def add_modconv(tf_name, ros_name, flip=False):
        """ros_name like 'synthesis.conv1' or 'synthesis.convs.0'"""
        w    = get(f'{synth_prefix}{tf_name}/weight')
        mw   = get(f'{synth_prefix}{tf_name}/mod_weight')
        mb   = get(f'{synth_prefix}{tf_name}/mod_bias')
        ns   = get(f'{synth_prefix}{tf_name}/noise_strength')
        bias = get(f'{synth_prefix}{tf_name}/bias')
        if w is not None:
            tw = _t(w, (3,2,0,1)).unsqueeze(0)
            if flip: tw = torch.flip(tw, [3,4])
            sd[f'{ros_name}.conv.weight'] = tw
        if mw   is not None: sd[f'{ros_name}.conv.modulation.weight'] = _t(mw, (1,0))
        if mb   is not None: sd[f'{ros_name}.conv.modulation.bias']   = _t(mb) + 1
        if ns   is not None:
            ns_arr = np.atleast_1d(np.array(ns, dtype=np.float32))
            # Wrapper: 'synthesis.conv1.noise' or 'synthesis.convs.N.noise'
            sd[f'{ros_name}.noise'] = _t(ns_arr)
        # Wrapper: 'synthesis.conv1.bias' or 'synthesis.convs.N.bias' — shape (1,C,1,1)
        if bias is not None: sd[f'{ros_name}.bias'] = _t(bias).reshape(1,-1,1,1)

    add_torgb  ('4x4/ToRGB',  'synthesis.to_rgb1')
    add_modconv('4x4/Conv',   'synthesis.conv1')

    conv_i = 0
    for i in range(log_size - 2):
        reso = 4 * 2**(i+1)
        add_modconv(f'{reso}x{reso}/Conv0_up', f'synthesis.convs.{conv_i}',   flip=True)
        add_modconv(f'{reso}x{reso}/Conv1',    f'synthesis.convs.{conv_i+1}', flip=False)
        add_torgb  (f'{reso}x{reso}/ToRGB',    f'synthesis.to_rgbs.{i}')
        conv_i += 2

    # No separate noises dict — noises stored inline above.
    # If noise_strength was not in PKL per-conv, fill with zeros (learnable scalars).
    noise_keys_expected = ['synthesis.conv1.noise'] + [f'synthesis.convs.{i}.noise' for i in range(12)]
    for k in noise_keys_expected:
        if k not in sd:
            sd[k] = torch.zeros(1)  # rosinality noise is a scalar weight

    print(f'[3] Mapped {len(sd)} tensors.')
    return sd


# ─────────────────────────────────────────────────────────────────────────────
# Verify the saved .pt loads into our wrapper correctly
# ─────────────────────────────────────────────────────────────────────────────

def verify(path: str):
    # Remove dnnlib stubs before importing torchvision-dependent modules
    for mod in list(sys.modules.keys()):
        if mod.startswith('dnnlib') or mod.startswith('tflib'):
            del sys.modules[mod]

    print('\n[verify] Loading .pt and running forward pass ...')
    from models.stylegan2_wrapper import StyleGAN2Generator
    G = StyleGAN2Generator(ckpt_path=path, style_dim=512).eval()
    # 256px StyleGAN2: 15 style inputs needed by synthesis (2 + 6*2 convs + 1 last toRGB)
    n_styles = 15
    w = torch.randn(1, n_styles, 512)
    with torch.no_grad():
        img = G(w, input_is_latent=True)
    print(f'    n_styles={n_styles}  output={tuple(img.shape)}  range=[{img.min():.3f},{img.max():.3f}]')
    print('    PASSED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl',       default=PKL_PATH)
    parser.add_argument('--out',       default=OUT_PATH)
    parser.add_argument('--size',      type=int, default=256)
    parser.add_argument('--n_mlp',     type=int, default=8)
    parser.add_argument('--no_verify', action='store_true')
    args = parser.parse_args()

    print('='*60)
    print('  StyleGAN2 TF-PKL -> PyTorch .pt Converter')
    print('='*60)

    data   = load_pkl(args.pkl)
    g_ema  = data[-1]                 # (G_train, D_train, G_ema) tuple
    vd     = build_vars(g_ema)
    sd     = convert(vd, size=args.size, n_mlp=args.n_mlp)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save({'g_ema': sd}, args.out)
    print(f'[save] {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)')

    if not args.no_verify:
        verify(args.out)

    print('\nDone! Now run:')
    print('  python setup_stylegan.py --skip_download')
