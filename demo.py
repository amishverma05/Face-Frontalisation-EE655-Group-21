"""
demo.py
───────────────────────────────────────────────────────────────────────────────
Live interactive demo for the Hybrid Face Frontalization pipeline.
Pose-Conditioned FPN Encoder + Frozen StyleGAN2 Generator (FFHQ-256)

Launch:
    .\venv\Scripts\python demo.py

Opens at: http://localhost:7860
───────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import glob
import torch
import numpy as np
import gradio as gr
from pathlib import Path
from PIL import Image
from torchvision import transforms as T

# ── Model imports ─────────────────────────────────────────────────────────────
from models.generator import PoseConditionedEncoder
from models.stylegan2_wrapper import StyleGAN2Generator

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT     = "checkpoints/best.pth"
STYLEGAN_CKPT  = "checkpoints/ffhq-256-config-e.pt"
W_AVG_PATH     = "checkpoints/frontal_latent_avg.pt"
ARCH_IMG_PATH  = "assets/arch_diagram.png"
SAMPLES_DIR    = "samples"
IMG_SIZE       = 256
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_STYLES       = 14

# ── CSS Theme ─────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

body, .gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}

.header-box {
    background: linear-gradient(90deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 8px;
    text-align: center;
}

.header-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.header-subtitle {
    color: rgba(255,255,255,0.6);
    font-size: 0.95rem;
    margin-top: 6px;
}

.meta-card {
    background: rgba(30, 30, 50, 0.7);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 16px;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.75);
    line-height: 1.8;
}

.frontalize-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 14px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.frontalize-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
}

label { color: rgba(255,255,255,0.85) !important; font-weight: 500 !important; }
.gr-slider input[type=range]  { accent-color: #818cf8; }
"""

# ── Model Loader (runs once at startup) ───────────────────────────────────────
_encoder  = None
_stylegan = None
_w_avg    = None
_epoch_loaded = "N/A"

def load_model():
    global _encoder, _stylegan, _w_avg, _epoch_loaded

    if not os.path.exists(CHECKPOINT):
        print(f"\n[Demo] WARNING: No checkpoint found at '{CHECKPOINT}'.")
        print(f"[Demo] The model will not produce real results until training saves a checkpoint.\n")
        return False

    print(f"[Demo] Loading model from {CHECKPOINT} ...")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    _epoch_loaded = ckpt.get("epoch", "?")

    _encoder = PoseConditionedEncoder(n_styles=N_STYLES, style_dim=512).to(DEVICE)
    _encoder.load_state_dict(ckpt["encoder"])
    _encoder.eval()

    _stylegan = StyleGAN2Generator(STYLEGAN_CKPT).to(DEVICE)
    _stylegan.eval()

    if os.path.exists(W_AVG_PATH):
        _w_avg = torch.load(W_AVG_PATH, map_location=DEVICE, weights_only=False)
    else:
        _w_avg = torch.zeros(1, N_STYLES, 512, device=DEVICE)

    print(f"[Demo] Model ready! Epoch {_epoch_loaded} | Device: {DEVICE}")
    return True


# ── Inference Function ─────────────────────────────────────────────────────────
def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).detach().cpu().clamp(-1, 1)
    t = ((t + 1.0) / 2.0 * 255).to(torch.uint8)
    return Image.fromarray(t.permute(1, 2, 0).numpy())


def frontalize_image(input_img: Image.Image, yaw: float, pitch: float):
    """Main inference callback for Gradio."""
    if input_img is None:
        return None, "⚠️ Please upload an image first."

    if _encoder is None or _stylegan is None:
        return None, "⚠️ No trained checkpoint found. Please complete at least 1 training epoch (`.\main.bat`)."

    tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

    t0 = time.perf_counter()
    with torch.no_grad():
        img_t  = tf(input_img.convert("RGB")).unsqueeze(0).to(DEVICE)
        yaw_t  = torch.tensor([yaw],   dtype=torch.float32, device=DEVICE)
        pitch_t = torch.tensor([pitch], dtype=torch.float32, device=DEVICE)

        w_pred  = _encoder(img_t, yaw_t, pitch_t, w_avg=_w_avg)
        fake    = _stylegan(w_pred, input_is_latent=True)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    result_img = tensor_to_pil(fake)

    meta = (
        f" **Frontalization Complete**\n\n"
        f" Inference time: **{elapsed_ms:.0f} ms**\n"
        f" Device: **{DEVICE}** ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})\n"
        f" Epoch loaded: **{_epoch_loaded}**\n"
        f" Yaw: **{yaw:.1f}°** | Pitch: **{pitch:.1f}°**"
    )

    return result_img, meta


# ── Gallery — pre-stored examples + training samples ─────────────────────────
EXAMPLES_DIR = "assets/examples"   # <── drop your own face images here!

def get_example_images():
    """Load pre-stored examples first, then fall back to training samples."""
    # 1. Check assets/examples/ for user-supplied profile face images
    custom = []
    if os.path.exists(EXAMPLES_DIR):
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            custom.extend(glob.glob(os.path.join(EXAMPLES_DIR, ext)))

    if len(custom) >= 2:
        return sorted(custom)[:6]   # show up to 6 custom examples

    # 2. Fall back to recent training output grids from samples/
    files = sorted(glob.glob(f"{SAMPLES_DIR}/epoch*.jpg"), reverse=True)
    seen, examples = set(), []
    for f in files:
        epoch = Path(f).stem.split("_")[0]
        if epoch not in seen:
            seen.add(epoch)
            examples.append(f)
        if len(examples) >= 4:
            break
    return examples


# ── Build UI ──────────────────────────────────────────────────────────────────
def build_demo():
    examples = get_example_images()

    with gr.Blocks(title="Face Frontalization — EE655 Group 21") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-box">
            <p class="header-title">Face Frontalization</p>
            <p class="header-subtitle">
                Hybrid Pose-Conditioned FPN Encoder + Frozen StyleGAN2 Generator &nbsp;|&nbsp;
                EE655 Group Project — Group 21
            </p>
        </div>
        """)

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════════
            # Tab 1 — Live Demo
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("Live Frontalization"):

                with gr.Row():

                    # ── Left column — controls ─────────────────────────────
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Profile Face")
                        input_image = gr.Image(
                            type="pil",
                            label="Profile / Side-View Image",
                            height=280,
                        )

                        gr.Markdown("### Pose Parameters")
                        yaw_slider = gr.Slider(
                            minimum=-90, maximum=90, value=45, step=1,
                            label="Yaw Angle (°)   —   left(-) / right(+)",
                        )
                        pitch_slider = gr.Slider(
                            minimum=-30, maximum=30, value=0, step=1,
                            label="Pitch Angle (°)   —   down(-) / up(+)",
                        )

                        gr.Markdown("""
                        > **Yaw (Horizontal Rotation):** How much the face is turned sideways.<br>
                        > &nbsp;&nbsp;&nbsp;&nbsp; 0° = Fully frontal, 45° = 3/4 profile view, 90° = Full side profile view.<br>
                        > **Pitch (Vertical Rotation):** How much the face is tilting up or down.
                        """)

                        frontalize_btn = gr.Button(
                            "Frontalize  →",
                            elem_classes=["frontalize-btn"]
                        )

                    # ── Right column — output ──────────────────────────────
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Frontal Face")
                        output_image = gr.Image(
                            type="pil",
                            label="Synthesized Frontal Output",
                            height=280,
                        )

                        meta_box = gr.Markdown(
                            value="*Run the model to see results...*",
                            elem_classes=["meta-card"],
                        )

                # ── Examples gallery ──────────────────────────────────────
                if examples:
                    gr.Markdown("---\n### Training Sample Gallery\n*Click any image below to auto-load it*")
                    gr.Examples(
                        examples=examples,
                        inputs=input_image,
                        label="Recent Training Output Grids",
                    )

                # ── Wire up the button ────────────────────────────────────
                frontalize_btn.click(
                    fn=frontalize_image,
                    inputs=[input_image, yaw_slider, pitch_slider],
                    outputs=[output_image, meta_box],
                )

            with gr.Tab("🧠  Model Architecture"):
                gr.HTML("""
<style>
@keyframes float-in { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
.arch2 { background:radial-gradient(ellipse at top,#1a0533 0%,#080818 65%); padding:28px 16px 36px; font-family:'Inter',system-ui,sans-serif; color:#fff; animation:float-in .45s ease; }
.arch2-h1 { text-align:center; font-size:1.3rem; font-weight:800; letter-spacing:-.01em; background:linear-gradient(90deg,#818cf8,#c084fc,#f472b6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:4px; }
.arch2-h2 { text-align:center; font-size:0.76rem; color:rgba(255,255,255,.35); margin-bottom:26px; letter-spacing:.05em; }
.pipeline2 { display:flex; align-items:center; gap:5px; justify-content:center; padding-bottom:6px; padding-top:140px; margin-top:-120px; }
/* tooltip */
.tt2 { position:relative; cursor:pointer; }
.tt2 > .tip2 { visibility:hidden; opacity:0; position:absolute; bottom:calc(100% + 10px); left:50%; transform:translateX(-50%); background:rgba(8,8,24,.97); border:1px solid rgba(139,92,246,.5); border-radius:11px; padding:13px 15px; width:245px; font-size:0.72rem; font-weight:400; line-height:1.65; color:rgba(255,255,255,.88); text-align:left; pointer-events:none; z-index:9999; transition:opacity .15s; backdrop-filter:blur(14px); box-shadow:0 10px 36px rgba(0,0,0,.7); }
.tt2 > .tip2::after { content:''; position:absolute; top:100%; left:50%; transform:translateX(-50%); border:7px solid transparent; border-top-color:rgba(139,92,246,.5); }
.tt2:hover > .tip2 { visibility:visible; opacity:1; }
.tt2:hover { z-index:100; }
/* badges */
.bdg { display:inline-block; font-size:.58rem; font-weight:700; padding:2px 7px; border-radius:20px; letter-spacing:.06em; text-transform:uppercase; margin-bottom:4px; }
.bdg-fr { background:rgba(56,189,248,.15); border:1px solid #38bdf8; color:#7dd3fc; }
.bdg-tr { background:rgba(74,222,128,.15); border:1px solid #4ade80; color:#86efac; }
.bdg-nv { background:rgba(251,191,36,.15); border:1px solid #fbbf24; color:#fde68a; }
/* tensor chip */
.tc { display:inline-block; background:rgba(0,0,0,.45); border:1px dashed rgba(148,163,184,.4); border-radius:6px; font-size:.62rem; padding:3px 8px; color:#93c5fd; font-family:monospace; letter-spacing:.02em; margin-top:4px; }
/* conn arrow */
.conn2 { display:flex; align-items:center; flex-shrink:0; opacity:.45; }
.conn2 svg { width:30px; }
/* input/output */
.blk-io { border-radius:13px; padding:15px 10px; text-align:center; font-size:2.3rem; transition:transform .2s,box-shadow .2s; position:relative; width:78px; }
.blk-in  { background:linear-gradient(145deg,#1e1b4b,#2d2470); border:2px solid #6366f1; color:#c7d2fe; }
.blk-in:hover  { transform:translateY(-5px); box-shadow:0 0 22px #6366f1; }
.blk-out { background:linear-gradient(145deg,#0f172a,#1e293b); border:2px solid #475569; color:#94a3b8; }
.blk-out:hover { transform:translateY(-5px); box-shadow:0 0 18px #475569; }
.io-label { font-size:.6rem; margin-top:4px; opacity:.7; }
/* encoders */
.enc-grp { background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.09); border-radius:14px; padding:10px; display:flex; flex-direction:column; gap:7px; width:172px; flex-shrink:0; }
.enc-grp-lbl { font-size:.62rem; color:rgba(255,255,255,.32); text-align:center; text-transform:uppercase; letter-spacing:.09em; margin-bottom:1px; }
.enc-blk { border-radius:11px; padding:10px 11px; transition:transform .2s,box-shadow .2s; position:relative; cursor:pointer; }
.enc-blk:hover { transform:translateY(-4px); }
.enc-blk .enc-title { font-size:.76rem; font-weight:700; margin-bottom:2px; }
.enc-sub { background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.11); border-radius:6px; padding:4px 8px; font-size:.65rem; color:rgba(255,255,255,.58); margin-top:3px; font-weight:400; }
.eg { background:linear-gradient(135deg,#7c2d12,#c2410c); border:1.5px solid #fb923c; color:#fed7aa; }
.eg:hover { box-shadow:0 0 20px #fb923c66; }
.ei { background:linear-gradient(135deg,#3b0764,#6d28d9); border:1.5px solid #a855f7; color:#e9d5ff; }
.ei:hover { box-shadow:0 0 20px #a855f766; }
.ep { background:linear-gradient(135deg,#1e3a8a,#1d4ed8); border:1.5px solid #60a5fa; color:#bfdbfe; }
.ep:hover { box-shadow:0 0 20px #60a5fa66; }
/* refinement */
.ref-blk { background:linear-gradient(165deg,#1e0845,#3b0764); border:2px solid #7c3aed; border-radius:13px; padding:11px; width:182px; flex-shrink:0; display:flex; flex-direction:column; gap:5px; transition:box-shadow .2s; }
.ref-blk:hover { box-shadow:0 0 28px #7c3aed55; }
.ref-hdr { font-size:.79rem; font-weight:800; color:#c4b5fd; text-align:center; border-bottom:1px solid rgba(196,181,253,.22); padding-bottom:7px; margin-bottom:1px; letter-spacing:.02em; }
.rs { background:rgba(0,0,0,.35); border:1px solid rgba(196,181,253,.17); border-radius:8px; padding:7px 10px; font-size:.7rem; color:#ede9fe; cursor:pointer; transition:all .15s; position:relative; }
.rs:hover { background:rgba(124,58,237,.32); transform:scale(1.02); }
.rs:hover .tip2 { visibility:visible; opacity:1; }
.rs-lbl { font-weight:700; font-size:.72rem; margin-bottom:1px; }
.rs-sub { color:rgba(255,255,255,.42); font-size:.62rem; font-family:monospace; }
/* stylegan2 */
.sg-blk { background:linear-gradient(165deg,#022c22,#065f46); border:2px solid #10b981; border-radius:13px; padding:11px; width:152px; flex-shrink:0; display:flex; flex-direction:column; gap:5px; transition:box-shadow .2s; }
.sg-blk:hover { box-shadow:0 0 28px #10b98155; }
.sg-hdr { font-size:.77rem; font-weight:800; color:#6ee7b7; text-align:center; border-bottom:1px solid rgba(110,231,183,.22); padding-bottom:7px; margin-bottom:1px; }
.ss { background:rgba(0,0,0,.28); border:1px solid rgba(110,231,183,.18); border-radius:8px; padding:6px 9px; font-size:.68rem; color:#d1fae5; cursor:pointer; transition:all .15s; position:relative; }
.ss:hover { background:rgba(16,185,129,.32); transform:scale(1.02); }
.ss:hover .tip2 { visibility:visible; opacity:1; }
.ss-res { font-size:.59rem; color:rgba(110,231,183,.52); font-family:monospace; }
/* loss */
.loss-sec { margin-top:28px; }
.loss-hdr2 { text-align:center; font-size:.76rem; font-weight:700; color:rgba(255,255,255,.38); text-transform:uppercase; letter-spacing:.1em; margin-bottom:14px; }
.lrow { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; }
.lc { border-radius:12px; padding:12px 15px; min-width:148px; font-size:.7rem; cursor:pointer; transition:all .2s; position:relative; text-align:center; }
.lc:hover { transform:translateY(-4px); }
.lc:hover > .tip2 { visibility:visible; opacity:1; }
.lc-ttl { font-weight:800; font-size:.74rem; margin-bottom:6px; }
.lc-eq  { font-family:monospace; font-size:.64rem; opacity:.68; line-height:1.75; }
.lr { background:rgba(190,18,60,.18); border:1.5px solid #f43f5e; color:#fda4af; } .lr:hover{box-shadow:0 6px 22px #f43f5e44;}
.lp { background:rgba(109,40,217,.18); border:1.5px solid #a855f7; color:#d8b4fe; } .lp:hover{box-shadow:0 6px 22px #a855f744;}
.lb { background:rgba(37,99,235,.18);  border:1.5px solid #3b82f6; color:#93c5fd; } .lb:hover{box-shadow:0 6px 22px #3b82f644;}
.lg { background:rgba(5,150,105,.18);  border:1.5px solid #10b981; color:#6ee7b7; } .lg:hover{box-shadow:0 6px 22px #10b98144;}
.arch-sep2 { border:none; border-top:1px solid rgba(255,255,255,.07); margin:22px 0; }
</style>

<div class="arch2">
  <div class="arch2-h1">Detailed Hybrid StyleGAN2 Face Frontalization Architecture</div>
  <div class="arch2-h2">✦ Hover any block for tensor dimensions &amp; technical details ✦</div>

  <div class="pipeline2">

    <!-- INPUT -->
    <div class="blk-io blk-in tt2">
      🧑
      <div class="io-label">Profile<br>3×256×256</div>
      <div class="tip2" style="left:20px; transform:none;">
        <strong>Profile Face Image</strong><br>
        Tensor: <code>[B, 3, 256, 256]</code><br>
        Normalized [0,255] → [-1, 1]<br><br>
        Supports any horizontal yaw angle 0°–90°. Yaw &amp; pitch values are passed explicitly as pose conditioning inputs.
      </div>
    </div>

    <div class="conn2"><svg viewBox="0 0 30 10" fill="none"><path d="M0 5h22M18 1.5l6 3.5-6 3.5" stroke="white" stroke-width="1.5" stroke-linecap="round"/></svg></div>

    <!-- ENCODERS -->
    <div class="enc-grp">
      <div class="enc-grp-lbl">Encoder Trio</div>

      <div class="enc-blk eg tt2">
        <div class="enc-title">Geometry Encoder</div>
        <div class="bdg bdg-tr">🔥 Trainable</div>
        <div class="enc-sub">Landmark Extractor (Heatmaps)</div>
        <div class="enc-sub">Conv Blocks + Flatten + FC</div>
        <div class="tc">f_geo ∈ ℝ⁵¹²</div>
        <div class="tip2">
          <strong>Geometry Encoder</strong><br>
          Produces 68 facial landmark heatmaps from the profile image, then flattens into a 512-D geometry descriptor.<br><br>
          <code>f_geo ∈ ℝ^512</code> encodes eye spacing, jaw angle, and nose bridge position independent of lighting or skin tone.
        </div>
      </div>

      <div class="enc-blk ei tt2" style="margin-top:6px">
        <div class="enc-title">Identity Encoder</div>
        <div class="bdg bdg-fr">❄️ Frozen</div>
        <div class="enc-sub">ArcFace IR-SE50 Backbone</div>
        <div class="enc-sub">Global Average Pooling</div>
        <div class="tc">f_id ∈ ℝ⁵¹²</div>
        <div class="tip2">
          <strong>Identity Encoder (ArcFace)</strong><br>
          Pre-trained <b>IR-SE50</b> backbone with ArcFace margin loss. Extracts a 512-D embedding invariant to pose &amp; lighting.<br><br>
          <code>f_id ∈ ℝ^512</code> is the Key/Value in the cross-attention — it ensures the output face matches the input person's identity.
        </div>
      </div>

      <div class="enc-blk ep tt2" style="margin-top:6px">
        <div class="enc-title">pSp Encoder</div>
        <div class="bdg bdg-tr">🔥 Trainable</div>
        <div class="enc-sub">ResNet-50 Feature Pyramid (FPN)</div>
        <div style="display:flex;gap:3px;margin-top:4px">
          <div class="tc" style="flex:1;font-size:.57rem">S 2–2</div>
          <div class="tc" style="flex:1;font-size:.57rem">S 3–6</div>
          <div class="tc" style="flex:1;font-size:.57rem">S 7–17</div>
        </div>
        <div class="enc-sub">18 × Map2Style Linear Modules</div>
        <div class="tc">W_coarse [B, 18, 512]</div>
        <div class="tip2">
          <strong>pSp (pixel2style2pixel) Encoder</strong><br>
          ResNet-50 FPN extracts features at 3 scales mapped to 18 style vectors:<br>
          • <b>Styles 2–2:</b> coarse head structure<br>
          • <b>Styles 3–6:</b> mid-level features (eyes, nose)<br>
          • <b>Styles 7–17:</b> fine texture (pores, hair)<br><br>
          Output: <code>W_coarse [B, 18, 512]</code>
        </div>
      </div>
    </div>

    <div class="conn2"><svg viewBox="0 0 30 10" fill="none"><path d="M0 5h22M18 1.5l6 3.5-6 3.5" stroke="white" stroke-width="1.5" stroke-linecap="round"/></svg></div>

    <!-- REFINEMENT -->
    <div class="ref-blk tt2">
      <div class="ref-hdr">⚡ Hybrid Latent Refinement</div>

      <div class="rs tt2">
        <div class="rs-lbl">① Condition Embedding</div>
        <div class="rs-sub">Cond = [f_geo, f_id] → [B,2,512]</div>
        <div class="tip2">
          <strong>Condition Embedding Concat</strong><br>
          Stacks geometry + identity into a joint conditioning tensor:<br>
          <code>Cond = stack(f_geo, f_id)</code><br>
          Shape: <code>[B, 2, 512]</code><br><br>
          Becomes the <b>Key</b> and <b>Value</b> in the MHA cross-attention.
        </div>
      </div>

      <div class="rs tt2">
        <div class="rs-lbl">② Multi-Head Cross-Attention</div>
        <div class="rs-sub">Q: W_coarse · K,V: Cond · Softmax(QKᵀ/√d)V</div>
        <div class="tip2">
          <strong>Cross-Attention (MHA)</strong><br>
          W_coarse serves as the Query; Cond as Key &amp; Value:<br>
          <code>Attn = Softmax(QKᵀ / √d) · V</code><br><br>
          Each style vector attends to the most relevant identity features, steering the latent toward the correct face geometry.
        </div>
      </div>

      <div class="rs tt2">
        <div class="rs-lbl">③ Feed-Forward MLP</div>
        <div class="rs-sub">Linear → GELU → Linear → ΔW [B,18,512]</div>
        <div class="tip2">
          <strong>Feed-Forward Network</strong><br>
          Two-layer MLP with GELU activation computes the latent correction ΔW.<br>
          Shape: <code>[B, 18, 512]</code><br><br>
          Provides non-linear feature mixing on top of the attention output.
        </div>
      </div>

      <div class="rs tt2">
        <div class="rs-lbl">④ Gated Residual</div>
        <div class="rs-sub">W_ref = W_coarse + γ · ΔW</div>
        <div class="tip2">
          <strong>Gated Residual Connection</strong><br>
          Learnable scalar gate γ controls the correction magnitude:<br>
          <code>W_ref = W_coarse + γ · ΔW</code><br><br>
          Initializes at γ≈0 — training starts safely from W_coarse and gradually learns refinement.
        </div>
      </div>

      <div class="tc" style="text-align:center;margin-top:3px">W_ref [B, 18, 512] → Decoder</div>
    </div>

    <div class="conn2"><svg viewBox="0 0 30 10" fill="none"><path d="M0 5h22M18 1.5l6 3.5-6 3.5" stroke="white" stroke-width="1.5" stroke-linecap="round"/></svg></div>

    <!-- STYLEGAN2 -->
    <div class="sg-blk">
      <div class="sg-hdr">🧊 StyleGAN2<br><span style="font-weight:400;font-size:.62rem;opacity:.5">Frozen · FFHQ-256</span></div>
      <div class="bdg bdg-fr" style="text-align:center;display:block">❄️ 110/110 Weights Frozen</div>

      <div class="ss tt2">
        Learned Constant<br>
        <span class="ss-res">[512, 4, 4]</span>
        <div class="tip2">
          <strong>Learned Constant Tensor</strong><br>
          StyleGAN2 starts from a fixed learned <code>[512, 4, 4]</code> tensor (unlike StyleGAN1 which used noise). This "blank canvas" is the starting point that all 14 synthesis layers paint onto.
        </div>
      </div>

      <div class="ss tt2">
        ModConv Synthesis<br>
        <span class="ss-res">4×4 → 8×8 → 16×16</span>
        <div class="tip2">
          <strong>Modulated Convolution</strong><br>
          Each synthesis layer modulates its conv kernels via style from W_ref:<br>
          <code>W_mod = S · W_orig</code><br>
          Then demodulates to unit std. Injects head shape, face width, and global structure from W+ coarse layers.
        </div>
      </div>

      <div class="ss tt2">
        Progressive Upsample<br>
        <span class="ss-res">32→64→128→256px</span>
        <div class="tip2">
          <strong>Progressive Upsampling</strong><br>
          Resolution ladder: 4→8→16→32→64→128→256px<br>
          Each step: Bilinear upsample → ModConv → Noise injection → LReLU<br><br>
          Later layers (64–256px) control fine skin texture, eyelashes, and hair.
        </div>
      </div>

      <div class="ss tt2">
        ToRGB Skip Connections
        <div class="tip2">
          <strong>ToRGB at Each Scale</strong><br>
          A 1×1 conv produces an RGB image at each resolution. These are progressively alpha-blended into the final output via skip connections, preventing gradient starvation in shallow layers.
        </div>
      </div>
    </div>

    <div class="conn2"><svg viewBox="0 0 30 10" fill="none"><path d="M0 5h22M18 1.5l6 3.5-6 3.5" stroke="white" stroke-width="1.5" stroke-linecap="round"/></svg></div>

    <!-- OUTPUT -->
    <div class="blk-io blk-out tt2">
      🎭
      <div class="io-label">Frontal<br>3×256×256</div>
      <div class="tip2" style="left:auto;right:0;transform:none">
        <strong>Generated Frontal Face</strong><br>
        Tensor: <code>[B, 3, 256, 256]</code><br>
        Range: [-1, 1]<br><br>
        High-fidelity photorealistic synthesis decoded from W_ref using FFHQ-pretrained StyleGAN2 — guaranteed never to produce non-face outputs.
      </div>
    </div>

  </div><!-- end pipeline -->

  <hr class="arch-sep2">

  <div class="loss-sec">
    <div class="loss-hdr2">✦ Comprehensive Training Objectives ✦</div>
    <div class="lrow">

      <div class="lc lr tt2">
        <div class="lc-ttl">Image Reconstruction</div>
        <div class="lc-eq">L_l1 &nbsp;= ||I_f − I_gt||₁</div>
        <div class="lc-eq">L_lpips = ||VGG(I_f)−VGG(I_gt)||₂</div>
        <div class="tip2">
          <strong>Pixel L1 + LPIPS Loss</strong><br>
          <b>L1 (λ=0.1):</b> Discounted pixel reconstruction — kept low to prevent the auto-encoder shortcut where the model simply copies the sideways background.<br>
          <b>LPIPS (λ=0.5):</b> AlexNet perceptual similarity on face crops — penalizes structural blurring not captured by raw pixel metrics.
        </div>
      </div>

      <div class="lc lp tt2">
        <div class="lc-ttl">Identity &amp; Geometry</div>
        <div class="lc-eq">L_id &nbsp;= 1−CosSim(f_id(I_f), f_id(I_gt))</div>
        <div class="lc-eq">L_geo = ||lmk(I_f)−lmk(I_gt)||₂²</div>
        <div class="tip2">
          <strong>Identity + Landmark Loss</strong><br>
          <b>ID Loss (λ=4.0):</b> Dominant loss — FaceNet cosine similarity forces the output to match the input person's face embedding regardless of background.<br>
          <b>Landmark (λ=0.0):</b> 68-point geometry constraint — infrastructure built, pending integration.
        </div>
      </div>

      <div class="lc lb tt2">
        <div class="lc-ttl">W+ Regularization</div>
        <div class="lc-eq">L_delta = ||W_ref−W_coarse||₂²</div>
        <div class="lc-eq">L_reg &nbsp;= ||W_ref−W_avg||₂²</div>
        <div class="tip2">
          <strong>W-Norm Regularization (λ=0.05)</strong><br>
          <b>Delta Penalty:</b> Prevents the Refinement Block from making catastrophically large corrections to the latent.<br>
          <b>Mean Reg:</b> Draws W_ref toward the frontal face mean — prevents out-of-distribution artifacts like dissolving skin or merged hair.
        </div>
      </div>

      <div class="lc lg tt2">
        <div class="lc-ttl">GAN Objective</div>
        <div class="lc-eq">L_adv = E_z[−log(D(I_f))]</div>
        <div class="lc-eq">R1 &nbsp;&nbsp;= (γ/2)·E[||∇D(I_real)||²]</div>
        <div class="tip2">
          <strong>PatchGAN LSGAN (λ=0.1)</strong><br>
          5-layer PatchDiscriminator evaluates 30×30 crop tiles providing granular texture feedback (pores, hair).<br>
          <b>LSGAN:</b> Least-squares loss avoids vanishing gradients during early training.<br>
          <b>20-epoch warmup:</b> GAN is disabled early so the encoder stabilizes first.
        </div>
      </div>

    </div>
  </div>
</div>
""")


    return demo


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "gradio" not in sys.modules:
        try:
            import gradio
        except ImportError:
            print("Gradio not installed! Run:  .\\venv\\Scripts\\pip install gradio")
            sys.exit(1)

    load_model()

    app = build_demo()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=CSS,
    )
