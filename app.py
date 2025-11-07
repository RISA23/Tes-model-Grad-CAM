import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from streamlit_cropper import st_cropper
import os

# ============================
# CONFIG HALAMAN
# ============================

st.set_page_config(
    page_title="DFU / PU / Normal Classifier",
    page_icon="🩺",
    layout="wide"
)

# ============================
# GLOBAL STYLE (CSS)
# ============================

st.markdown(
    """
    <style>
    /* global font */
    html, body, [class*="css"]  {
        font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    }

    /* card style */
    .metric-card {
        padding: 14px 18px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 253, 0.25);
        background: radial-gradient(circle at top left, rgba(79,70,229,0.10), rgba(15,23,42,1));
        color: #e5e7eb;
    }

    .section-title {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.25rem;
        color: #e5e7eb;
    }

    .sub-label {
        font-size: 0.82rem;
        color: #9ca3af;
        margin-bottom: 0.4rem;
    }

    .stButton>button {
        width: 100%;
        border-radius: 999px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: 1px solid rgba(129, 140, 248, 0.6);
        background: linear-gradient(90deg, #4f46e5, #6366f1);
        color: #f9fafb;
    }

    .stButton>button:hover {
        opacity: 0.94;
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(15,23,42,0.55);
    }

    .prob-title {
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 0.5rem;
    }

    .footer-note {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================
# KONFIGURASI MODEL
# ============================

ARCH = "resnet50"           # atau "resnext50"
CKPT = "best_weights.pt"    # path checkpoint model kamu

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ['DiabeticFootUlcer', 'Normal(Healthy skin)', 'Pressure Ulcer']
NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ============================
# FUNGSI MODEL & CKPT
# ============================

def build_model_from_arch(arch: str, num_classes: int):
    if arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        target_layer = m.layer4[-1].conv3
    elif arch == "resnext50":
        m = models.resnext50_32x4d(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        target_layer = m.layer4[-1].conv3
    else:
        raise ValueError("arch tidak dikenal (gunakan 'resnet50' / 'resnext50')")
    return m.to(DEVICE).eval(), target_layer

def _strip_module_prefix(state_dict):
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def load_weights(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint tidak ditemukan di path: `{ckpt_path}`")
        st.stop()
    state = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = state.get("model", state) if isinstance(state, dict) else state
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    return model

@st.cache_resource
def load_model_and_layer():
    model, target_layer = build_model_from_arch(ARCH, NUM_CLASSES)
    model = load_weights(model, CKPT)
    model.eval()
    return model, target_layer

model, target_layer = load_model_and_layer()

# ============================
# GRAD-CAM
# ============================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h_fwd = self.target_layer.register_forward_hook(self._fwd_hook)
        self.h_bwd = self.target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        acts = self.activations
        grads = self.gradients

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).relu().squeeze(0).cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        return cam, class_idx, probs

    def remove(self):
        self.h_fwd.remove()
        self.h_bwd.remove()

# ============================
# OVERLAY CAM
# ============================

def overlay_cam_on_pil(pil_img, cam, alpha=0.40):
    import matplotlib.cm as cm
    w, h = pil_img.size

    cam_resized = (cam * 255).astype(np.uint8)
    cam_img = Image.fromarray(cam_resized).resize((w, h), Image.BILINEAR)

    heatmap_jet = cm.jet(np.array(cam_img) / 255.0)
    heat_rgb = (heatmap_jet[:, :, :3] * 255).astype(np.uint8)

    base = np.array(pil_img).astype(np.uint8)
    overlay = (alpha * heat_rgb + (1 - alpha) * base).astype(np.uint8)

    return Image.fromarray(overlay)

# ============================
# SIDEBAR
# ============================

with st.sidebar:
    st.markdown("## 🩻 Panel Informasi")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="section-title">Model</div>
            <div class="sub-label">{ARCH.upper()} (Transfer Learning)</div>
            <div class="section-title">Jumlah Kelas</div>
            <div class="sub-label">{NUM_CLASSES} kelas:
                DFU, Normal, Pressure Ulcer
            </div>
            <div class="section-title">Device</div>
            <div class="sub-label">{DEVICE}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📌 Petunjuk Singkat")
    st.markdown(
        """
        1. Upload citra telapak kaki.
        2. Sesuaikan kotak merah untuk memilih **ROI** (area luka / telapak).
        3. Tekan tombol **Klasifikasikan & Tampilkan Grad-CAM**.
        4. Interpretasi:
           - Cek kelas & confidence.
           - Cek apakah hotspot Grad-CAM berada di area lesi yang relevan.
        """
    )

    st.markdown(
        "<div class='footer-note'>Model ini bersifat alat bantu & tidak menggantikan diagnosis klinis.</div>",
        unsafe_allow_html=True
    )

# ============================
# HEADER UTAMA
# ============================

st.markdown(
    """
    <h2 style="margin-bottom: 0.2rem;">
        🩺 DFU / Pressure Ulcer / Normal Classifier with ROI & Grad-CAM
    </h2>
    <p style="color:#9ca3af; font-size:0.9rem; margin-top:0;">
        Demo sistem bantu keputusan berbasis ResNet50 dengan pemilihan Region of Interest (ROI) manual
        dan visualisasi Grad-CAM untuk interpretabilitas.
    </p>
    """,
    unsafe_allow_html=True
)

# ============================
# INPUT GAMBAR + CROP
# ============================

uploaded_file = st.file_uploader(
    "Upload citra telapak kaki (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col_left, col_right = st.columns([1.2, 1.4], vertical_alignment="top")

    with col_left:
        st.markdown("#### Gambar Asli")
        st.image(image, use_container_width=True)

    with col_right:
        st.markdown("#### Pilih ROI (Drag & Resize)")
        st.caption(
            "Geser dan ubah ukuran kotak merah untuk memilih area yang akan dianalisis. "
            "Jika tidak diubah, seluruh gambar digunakan."
        )

        cropped_img = st_cropper(
            image,
            realtime_update=True,
            box_color='#EF4444',
            aspect_ratio=None
        )

        if not isinstance(cropped_img, Image.Image):
            cropped_img = Image.fromarray(cropped_img)

        st.markdown("##### Preview ROI")
        st.image(cropped_img, use_container_width=True)

    proc_image = cropped_img

    # ============================
    # PREDIKSI + GRAD-CAM
    # ============================

    st.markdown("---")
    predict_col1, predict_col2, _ = st.columns([1, 1, 2])

    with predict_col1:
        run_pred = st.button("🚀 Klasifikasikan & Tampilkan Grad-CAM")

    if run_pred:
        with st.spinner("Menghitung prediksi & Grad-CAM..."):
            x = eval_tfms(proc_image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                predicted_idx = int(logits.argmax(1).item())
                confidence = float(probs[predicted_idx])

            cam_engine = GradCAM(model, target_layer)
            cam, cls_idx, _ = cam_engine(x, class_idx=predicted_idx)
            cam_engine.remove()

            predicted_class_name = CLASS_NAMES[predicted_idx]
            overlay_image = overlay_cam_on_pil(proc_image, cam, alpha=0.40)

        # ===== HASIL =====
        col_a, col_b = st.columns([1.2, 1.8])

        with col_a:
            st.markdown("### 🔍 Hasil Prediksi")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="section-title">Kelas Prediksi</div>
                    <div class="sub-label" style="font-size:1.05rem; font-weight:600; color:#a5b4fc;">
                        {predicted_class_name}
                    </div>
                    <div class="section-title">Confidence</div>
                    <div class="sub-label" style="font-size:0.98rem;">
                        {confidence:.4f}
                    </div>
                    <div class="section-title">Interpretasi Threshold</div>
                    <div class="sub-label">
                        {("≥" if confidence >= thr else "<")} threshold {thr:.2f} 
                        &mdash; gunakan bersama konteks klinis dan visual Grad-CAM.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_b:
            st.markdown("### 🧠 Grad-CAM Visualization (ROI)")
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
            ax[0].imshow(proc_image)
            ax[0].set_title("ROI Input")
            ax[0].axis("off")
            ax[1].imshow(overlay_image)
            ax[1].set_title(f"Grad-CAM → {CLASS_NAMES[cls_idx]}")
            ax[1].axis("off")
            st.pyplot(fig)

        # Probabilitas
        st.markdown("### 📊 Distribusi Probabilitas Kelas")
        fig2, ax2 = plt.subplots(figsize=(6.5, 2.8))
        y_pos = np.arange(len(CLASS_NAMES))
        ax2.barh(y_pos, probs)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(CLASS_NAMES)
        ax2.set_xlabel("Probabilitas")
        ax2.set_xlim(0, 1)
        ax2.grid(axis='x', linestyle='--', alpha=0.3)
        st.pyplot(fig2)

        st.markdown(
            "<div class='footer-note'>Gunakan hasil ini sebagai decision support, bukan keputusan klinis tunggal.</div>",
            unsafe_allow_html=True
        )

else:
    st.info("Silakan upload citra terlebih dahulu untuk mulai analisis.")
