# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import os

# --- Konfigurasi Utama ---
# Ganti path ini sesuai dengan lokasi model Anda
  # Folder root eksperimen
ARCH = "resnet50"           # Ganti ke "resnext50" jika model Anda adalah ResNeXt
CKPT = "best_weights.pt"

# Ganti dengan ukuran gambar yang digunakan saat pelatihan
IMG_SIZE = 224
# Mean dan Std ImageNet, biasanya tidak berubah jika tidak disesuaikan
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# INI PENTING: Ganti dengan NAMA KELAS yang sesuai dengan model Anda, dalam urutan indeks
# Misalnya, jika model Anda memprediksi ["Cat", "Dog", "Bird"], maka:
# CLASS_NAMES = ["Cat", "Dog", "Bird"]
CLASS_NAMES = ['DiabeticFootUlcer', 'Normal(Healthy skin)', 'Pressure Ulcer']

NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Streamlit app akan menggunakan device: {DEVICE}")

# --- Transformasi ---
eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# --- Fungsi Load Model ---
def build_model_from_arch(arch: str, num_classes: int):
    if arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        target_layer = m.layer4[-1].conv3  # Atau m.layer4[-1] jika conv3 tidak ada
    elif arch == "resnext50":
        m = models.resnext50_32x4d(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        target_layer = m.layer4[-1].conv3
    else:
        raise ValueError("arch tidak dikenal (pakai 'resnet50' / 'resnext50')")
    return m.to(DEVICE).eval(), target_layer

def _strip_module_prefix(state_dict):
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def load_weights(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint tidak ditemukan di path: {ckpt_path}")
        st.stop()
    state = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = state.get("model", state) if isinstance(state, dict) else state
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    return model

# --- Inisialisasi Model dan GradCAM ---
@st.cache_resource  # Gunakan cache untuk memuat model sekali saja
def load_model_and_layer():
    model, target_layer = build_model_from_arch(ARCH, NUM_CLASSES)
    model = load_weights(model, CKPT)
    model.eval()
    return model, target_layer

model, target_layer = load_model_and_layer()
st.success(f"Model {ARCH} berhasil dimuat dari {CKPT}")

# --- Kelas GradCAM ---
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
        acts  = self.activations
        grads = self.gradients
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1).relu().squeeze(0).cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0: cam /= cam.max()
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        return cam, class_idx, probs

    def remove(self):
        self.h_fwd.remove()
        self.h_bwd.remove()

# --- Fungsi Overlay ---
def overlay_cam_on_pil(pil_img, cam, alpha=0.35):
    import matplotlib.cm as cm
    w, h = pil_img.size
    cam_resized = (cam * 255).astype(np.uint8)
    cam_img = Image.fromarray(cam_resized).resize((w, h), Image.BILINEAR)
    # Konversi heatmap ke RGB
    heatmap_jet = cm.jet(np.array(cam_img)/255.0)
    # Ambil hanya 3 channel RGB, buang alpha
    heat_rgb = (heatmap_jet[:, :, :3] * 255).astype(np.uint8)
    base = np.array(pil_img).astype(np.uint8)
    overlay = (alpha * heat_rgb + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(overlay)


# --- UI Streamlit ---
st.title("Demo: Klasifikasi Gambar & Grad-CAM")
st.write(f"Model yang digunakan: `{ARCH}` | Kelas: `{CLASS_NAMES}`")

uploaded_file = st.file_uploader("Pilih gambar untuk diprediksi...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Gambar yang Diunggah', use_container_width=True)

    if st.button('Klasifikasikan & Tampilkan Grad-CAM'):
        with st.spinner('Memproses...'):
            # Prediksi
            x = eval_tfms(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                predicted_idx = int(logits.argmax(1).item())
                confidence = float(probs[predicted_idx])

            # GradCAM
            cam_engine = GradCAM(model, target_layer)
            cam, cls_idx, _ = cam_engine(x, class_idx=predicted_idx) # Gunakan prediksi
            cam_engine.remove()

            # Hasil
            predicted_class_name = CLASS_NAMES[predicted_idx]
            st.subheader("Hasil Prediksi")
            st.write(f"**Kelas Prediksi:** {predicted_class_name}")
            st.write(f"**Confidence:** {confidence:.4f}")

            # Visualisasi GradCAM
            overlay_image = overlay_cam_on_pil(image, cam, alpha=0.4)

            st.subheader("Visualisasi Grad-CAM")
            st.write(f"Grad-CAM untuk kelas: **{CLASS_NAMES[cls_idx]}**")

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].imshow(image)
            ax[0].set_title("Gambar Input")
            ax[0].axis('off')
            ax[1].imshow(overlay_image)
            ax[1].set_title(f"Grad-CAM -> {CLASS_NAMES[cls_idx]}")
            ax[1].axis('off')
            st.pyplot(fig)

            # Plot probabilitas kelas
            st.subheader("Probabilitas untuk Setiap Kelas")
            fig2, ax2 = plt.subplots()
            ax2.barh(CLASS_NAMES, probs)
            ax2.set_xlabel("Probabilitas")
            ax2.set_title("Probabilitas Kelas")
            st.pyplot(fig2)

else:
    st.info("Silakan unggah file gambar (JPG, PNG).")
