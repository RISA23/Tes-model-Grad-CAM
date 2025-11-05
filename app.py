import io
import numpy as np
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import timm

# ==============================
# Helpers
# ==============================

IMSIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_preprocess(img_size=IMSIZE):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Pick a reasonable last conv layer for Grad-CAM depending on architecture.
    Falls back to the last Conv2d found if specific mapping fails.
    """
    try:
        low = arch.lower()
        if "resnet" in low or "resnext" in low:
            # timm's resnet/resnext keep layer4 blocks compatible
            return model.layer4[-1].conv3
        if "efficientnet" in low:
            # timm efficientnet has conv_head at the end
            return model.conv_head
    except Exception:
        pass

    # Fallback: last conv found
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("Tidak menemukan layer konvolusi untuk Grad-CAM.")
    return last

@st.cache_resource(show_spinner=False)
def load_model(arch: str, num_classes: int, state_dict_bytes: bytes | None):
    """Create a timm model and (optionally) load user-provided weights.
    Returns (model, target_layer) on CPU, eval mode.
    """
    # Create model
    try:
        model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    except Exception as e:
        st.error(f"Gagal membuat model untuk arsitektur '{arch}': {e}")
        raise

    # Load weights if provided
    if state_dict_bytes:
        try:
            buffer = io.BytesIO(state_dict_bytes)
            state_dict = torch.load(buffer, map_location="cpu")
            # Some checkpoints wrap with {"model": state_dict, ...}
            if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                state_dict = state_dict["state_dict"]
            if isinstance(state_dict, dict) and "model_state" in state_dict and isinstance(state_dict["model_state"], dict):
                state_dict = state_dict["model_state"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                st.warning(
                    "Peringatan saat memuat bobot (strict=False):\n"
                    f"- Missing keys: {len(missing)}\n"
                    f"- Unexpected keys: {len(unexpected)}"
                )
            st.success("✅ Bobot kustom berhasil dimuat.")
        except Exception as e:
            st.error(f"Gagal memuat bobot kustom (.pth/.pt): {e}")
            st.stop()

    model.eval()
    target_layer = get_target_layer(model, arch)
    return model, target_layer

def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)

def to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """img_tensor: (C,H,W) in 0..1 (unnormalized)."""
    img = img_tensor.clone().detach()
    img = torch.clamp(img, 0, 1)
    img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img)

def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD).view(3,1,1)
    return img_tensor * std + mean

def gradcam_heatmap(model: nn.Module, target_layer: nn.Module, inp: torch.Tensor, class_idx: int | None):
    """Compute Grad-CAM heatmap for a single input (1,C,H,W)."""
    activations = []
    gradients = []

    def fwd_hook(_, __, output):
        activations.append(output.detach())

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(inp)  # (1, num_classes)
    if class_idx is None:
        class_idx = int(logits.argmax(dim=-1).item())

    score = logits[:, class_idx].sum()
    model.zero_grad(set_to_none=True)
    score.backward()

    # Clean hooks
    h1.remove()
    h2.remove()

    if not activations or not gradients:
        raise RuntimeError("Gagal mengambil aktivasi/gradien untuk Grad-CAM.")

    A = activations[0]            # (1, C, H', W')
    dA = gradients[0]             # (1, C, H', W')
    weights = dA.mean(dim=(2,3), keepdim=True)  # (1, C, 1, 1)
    cam = (weights * A).sum(dim=1, keepdim=True)  # (1, 1, H', W')
    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = F.interpolate(cam, size=inp.shape[2:], mode="bilinear", align_corners=False)  # (1,1,H,W)
    cam = cam.squeeze().cpu().numpy()  # (H,W)
    return logits.detach().cpu(), cam

def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """Overlay a heatmap (0..1) on a PIL image using matplotlib colormap."""
    import matplotlib.cm as cm

    base = pil_img.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    heat = np.uint8(cm.jet(heatmap) * 255)  # RGBA
    heat_rgb = Image.fromarray(heat[:, :, :3]).convert("RGB")
    return Image.blend(base, heat_rgb, alpha=alpha)

def predict_one(image: Image.Image, model, target_layer, preprocess, class_names):
    # Keep a copy for visualization
    img_vis = image.copy().convert("RGB")
    # Preprocess
    x = preprocess(img_vis).unsqueeze(0)  # (1,C,H,W)
    with torch.no_grad():
        logits = model(x)
    probs = softmax_probs(logits)[0].cpu().numpy()

    # Grad-CAM (use top-1 class)
    logits_gc, cam = gradcam_heatmap(model, target_layer, x, None)
    topk = int(min(3, len(class_names)))
    top_idx = probs.argsort()[::-1][:topk]
    top = [(class_names[i], float(probs[i])) for i in top_idx]

    # Build overlay image (denormalize x for background)
    x_denorm = denormalize(x[0].cpu())
    bg = to_pil(x_denorm)
    over = overlay_heatmap(bg, cam, alpha=0.35)
    return top, over

# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="CNN Inference + Grad-CAM", page_icon="🧠", layout="wide")
st.title("🧠 CNN Image Classifier with Grad‑CAM (PyTorch + timm)")

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    arch = st.selectbox(
        "Arsitektur",
        options=["resnet50", "resnext50_32x4d", "efficientnet_b0"],
        index=0,
        help="Gunakan arsitektur yang sama dengan yang dipakai saat training bobotmu."
    )

    st.caption("**Kelas** — pisahkan dengan koma, urut sesuai saat training.")
    classes_text = st.text_area(
        "Daftar kelas",
        value="Class0, Class1, Class2",
        height=80
    )
    class_names = [c.strip() for c in classes_text.split(",") if c.strip()]
    if len(class_names) < 2:
        st.warning("Minimal 2 kelas.")
    num_classes = len(class_names)

    weights_file = st.file_uploader(
        "Upload bobot (.pth / .pt) — opsional",
        type=["pth", "pt"],
        help="Jika dikosongkan: pakai pre-trained ImageNet (head num_classes disesuaikan)."
    )
    st.info("Catatan: Bobot harus cocok dengan arsitektur yang dipilih. App memuat dengan strict=False untuk toleransi perbedaan kecil.")

    img_size = st.number_input("Ukuran input (px)", min_value=128, max_value=640, value=IMSIZE, step=32)
    preprocess = build_preprocess(img_size)

model, target_layer = load_model(
    arch=arch,
    num_classes=num_classes,
    state_dict_bytes=(weights_file.read() if weights_file is not None else None)
)

st.success(f"Model **{arch}** siap • num_classes={num_classes}")

imgs = st.file_uploader("Unggah gambar untuk prediksi", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if not imgs:
    st.stop()

cols = st.columns(2, gap="large")
left, right = cols

with left:
    st.subheader("📥 Gambar Masukan")
    for up in imgs:
        st.image(Image.open(up).convert("RGB"), caption=up.name, use_container_width=True)

with right:
    st.subheader("🔮 Prediksi & Grad‑CAM")
    for up in imgs:
        img = Image.open(up).convert("RGB")
        try:
            top, over = predict_one(img, model, target_layer, preprocess, class_names)
        except Exception as e:
            st.error(f"Gagal memproses {up.name}: {e}")
            continue

        st.markdown(f"**{up.name}**")
        # Show top-k
        for cname, p in top:
            st.write(f"- {cname}: {p:.4f}")
        st.image(over, caption="Grad‑CAM overlay", use_container_width=True)
        st.divider()

st.caption("💡 Tips deployment: `streamlit run app.py` di lokal, atau deploy ke Streamlit Cloud. Pastikan `requirements.txt` sesuai.")