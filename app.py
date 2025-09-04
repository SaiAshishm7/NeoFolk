import io
import os
from typing import Optional

import streamlit as st
from PIL import Image

from src.inference import StyleTranslator, list_available_styles

st.set_page_config(page_title="classical art transfer", page_icon="🎨", layout="centered")

# ---------- THEME & GLOBAL UI ----------
st.markdown(
	"""
	<style>
		/* Background */
		.stApp {
			background: radial-gradient(1200px 800px at 10% 0%, rgba(240,245,255,0.9), rgba(245,245,245,0.95)),
				linear-gradient(135deg, #f2efe9 0%, #f7f3eb 50%, #eef4ff 100%);
			background-attachment: fixed;
		}
		/* Typography */
		h1, h2, h3, h4 { font-family: 'Georgia', serif; letter-spacing: .2px; }
		p, label, span, div { font-family: 'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
		/* Accent elements */
		.section-card {
			background: rgba(255,255,255,0.7);
			backdrop-filter: blur(8px);
			border: 1px solid rgba(0,0,0,0.05);
			border-radius: 16px;
			padding: 18px 18px 10px 18px;
			box-shadow: 0 8px 24px rgba(14, 30, 68, 0.08);
		}
		.hero {
			background: linear-gradient(120deg, rgba(78,93,120,.10), rgba(149,173,200,.10));
			border-radius: 18px;
			border: 1px solid rgba(0,0,0,0.06);
			padding: 18px 20px;
			margin-bottom: 10px;
		}
		.small-note { color: #5b6270; font-size: 13px; }
		/* Buttons */
		button[kind="secondary"] { border-radius: 10px !important; }
		.stButton>button {
			background: linear-gradient(135deg, #3c7bd6 0%, #3953b3 100%);
			color: #fff; border: none; border-radius: 12px; padding: 10px 16px;
			box-shadow: 0 6px 20px rgba(57,83,179,0.25);
		}
		.stButton>button:hover { filter: brightness(1.05); }
		/* File uploader */
		[data-testid="stFileUploadDropzone"] {
			border-radius: 14px; border: 1px dashed rgba(0,0,0,0.2);
			background: rgba(255,255,255,0.65);
		}
		/* Footer */
		.footer { text-align:center; color:#6a7485; font-size: 12px; padding: 10px 0 0 0; }
	</style>
	""",
	unsafe_allow_html=True,
)

st.markdown(
	"""
	<div class="hero">
		<h1 style="margin: 0 0 6px 0;">Classical Art Transfer</h1>
		<p style="margin: 0;">Transform your photos into painterly renderings inspired by classical art movements. Upload a photo, choose a style, and fine‑tune with artistic controls.</p>
	</div>
	""",
	unsafe_allow_html=True,
)

weights_root = os.path.join(os.path.dirname(__file__), "weights")

styles = list_available_styles(weights_root)
if not styles:
	st.warning("No styles found. Place weights under `weights/<style>/G_A2B.pth`.")

# ---------- LAYOUT ----------
left, right = st.columns([3, 2], gap="large")
with right:
	st.markdown("<div class='section-card'>", unsafe_allow_html=True)
	style = st.selectbox("Art style", options=styles or ["monet", "cezanne", "ukiyoe", "vangogh"]) 
	style_lower = style.lower()
	is_monet = style_lower in ["monet", "style_monet", "monet2photo"]
	device = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
	size = st.slider("Max size (px)", min_value=256, max_value=1024, value=512, step=64, help="Inference size; larger = more detail but slower.")
	intensity = st.slider("Style intensity", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
	if is_monet:
		edge = st.slider("Edge enhance", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
		brush = st.slider("Brush strokes", min_value=0.0, max_value=1.5, value=0.6, step=0.1)
		monet_cool = st.slider("Monet cool grade", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
		detail = st.slider("Preserve details", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
		palette_ref_upload = st.file_uploader("Palette reference (optional)", type=["jpg", "jpeg", "png"], key="palette")
	else:
		edge = 0.0; brush = 0.0; monet_cool = 0.0; detail = 0.0; palette_ref_upload = None
	st.markdown("<div class='small-note'>Tip: For Monet try cool grade 0.6, details 0.5, strokes 0.6.</div>", unsafe_allow_html=True)
	run_btn = st.button("Transform", use_container_width=True)
	st.markdown("</div>", unsafe_allow_html=True)

with left:
	st.markdown("<div class='section-card'>", unsafe_allow_html=True)
	upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
	if upload:
		image = Image.open(upload).convert("RGB")
		st.image(image, caption="Input", use_container_width=True)
	else:
		image = None
	st.markdown("</div>", unsafe_allow_html=True)

translator: Optional[StyleTranslator] = None

@st.cache_resource(show_spinner=False)
def get_translator(weights_root: str, style: str, device: str) -> StyleTranslator:
	return StyleTranslator(weights_root=weights_root, style=style, device_preference=device)

# ---------- ACTION ----------
if run_btn:
	if image is None:
		st.error("Please upload an image first.")
		st.stop()
	palette_img = None
	if palette_ref_upload is not None:
		palette_img = Image.open(palette_ref_upload).convert("RGB")
	with st.spinner("Painting your image…"):
		translator = get_translator(weights_root, style, device)
		out_img = translator.translate_pil(
			image,
			max_size=size,
			intensity=intensity,
			palette_ref=palette_img,
			edge_enhance=edge,
			brush_strokes=brush,
			monet_cool=monet_cool,
			detail_preserve=detail,
		)
		st.markdown("<div class='section-card'>", unsafe_allow_html=True)
		st.image(out_img, caption=f"Stylized: {style}", use_container_width=True)
		buf = io.BytesIO(); out_img.save(buf, format="PNG")
		st.download_button("Download PNG", data=buf.getvalue(), file_name=f"{style}_stylized.png", mime="image/png", use_container_width=True)
		st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">🎨 Classical Art Transfer • CycleGAN inference with tasteful post‑processing · For education and research</div>
""", unsafe_allow_html=True) 