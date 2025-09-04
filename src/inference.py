import os
from typing import Dict, List, Tuple, Optional

import torch
import torchvision.transforms as T
from PIL import Image

from src.models.cyclegan import ResnetGenerator
from src.utils.image_io import center_crop_to_square, histogram_color_transfer, apply_unsharp_mask, apply_brush_strokes, apply_monet_cool_grade, preserve_details


def list_available_styles(weights_root: str) -> List[str]:
	if not os.path.isdir(weights_root):
		return []
	styles = []
	for name in sorted(os.listdir(weights_root)):
		style_dir = os.path.join(weights_root, name)
		if os.path.isdir(style_dir) and os.path.isfile(os.path.join(style_dir, "G_A2B.pth")):
			styles.append(name)
	return styles


class StyleTranslator:
	def __init__(self, weights_root: str, style: str, device_preference: str = "auto") -> None:
		self.weights_root = weights_root
		self.style = style
		self.device = self._resolve_device(device_preference)
		self.model = self._load_generator()
		self.model.eval()

	def _resolve_device(self, preference: str) -> torch.device:
		if preference == "cuda" and torch.cuda.is_available():
			return torch.device("cuda")
		if preference == "cpu":
			return torch.device("cpu")
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def _load_generator(self) -> torch.nn.Module:
		weights_path = os.path.join(self.weights_root, self.style, "G_A2B.pth")
		if not os.path.isfile(weights_path):
			raise FileNotFoundError(f"Weights not found: {weights_path}")
		model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9, norm_layer=torch.nn.InstanceNorm2d)
		state = torch.load(weights_path, map_location="cpu")
		if isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
			state_dict = state
		elif isinstance(state, dict):
			for key, val in state.items():
				if isinstance(val, dict) and all(isinstance(v, torch.Tensor) for v in val.values()):
					state_dict = val
					break
			else:
				raise RuntimeError("Unsupported checkpoint format")
		else:
			raise RuntimeError("Unsupported checkpoint format")
		cleaned = {}
		for k, v in state_dict.items():
			k = k.replace('module.', '')
			if k.endswith('.running_mean') or k.endswith('.running_var') or k.endswith('.num_batches_tracked'):
				continue
			cleaned[k] = v
		missing, unexpected = model.load_state_dict(cleaned, strict=False)
		if missing:
			print(f"[warn] Missing keys: {missing}")
		if unexpected:
			print(f"[warn] Unexpected keys: {unexpected}")
		return model.to(self.device)

	@torch.inference_mode()
	def translate_pil(self, image: Image.Image, max_size: int = 256, intensity: float = 1.0, palette_ref: Optional[Image.Image] = None, edge_enhance: float = 0.0, brush_strokes: float = 0.0, monet_cool: float = 0.0, detail_preserve: float = 0.0) -> Image.Image:
		image = image.convert("RGB")
		w, h = image.size
		short = min(w, h)
		scale = 256 / short
		new_w, new_h = int(round(w * scale)), int(round(h * scale))
		image_resized = image.resize((new_w, new_h), Image.BICUBIC)
		left = (new_w - 256) // 2
		top = (new_h - 256) // 2
		image_resized = image_resized.crop((left, top, left + 256, top + 256))
		transform = T.Compose([
			T.ToTensor(),
			T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])
		tensor = transform(image_resized).unsqueeze(0).to(self.device)
		out = self.model(tensor)
		out = (out.clamp(-1, 1) * 0.5 + 0.5).squeeze(0).cpu()
		to_pil = T.ToPILImage()
		stylized = to_pil(out)
		intensity = float(max(0.0, min(1.0, intensity)))
		if intensity < 1.0:
			stylized = Image.blend(image_resized, stylized, intensity)
		if palette_ref is not None:
			stylized = histogram_color_transfer(stylized, palette_ref)
		if edge_enhance > 0.0:
			stylized = apply_unsharp_mask(stylized, radius=1.5, amount=edge_enhance)
		if brush_strokes > 0.0:
			stylized = apply_brush_strokes(stylized, strength=brush_strokes)
		if monet_cool > 0.0 or self.style.lower() in {"monet", "style_monet", "monet2photo"}:
			amt = monet_cool if monet_cool > 0.0 else 0.5
			stylized = apply_monet_cool_grade(stylized, amount=amt)
		if detail_preserve > 0.0:
			stylized = preserve_details(image_resized, stylized, amount=detail_preserve)
		return stylized 