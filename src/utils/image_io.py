from PIL import Image, ImageFilter
import numpy as np

try:
	from skimage.exposure import match_histograms as _match_hist
	skimage_available = True
except Exception:
	skimage_available = False
	_match_hist = None

try:
	import cv2
	cv_available = True
except Exception:
	cv_available = False
	cv2 = None


def center_crop_to_square(image: Image.Image) -> Image.Image:
	w, h = image.size
	side = min(w, h)
	left = (w - side) // 2
	top = (h - side) // 2
	return image.crop((left, top, left + side, top + side))


def apply_unsharp_mask(image: Image.Image, radius: float = 2.0, amount: float = 1.0) -> Image.Image:
	percent = int(100 * amount)
	return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))


def histogram_color_transfer(source: Image.Image, reference: Image.Image) -> Image.Image:
	if not skimage_available:
		return source
	src = np.asarray(source)
	ref = np.asarray(reference.resize(source.size, Image.LANCZOS))
	matched = _match_hist(src, ref, channel_axis=-1)
	matched = np.clip(matched, 0, 255).astype(np.uint8)
	return Image.fromarray(matched)


def apply_brush_strokes(image: Image.Image, strength: float = 0.6) -> Image.Image:
	if not cv_available or strength <= 0.0:
		return image
	img = np.asarray(image).astype(np.float32) / 255.0
	smoothed = cv2.bilateralFilter((img * 255).astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=9).astype(np.float32) / 255.0
	gray = cv2.cvtColor((smoothed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
	gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
	gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
	angle = (np.arctan2(gy, gx) + np.pi) * (180.0 / np.pi)
	bins = np.array([0, 45, 90, 135], dtype=np.float32)
	def motion_kernel(size: int, theta_deg: float):
		kernel = np.zeros((size, size), dtype=np.float32)
		center = size // 2
		theta = np.deg2rad(theta_deg)
		cos_t, sin_t = np.cos(theta), np.sin(theta)
		for i in range(size):
			x = int(center + (i - center) * cos_t)
			y = int(center + (i - center) * sin_t)
			if 0 <= x < size and 0 <= y < size:
				kernel[y, x] = 1.0
		kernel /= max(kernel.sum(), 1.0)
		return kernel
	kern_size = 7
	kernels = [motion_kernel(kern_size, d) for d in bins]
	variants = []
	for k in kernels:
		var = cv2.filter2D(smoothed, -1, k)
		variants.append(var)
	ang180 = np.mod(angle, 180.0)
	nearest = np.argmin(np.stack([np.abs(ang180 - b) for b in bins], axis=0), axis=0)
	masks = [(nearest == idx).astype(np.float32) for idx in range(4)]
	masks = [cv2.GaussianBlur(m, (0, 0), 1.0) for m in masks]
	mask_sum = np.clip(np.sum(np.stack(masks, axis=0), axis=0, keepdims=True), 1e-6, None)
	norm_masks = [m[..., None] / mask_sum.transpose(1, 2, 0) for m in masks]
	stroke_img = np.zeros_like(smoothed)
	for m, v in zip(norm_masks, variants):
		stroke_img += v * m
	out = (1.0 - strength) * img + strength * stroke_img
	out = np.clip(out, 0.0, 1.0)
	out_img = Image.fromarray((out * 255).astype(np.uint8))
	out_img = apply_unsharp_mask(out_img, radius=1.5, amount=0.6 * strength)
	return out_img


def preserve_details(original: Image.Image, stylized: Image.Image, amount: float = 0.5) -> Image.Image:
	amount = float(max(0.0, min(1.0, amount)))
	if amount <= 0.0:
		return stylized
	if not cv_available:
		return apply_unsharp_mask(stylized, radius=1.0, amount=0.5 * amount)
	orig = np.asarray(original).astype(np.float32) / 255.0
	sty = np.asarray(stylized).astype(np.float32) / 255.0
	orig_gray = cv2.cvtColor((orig * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
	low = cv2.GaussianBlur(orig_gray, (0, 0), 1.5)
	high = np.clip(orig_gray - low, -1.0, 1.0)
	high3 = np.stack([high, high, high], axis=-1)
	res = np.clip(sty + amount * 0.7 * high3, 0.0, 1.0)
	return Image.fromarray((res * 255).astype(np.uint8))


def apply_monet_cool_grade(image: Image.Image, amount: float = 0.6) -> Image.Image:
	amount = float(max(0.0, min(1.0, amount)))
	arr = np.asarray(image).astype(np.float32) / 255.0
	gains = np.array([1.0 - 0.20 * amount, 1.0 + 0.05 * amount, 1.0 + 0.20 * amount], dtype=np.float32)
	arr = arr * gains
	tint = np.array([0.95, 1.03, 1.05], dtype=np.float32)
	arr = arr * tint
	gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
	sat = 1.0 - 0.15 * amount
	arr[..., 0] = gray * (1 - sat) + arr[..., 0] * sat
	arr[..., 1] = gray * (1 - sat) + arr[..., 1] * sat
	arr[..., 2] = gray * (1 - sat) + arr[..., 2] * sat
	gamma = 1.0 + 0.1 * amount
	arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
	arr = np.clip(arr, 0.0, 1.0)
	return Image.fromarray((arr * 255).astype(np.uint8)) 