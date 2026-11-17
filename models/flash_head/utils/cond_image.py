"""Build conditioning image dict for FlashHead (face crop optional)."""

from __future__ import annotations

import glob
import os

from PIL import Image

from flash_head.utils.facecrop import process_image


def get_cond_image_dict(cond_image_path_or_dir: str, use_face_crop: bool) -> dict:
    def get_image(cond_image_path: str, use_face_crop: bool) -> Image.Image:
        if use_face_crop:
            return process_image(cond_image_path)
        return Image.open(cond_image_path).convert("RGB")

    if os.path.isdir(cond_image_path_or_dir):
        cond_image_list = glob.glob(os.path.join(cond_image_path_or_dir, "*.png"))
        cond_image_list.sort()
        return {
            cond_image.split("/")[-1].split(".")[0]: get_image(cond_image, use_face_crop)
            for cond_image in cond_image_list
        }
    key = cond_image_path_or_dir.split("/")[-1].split(".")[0]
    return {key: get_image(cond_image_path_or_dir, use_face_crop)}