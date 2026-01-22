import torch
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as T
from comfy.utils import ProgressBar

def tensor_to_pil(img):
    return T.ToPILImage()(img.permute(2, 0, 1)).convert('RGB')

def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR)
    For DLib (6 points): 0-3 horizontal, 1-5 and 2-4 vertical
    For InsightFace (10 points): we'll take average vertical / horizontal
    """
    if len(eye_landmarks) == 6:
        # DLib style
        p = eye_landmarks
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * h)
    elif len(eye_landmarks) == 10:
        # InsightFace style (usually perimeter)
        # Assuming order is somewhat standard, we'll use bounding box ratio as fallback or pick points
        # For simplicity, let's use the bounding box of the eye points
        min_p = np.min(eye_landmarks, axis=0)
        max_p = np.max(eye_landmarks, axis=0)
        width = max_p[0] - min_p[0]
        height = max_p[1] - min_p[1]
        return height / width if width > 0 else 0
    return 0

class I2VChainHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "analysis_models": ("ANALYSIS_MODELS",),
                "min_face_similarity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_eyes_openness": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("trimmed_images", "frame_count", "first_frame", "last_frame")
    FUNCTION = "execute"
    CATEGORY = "I2VChain"

    def execute(self, images, analysis_models, min_face_similarity, min_eyes_openness):
        if images.shape[0] == 0:
            return (images, 0, images, images)

        # 1. Get reference embedding from the first frame
        ref_img = tensor_to_pil(images[0])
        ref_embed = analysis_models.get_embeds(np.array(ref_img))
        
        if ref_embed is None:
            print("I2VChainHelper: No face detected in the first frame. Returning empty batch.")
            return (images[:0], 0, images[:0], images[:0])

        ref_embed = ref_embed / np.linalg.norm(ref_embed)

        last_good_index = 0
        pbar = ProgressBar(images.shape[0])
        
        # Scan from last to first to save time as requested
        for i in range(images.shape[0] - 1, -1, -1):
            img_pil = tensor_to_pil(images[i])
            img_np = np.array(img_pil)
            
            # Check similarity
            curr_embed = analysis_models.get_embeds(img_np)
            if curr_embed is None:
                pbar.update(1)
                continue
            
            curr_embed = curr_embed / np.linalg.norm(curr_embed)
            similarity = np.dot(ref_embed, curr_embed)
            
            if similarity < min_face_similarity:
                pbar.update(1)
                continue
            
            # Check eyes
            landmarks = analysis_models.get_landmarks(img_np)
            if landmarks is None:
                pbar.update(1)
                continue
                
            left_eye = landmarks[3]
            right_eye = landmarks[4]
            
            ear_l = calculate_ear(left_eye)
            ear_r = calculate_ear(right_eye)
            avg_ear = (ear_l + ear_r) / 2.0
            
            if avg_ear < min_eyes_openness:
                pbar.update(1)
                continue
                
            # Found the last good frame!
            last_good_index = i
            pbar.update(images.shape[0]) # Mark as complete for the UI
            break
            
        trimmed_images = images[:last_good_index + 1]
        first_frame = trimmed_images[0:1]
        last_frame = trimmed_images[-1:]
        
        return (trimmed_images, trimmed_images.shape[0], first_frame, last_frame)

NODE_CLASS_MAPPINGS = {
    "I2VChainHelper": I2VChainHelper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "I2VChainHelper": "I2V Chain Helper"
}
