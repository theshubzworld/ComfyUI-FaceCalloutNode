import torch
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageColor
import cv2 # OpenCV for face detection
import os
import math

# --- Helper Functions (tensor_to_pil, pil_to_tensor - from previous correct version) ---
def tensor_to_pil(tensor_image, target_mode='RGB'):
    if tensor_image.ndim == 4:
        tensor_image = tensor_image[0]
    image_np = tensor_image.cpu().numpy()
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    if image_np.shape[-1] == 4:
        pil_img = Image.fromarray(image_np, 'RGBA')
    elif image_np.shape[-1] == 3:
        pil_img = Image.fromarray(image_np, 'RGB')
    else:
        pil_img = Image.fromarray(image_np).convert('RGB')

    if pil_img.mode != target_mode:
        pil_img = pil_img.convert(target_mode)
    return pil_img

def pil_to_tensor(pil_image):
    if pil_image.mode not in ['RGB', 'RGBA']:
        pil_image = pil_image.convert('RGBA')
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    if image_np.ndim == 2:
        image_np = np.stack((image_np,)*3, axis=-1)
    if image_np.shape[-1] == 3 :
        alpha_channel = np.ones_like(image_np[..., :1])
        image_np = np.concatenate((image_np, alpha_channel), axis=-1)
    tensor_image = torch.from_numpy(image_np).unsqueeze(0)
    return tensor_image
# --- End Helper Functions ---

class IntegratedFaceCompositeNode: # Renamed for clarity
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "source_image_for_face": ("IMAGE",), # Image to detect face from
                # Face Detection & Cropping Params (from IsolatedFaceCalloutNode)
                "face_padding_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "label": "A1. Face Padding (for crop)"}),
                "callout_scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1, "label": "A2. Callout Face Scale"}),
                "callout_border_thickness": ("INT", {"default": 5, "min": 0, "max": 30, "step": 1, "label": "A3. Callout Border Size"}),
                "callout_border_color": ("STRING", {"default": "#FFFFFF", "label": "A4. Callout Border Color (Hex)"}),
                "multiple_face_behavior": (["First Detected", "Largest Face"], {"default": "Largest Face", "label": "A5. Face Selection"}),
                # Compositing Params (from CompositeCalloutNode)
                "position_x_percent": ("FLOAT", {"default": 0.80, "min": -0.5, "max": 1.5, "step": 0.01, "label": "B1. X Position on BG (0-1)"}),
                "position_y_percent": ("FLOAT", {"default": 0.15, "min": -0.5, "max": 1.5, "step": 0.01, "label": "B2. Y Position on BG (0-1)"}),
                "composite_scale_factor": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 5.0, "step": 0.05, "label": "B3. Final Composite Scale"}), # Scales the generated callout before pasting
                "pivot_x_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "label": "B4. Pivot X on Callout (0-1)"}),
                "pivot_y_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "label": "B5. Pivot Y on Callout (0-1)"}),
                "fallback_no_face": (["Use Blank Callout", "Return Background", "Error"], {"default": "Use Blank Callout", "label": "A6. If No Face Detected"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited_image",)
    FUNCTION = "generate_and_composite_face"
    CATEGORY = "image/Layering"

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.haar_cascade_path = os.path.join(current_dir, "haarcascade_frontalface_default.xml")
        self.face_cascade = None
        if os.path.exists(self.haar_cascade_path):
            self.face_cascade = cv2.CascadeClassifier(self.haar_cascade_path)
            if self.face_cascade.empty():
                print(f"ERROR: IntegratedFaceCompositeNode: Failed to load Haar Cascade from: {self.haar_cascade_path}")
                self.face_cascade = None
        else:
            print(f"ERROR: IntegratedFaceCompositeNode: Haar Cascade file not found: {self.haar_cascade_path}")

    def _create_isolated_face_callout_pil(self, pil_source_image, face_padding_percent,
                                     callout_scale_factor, callout_border_thickness,
                                     callout_border_color_hex, multiple_face_behavior,
                                     fallback_no_face_action):
        """
        Internal helper to generate the RGBA PIL image of the circular face callout.
        This is essentially the core logic of the previous IsolatedFaceCalloutNode.
        """
        blank_fallback_dim = 100 # Size of blank callout if no face

        if not self.face_cascade:
            print("IntegratedFaceCompositeNode: Face cascade not loaded for callout generation.")
            if fallback_no_face_action == "Error":
                raise RuntimeError("Face cascade classifier not loaded.")
            return Image.new("RGBA", (blank_fallback_dim, blank_fallback_dim), (0,0,0,0))


        img_width, img_height = pil_source_image.size
        cv_image = np.array(pil_source_image.convert('L'))
        faces = self.face_cascade.detectMultiScale(cv_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("IntegratedFaceCompositeNode: No faces detected in source_image_for_face.")
            if fallback_no_face_action == "Error":
                raise ValueError("No faces detected in the source image for face.")
            if fallback_no_face_action == "Return Background": # This option is handled by caller
                return None 
            return Image.new("RGBA", (blank_fallback_dim, blank_fallback_dim), (0,0,0,0)) # Blank Callout

        if multiple_face_behavior == "Largest Face" and len(faces) > 1:
            fx, fy, fw, fh = faces[np.argmax([fwi * fhi for _, _, fwi, fhi in faces])]
        else:
            fx, fy, fw, fh = faces[0]

        base_crop_size = max(fw, fh)
        padded_dim_for_crop = int(base_crop_size * (1 + face_padding_percent))
        crop_center_x, crop_center_y = fx + fw // 2, fy + fh // 2
        crop_x1 = crop_center_x - padded_dim_for_crop // 2
        crop_y1 = crop_center_y - padded_dim_for_crop // 2
        
        face_crop_pil = pil_source_image.crop((
            max(0, crop_x1), max(0, crop_y1),
            min(img_width, crop_x1 + padded_dim_for_crop), min(img_height, crop_y1 + padded_dim_for_crop)
        ))
        face_crop_pil = face_crop_pil.resize((padded_dim_for_crop, padded_dim_for_crop), Image.Resampling.LANCZOS)

        callout_content_dim = int(max(fw, fh) * callout_scale_factor)
        if callout_content_dim <= 0: callout_content_dim = 50
        final_callout_image_dim = callout_content_dim + (2 * callout_border_thickness)
        if final_callout_image_dim <=0: final_callout_image_dim = 50
        
        scaled_face_for_callout_pil = face_crop_pil.resize((callout_content_dim, callout_content_dim), Image.Resampling.LANCZOS)
        
        output_callout_pil = Image.new("RGBA", (final_callout_image_dim, final_callout_image_dim), (0,0,0,0))
        draw_on_callout = ImageDraw.Draw(output_callout_pil)

        if callout_border_thickness > 0:
            try:
                parsed_border_color_rgb = ImageColor.getrgb(callout_border_color_hex)
                parsed_border_color_rgba = parsed_border_color_rgb + (255,)
            except ValueError:
                parsed_border_color_rgba = (255,255,255,255)
            draw_on_callout.ellipse((0, 0, final_callout_image_dim -1 , final_callout_image_dim -1), fill=parsed_border_color_rgba)

        mask_dim = callout_content_dim
        content_mask = Image.new('L', (mask_dim, mask_dim), 0)
        ImageDraw.Draw(content_mask).ellipse((0,0, mask_dim-1, mask_dim-1), fill=255)
        output_callout_pil.paste(scaled_face_for_callout_pil, (callout_border_thickness, callout_border_thickness), content_mask)
        
        return output_callout_pil


    def generate_and_composite_face(self, background_image: torch.Tensor, source_image_for_face: torch.Tensor,
                                    face_padding_percent: float, callout_scale_factor: float,
                                    callout_border_thickness: int, callout_border_color: str,
                                    multiple_face_behavior: str,
                                    position_x_percent: float, position_y_percent: float,
                                    composite_scale_factor: float,
                                    pivot_x_percent: float, pivot_y_percent: float,
                                    fallback_no_face: str):

        pil_source_for_face = tensor_to_pil(source_image_for_face, target_mode='RGB')
        
        # Generate the isolated face callout PIL image
        isolated_callout_pil = self._create_isolated_face_callout_pil(
            pil_source_for_face, face_padding_percent, callout_scale_factor,
            callout_border_thickness, callout_border_color,
            multiple_face_behavior, fallback_no_face
        )

        bg_pil_rgba = tensor_to_pil(background_image, target_mode='RGBA') # Ensure background is RGBA

        if isolated_callout_pil is None: # This happens if fallback_no_face was "Return Background" AND no face was found
            print("IntegratedFaceCompositeNode: No face found and fallback is 'Return Background'. Returning original background.")
            return (pil_to_tensor(bg_pil_rgba),) # Convert back to tensor

        # --- Now, the compositing part (similar to previous CompositeCalloutNode) ---
        callout_width_orig, callout_height_orig = isolated_callout_pil.size

        if callout_width_orig == 0 or callout_height_orig == 0:
            print("Warning: IntegratedFaceCompositeNode: Generated callout has zero dimensions. Compositing with background only.")
            return (pil_to_tensor(bg_pil_rgba),)

        # Scale the generated callout based on composite_scale_factor
        final_scaled_callout_width = int(callout_width_orig * composite_scale_factor)
        final_scaled_callout_height = int(callout_height_orig * composite_scale_factor)

        if final_scaled_callout_width <= 0 or final_scaled_callout_height <= 0:
            print("Warning: IntegratedFaceCompositeNode: Final scaled callout has zero/negative dimensions. Returning background.")
            return (pil_to_tensor(bg_pil_rgba),)
            
        callout_pil_final_scaled = isolated_callout_pil.resize(
            (final_scaled_callout_width, final_scaled_callout_height),
            Image.Resampling.LANCZOS
        )

        # Determine paste position
        bg_width, bg_height = bg_pil_rgba.size
        target_pivot_x_on_bg = int(bg_width * position_x_percent)
        target_pivot_y_on_bg = int(bg_height * position_y_percent)
        pivot_offset_x = int(final_scaled_callout_width * pivot_x_percent)
        pivot_offset_y = int(final_scaled_callout_height * pivot_y_percent)
        paste_x = target_pivot_x_on_bg - pivot_offset_x
        paste_y = target_pivot_y_on_bg - pivot_offset_y

        # Composite
        temp_layer_for_callout = Image.new('RGBA', bg_pil_rgba.size, (0,0,0,0))
        temp_layer_for_callout.paste(callout_pil_final_scaled, (paste_x, paste_y), callout_pil_final_scaled)
        composited_pil = Image.alpha_composite(bg_pil_rgba, temp_layer_for_callout)
        
        return (pil_to_tensor(composited_pil),)