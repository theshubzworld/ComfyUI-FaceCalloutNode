import torch
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageColor
import cv2 # OpenCV for face detection
import os
import math

# --- Helper Functions (tensor_to_pil, pil_to_tensor - same as before) ---
def tensor_to_pil(tensor_image):
    if tensor_image.ndim == 4:
        tensor_image = tensor_image[0]
    image_np = tensor_image.cpu().numpy()
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    # Ensure RGB if it's RGBA from a previous node that might output alpha
    if image_np.shape[-1] == 4:
        pil_img = Image.fromarray(image_np, 'RGBA').convert('RGB')
    else:
        pil_img = Image.fromarray(image_np, 'RGB')
    return pil_img


def pil_to_tensor(pil_image_with_alpha): # Expecting RGBA PIL image
    # Ensure it's RGBA
    if pil_image_with_alpha.mode != 'RGBA':
        pil_image_with_alpha = pil_image_with_alpha.convert('RGBA')
        
    image_np = np.array(pil_image_with_alpha).astype(np.float32) / 255.0 # HWC, 0-1 range
    tensor_image = torch.from_numpy(image_np).unsqueeze(0) # BHWC
    return tensor_image
# --- End Helper Functions ---

class IsolatedFaceCalloutNode: # Renamed class for clarity
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",), # Input image to find the face in
                "face_padding_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "label": "1. Face Padding (for crop)"}),
                "callout_scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1, "label": "2. Callout Image Scale"}),
                "callout_border_thickness": ("INT", {"default": 5, "min": 0, "max": 30, "step": 1, "label": "3. Callout Border Size"}),
                "callout_border_color": ("STRING", {"default": "#FFFFFF", "label": "4. Callout Border Color (Hex)"}),
                "fallback_behavior": (["Blank Transparent Image", "Error"], {"default": "Blank Transparent Image"}),
                "multiple_face_behavior": (["First Detected", "Largest Face"], {"default": "Largest Face"}),
            }
        }

    RETURN_TYPES = ("IMAGE",) # Outputting an IMAGE (which can have alpha)
    RETURN_NAMES = ("callout_image",)
    FUNCTION = "create_isolated_callout"
    CATEGORY = "image/Annotation" # Or image/Masking or image/Transform

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.haar_cascade_path = os.path.join(current_dir, "haarcascade_frontalface_default.xml")
        self.face_cascade = None
        
        print(f"DEBUG: IsolatedFaceCalloutNode: __file__ is {__file__}")
        print(f"DEBUG: IsolatedFaceCalloutNode: current_dir is {current_dir}")
        print(f"DEBUG: IsolatedFaceCalloutNode: Attempting to load cascade from: {self.haar_cascade_path}")
        
        if os.path.exists(self.haar_cascade_path):
            print(f"DEBUG: IsolatedFaceCalloutNode: Cascade file FOUND at {self.haar_cascade_path}")
            self.face_cascade = cv2.CascadeClassifier(self.haar_cascade_path)
            if self.face_cascade.empty():
                print(f"ERROR: IsolatedFaceCalloutNode: Failed to load Haar Cascade (it was found but cv2.CascadeClassifier returned empty) from: {self.haar_cascade_path}")
                self.face_cascade = None
            else:
                print(f"SUCCESS: IsolatedFaceCalloutNode: Successfully loaded Haar Cascade from {self.haar_cascade_path}")
        else:
            print(f"ERROR: IsolatedFaceCalloutNode: Haar Cascade file NOT FOUND at: {self.haar_cascade_path}")

    def create_isolated_callout(self, source_image: torch.Tensor, face_padding_percent: float,
                                callout_scale_factor: float, callout_border_thickness: int,
                                callout_border_color: str, fallback_behavior: str,
                                multiple_face_behavior: str):

        # Standard size for blank fallback, can be made configurable
        blank_fallback_dim = 256

        if not self.face_cascade:
            print("IsolatedFaceCalloutNode: Face cascade not loaded.")
            if fallback_behavior == "Error":
                raise RuntimeError("Face cascade classifier not loaded.")
            # Return a blank transparent image
            blank_pil = Image.new("RGBA", (blank_fallback_dim, blank_fallback_dim), (0,0,0,0))
            return (pil_to_tensor(blank_pil),)

        pil_source_image = tensor_to_pil(source_image) # Converts to RGB
        img_width, img_height = pil_source_image.size

        cv_image = np.array(pil_source_image.convert('L')) # Grayscale for detection
        faces = self.face_cascade.detectMultiScale(cv_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("IsolatedFaceCalloutNode: No faces detected.")
            if fallback_behavior == "Error":
                raise ValueError("No faces detected in the source image.")
            blank_pil = Image.new("RGBA", (blank_fallback_dim, blank_fallback_dim), (0,0,0,0))
            return (pil_to_tensor(blank_pil),)

        # Select face
        if multiple_face_behavior == "Largest Face" and len(faces) > 1:
            face_idx = np.argmax([fw * fh for _, _, fw, fh in faces])
            fx, fy, fw, fh = faces[face_idx]
        else:
            fx, fy, fw, fh = faces[0]

        # 1. Crop the face area from the source image with padding
        base_crop_size = max(fw, fh) # Use the larger dimension of the detected face
        padded_dim_for_crop = int(base_crop_size * (1 + face_padding_percent))
        
        # Center the crop on the detected face
        crop_center_x = fx + fw // 2
        crop_center_y = fy + fh // 2
        
        crop_x1 = crop_center_x - padded_dim_for_crop // 2
        crop_y1 = crop_center_y - padded_dim_for_crop // 2
        crop_x2 = crop_x1 + padded_dim_for_crop
        crop_y2 = crop_y1 + padded_dim_for_crop

        # Perform the crop from the original PIL source image (which is RGB)
        face_crop_pil = pil_source_image.crop((
            max(0, crop_x1), max(0, crop_y1),
            min(img_width, crop_x2), min(img_height, crop_y2)
        ))
        # Resize to ensure it's square if it was clipped at image edges
        face_crop_pil = face_crop_pil.resize((padded_dim_for_crop, padded_dim_for_crop), Image.Resampling.LANCZOS)


        # 2. Scale the cropped face for the final callout size
        # The callout_scale_factor is applied to the *original detected face size* (fw, fh)
        # not the padded crop, to make scaling more intuitive.
        callout_content_dim = int(max(fw, fh) * callout_scale_factor)
        if callout_content_dim <= 0: callout_content_dim = 50 # Minimum size

        # The final output image size will be this content dimension + border on all sides
        final_callout_image_dim = callout_content_dim + (2 * callout_border_thickness)
        if final_callout_image_dim <=0: final_callout_image_dim = 50 # ensure positive

        # Resize the (padded) face_crop_pil to fit into the content area of the callout
        scaled_face_for_callout_pil = face_crop_pil.resize((callout_content_dim, callout_content_dim), Image.Resampling.LANCZOS)

        # 3. Create the final circular callout image (RGBA)
        output_callout_pil = Image.new("RGBA", (final_callout_image_dim, final_callout_image_dim), (0,0,0,0)) # Transparent background
        draw_on_callout = ImageDraw.Draw(output_callout_pil)

        # Draw the circular border (if thickness > 0)
        if callout_border_thickness > 0:
            try:
                parsed_border_color_rgb = ImageColor.getrgb(callout_border_color)
                # Add full alpha for the border color
                parsed_border_color_rgba = parsed_border_color_rgb + (255,)
            except ValueError:
                parsed_border_color_rgba = (255,255,255,255) # White fallback
            
            draw_on_callout.ellipse(
                (0, 0, final_callout_image_dim -1 , final_callout_image_dim -1), # Outer ellipse for border
                fill=parsed_border_color_rgba
            )

        # Create a circular mask for the face content (inside the border)
        # The actual face image will be pasted into an area offset by border_thickness
        mask_dim = callout_content_dim 
        content_mask = Image.new('L', (mask_dim, mask_dim), 0)
        ImageDraw.Draw(content_mask).ellipse((0,0, mask_dim-1, mask_dim-1), fill=255)

        # Paste the scaled face content onto the callout image using the mask
        # The paste position is offset by the border thickness
        output_callout_pil.paste(
            scaled_face_for_callout_pil, # This is the RGB face image
            (callout_border_thickness, callout_border_thickness),
            content_mask # This mask makes the pasted part circular
        )
        
        return (pil_to_tensor(output_callout_pil),) # pil_to_tensor handles RGBA