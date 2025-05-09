import torch
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageColor
import cv2 # OpenCV for face detection
import os
import math

# --- Helper Functions (tensor_to_pil, pil_to_tensor - assumed to be the same as before) ---
def tensor_to_pil(tensor_image):
    if tensor_image.ndim == 4:
        tensor_image = tensor_image[0]
    image_np = tensor_image.cpu().numpy()
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    return Image.fromarray(image_np, 'RGB')

def pil_to_tensor(pil_image):
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    tensor_image = torch.from_numpy(image_np).unsqueeze(0)
    return tensor_image
# --- End Helper Functions ---

class FaceCalloutNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_padding_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "label": "1. Face Padding (for crop)"}),
                "callout_scale_factor": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 3.0, "step": 0.1, "label": "2. Callout Circle Scale"}),
                "callout_pos_x_percent": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01, "label": "3. Callout X Pos (0-1 right)"}),
                "callout_pos_y_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "label": "4. Callout Y Pos (0-1 down)"}),
                "callout_border_thickness": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1, "label": "5. Callout Circle Border Size"}),
                "callout_border_color": ("STRING", {"default": "#FFFFFF", "label": "6. Callout Border Color"}),
                "line_color": ("STRING", {"default": "#333333", "label": "7. Line Color"}),
                "line_thickness": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1, "label": "8. Line Thickness"}),
                "line_start_face_ratio_x": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "label": "9. Line Start X on Face (0-1)"}),
                "line_start_face_ratio_y": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "label": "10. Line Start Y on Face (0-1)"}),
                "curl_control_factor1_x": ("FLOAT", {"default": 0.3, "min": -1.0, "max": 1.0, "step": 0.05, "label": "11. Curl Ctrl1 X (rel to line)"}),
                "curl_control_factor1_y": ("FLOAT", {"default": -0.5, "min": -1.0, "max": 1.0, "step": 0.05, "label": "12. Curl Ctrl1 Y (rel to line)"}),
                "curl_control_factor2_x": ("FLOAT", {"default": 0.7, "min": -1.0, "max": 1.0, "step": 0.05, "label": "13. Curl Ctrl2 X (rel to line)"}),
                "curl_control_factor2_y": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.05, "label": "14. Curl Ctrl2 Y (rel to line)"}),
                "arrow_head_size_factor": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.3, "step": 0.01, "label": "15. Arrowhead Size (rel. to callout)"}),
                "fallback_behavior": (["Return Original", "Error", "Blank Image"], {"default": "Return Original"}),
                "multiple_face_behavior": (["First Detected", "Largest Face"], {"default": "Largest Face"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("modified_image",)
    FUNCTION = "process_callout"
    CATEGORY = "image/Annotation"

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.haar_cascade_path = os.path.join(current_dir, "haarcascade_frontalface_default.xml")
        self.face_cascade = None
        if os.path.exists(self.haar_cascade_path):
            self.face_cascade = cv2.CascadeClassifier(self.haar_cascade_path)
            if self.face_cascade.empty():
                print(f"ERROR: FaceCalloutNode: Failed to load Haar Cascade from: {self.haar_cascade_path}")
                self.face_cascade = None
        else:
            print(f"ERROR: FaceCalloutNode: Haar Cascade file not found at: {self.haar_cascade_path}")


    def _get_bezier_point(self, t, p0, p1, p2, p3):
        """Calculate a point on a cubic Bezier curve."""
        u = 1 - t
        tt = t * t
        uu = u * u
        uuu = uu * u
        ttt = tt * t
        x = uuu * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + ttt * p3[0]
        y = uuu * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + ttt * p3[1]
        return (int(x), int(y))

    def process_callout(self, image: torch.Tensor, face_padding_percent: float,
                        callout_scale_factor: float, callout_pos_x_percent: float, callout_pos_y_percent: float,
                        callout_border_thickness: int, callout_border_color: str,
                        line_color: str, line_thickness: int,
                        line_start_face_ratio_x: float, line_start_face_ratio_y: float,
                        curl_control_factor1_x: float, curl_control_factor1_y: float,
                        curl_control_factor2_x: float, curl_control_factor2_y: float,
                        arrow_head_size_factor: float,
                        fallback_behavior: str, multiple_face_behavior: str):

        if not self.face_cascade:
            print("FaceCalloutNode: Face cascade not loaded.")
            if fallback_behavior == "Error":
                raise RuntimeError("Face cascade classifier not loaded.")
            pil_input_image_for_blank = tensor_to_pil(image)
            if fallback_behavior == "Blank Image":
                blank_img = Image.new("RGB", pil_input_image_for_blank.size, (0,0,0))
                return (pil_to_tensor(blank_img),)
            return (image,)

        pil_input_image = tensor_to_pil(image)
        output_image_pil = pil_input_image.copy() # Work on a copy
        draw = ImageDraw.Draw(output_image_pil, "RGBA")
        img_width, img_height = output_image_pil.size

        cv_image = np.array(pil_input_image.convert('L')) # Convert to grayscale for detection
        faces = self.face_cascade.detectMultiScale(cv_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("FaceCalloutNode: No faces detected.")
            if fallback_behavior == "Return Original": return (pil_to_tensor(output_image_pil),)
            if fallback_behavior == "Error": raise ValueError("No faces detected.")
            blank_img = Image.new("RGB", pil_input_image.size, (0,0,0))
            return (pil_to_tensor(blank_img),)

        # Select face
        if multiple_face_behavior == "Largest Face" and len(faces) > 1:
            face_idx = np.argmax([fw * fh for _, _, fw, fh in faces])
            fx, fy, fw, fh = faces[face_idx]
        else: # First Detected or only one face
            fx, fy, fw, fh = faces[0]

        # 1. Create Circular Face Crop for the Callout
        crop_size = max(fw, fh)
        padded_crop_dim = int(crop_size * (1 + face_padding_percent))
        radius_orig_crop = padded_crop_dim // 2

        fc_x1 = (fx + fw // 2) - radius_orig_crop
        fc_y1 = (fy + fh // 2) - radius_orig_crop
        
        # Crop from original image
        face_crop_pil = pil_input_image.crop((
            max(0, fc_x1), max(0, fc_y1),
            min(img_width, fc_x1 + padded_crop_dim), min(img_height, fc_y1 + padded_crop_dim)
        ))
        face_crop_pil = face_crop_pil.resize((padded_crop_dim, padded_crop_dim), Image.Resampling.LANCZOS)


        # Scale for callout
        callout_dim = int(max(fw, fh) * callout_scale_factor)
        if callout_dim <=0 : callout_dim = 50 # minimum size
        callout_radius = callout_dim // 2
        
        scaled_face_crop_pil = face_crop_pil.resize((callout_dim, callout_dim), Image.Resampling.LANCZOS)

        # Make it circular with border
        final_callout_pil = Image.new("RGBA", (callout_dim, callout_dim), (0,0,0,0))
        temp_draw = ImageDraw.Draw(final_callout_pil)

        if callout_border_thickness > 0:
            try:
                parsed_border_color = ImageColor.getrgb(callout_border_color)
            except ValueError:
                parsed_border_color = (255,255,255) # white fallback
            temp_draw.ellipse((0, 0, callout_dim-1, callout_dim-1), fill=parsed_border_color)

        # Inner circle for face
        mask = Image.new('L', (callout_dim, callout_dim), 0)
        ImageDraw.Draw(mask).ellipse(
            (callout_border_thickness, callout_border_thickness,
             callout_dim - callout_border_thickness, callout_dim - callout_border_thickness),
            fill=255
        )
        final_callout_pil.paste(scaled_face_crop_pil, (0,0), mask)


        # 2. Position Callout Circle
        callout_paste_x = int(img_width * callout_pos_x_percent) - callout_radius
        callout_paste_y = int(img_height * callout_pos_y_percent) - callout_radius
        callout_center_x = callout_paste_x + callout_radius
        callout_center_y = callout_paste_y + callout_radius

        # 3. Draw Connecting Line (Cubic Bezier)
        # Line start point on original face
        line_start_x = fx + int(fw * line_start_face_ratio_x)
        line_start_y = fy + int(fh * line_start_face_ratio_y)
        p0 = (line_start_x, line_start_y)

        # Line end point on edge of callout circle
        angle_to_orig_face = math.atan2(line_start_y - callout_center_y, line_start_x - callout_center_x)
        p3_x = callout_center_x + callout_radius * math.cos(angle_to_orig_face)
        p3_y = callout_center_y + callout_radius * math.sin(angle_to_orig_face)
        p3 = (int(p3_x), int(p3_y))

        # Control points for Bezier (these define the "curl")
        # Vector from start to end
        vec_x, vec_y = p3[0] - p0[0], p3[1] - p0[1]
        line_length = math.sqrt(vec_x**2 + vec_y**2)
        if line_length == 0: line_length = 1 # Avoid division by zero

        # Control Point 1
        cp1_x = p0[0] + vec_x * curl_control_factor1_x - vec_y * curl_control_factor1_y * (line_length / (callout_dim if callout_dim > 0 else 100))
        cp1_y = p0[1] + vec_y * curl_control_factor1_x + vec_x * curl_control_factor1_y * (line_length / (callout_dim if callout_dim > 0 else 100))
        p1 = (int(cp1_x), int(cp1_y))

        # Control Point 2
        cp2_x = p0[0] + vec_x * curl_control_factor2_x - vec_y * curl_control_factor2_y * (line_length / (callout_dim if callout_dim > 0 else 100))
        cp2_y = p0[1] + vec_y * curl_control_factor2_x + vec_x * curl_control_factor2_y * (line_length / (callout_dim if callout_dim > 0 else 100))
        p2 = (int(cp2_x), int(cp2_y))
        
        try:
            parsed_line_color = ImageColor.getrgb(line_color)
        except ValueError:
            parsed_line_color = (0,0,0) # black fallback

        # Draw Bezier curve by plotting segments
        num_segments = 30
        points = [p0]
        for i in range(1, num_segments + 1):
            t = i / num_segments
            points.append(self._get_bezier_point(t, p0, p1, p2, p3))
        draw.line(points, fill=parsed_line_color, width=line_thickness, joint="curve")

        # Draw arrowhead
        if arrow_head_size_factor > 0:
            arrow_len = callout_dim * arrow_head_size_factor
            # Last segment of the Bezier for direction
            if len(points) >= 2:
                last_p = points[-1]
                second_last_p = points[-2]
                angle_rad = math.atan2(last_p[1] - second_last_p[1], last_p[0] - second_last_p[0])

                # Arrowhead points
                # p3 is the tip
                p_arrow1_x = last_p[0] - arrow_len * math.cos(angle_rad - math.pi / 7)
                p_arrow1_y = last_p[1] - arrow_len * math.sin(angle_rad - math.pi / 7)
                p_arrow2_x = last_p[0] - arrow_len * math.cos(angle_rad + math.pi / 7)
                p_arrow2_y = last_p[1] - arrow_len * math.sin(angle_rad + math.pi / 7)
                draw.polygon([last_p, (p_arrow1_x, p_arrow1_y), (p_arrow2_x, p_arrow2_y)], fill=parsed_line_color)


        # 4. Paste Callout Circle
        output_image_pil.paste(final_callout_pil, (callout_paste_x, callout_paste_y), final_callout_pil)

        return (pil_to_tensor(output_image_pil),)

# Node Registration (assuming this file is in its own package folder with an __init__.py)
# Example __init__.py content for a folder named "FaceCalloutCustomNode":
#
# from .FaceCalloutNode import FaceCalloutNode # Assuming this file is FaceCalloutNode.py
#
# NODE_CLASS_MAPPINGS = {
#     "FaceCallout": FaceCalloutNode,
# }
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "FaceCallout": "Face Callout Effect"
# }
# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']