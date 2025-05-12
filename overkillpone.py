# #!/usr/bin/env python
# # coding: utf-8

# from transformers import GLPNImageProcessor, GLPNForDepthEstimation
# import torch
# import numpy as np
# import cv2
# from ultralytics import YOLO
# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt

# # Load GLPN (Global-Local Path Network) depth model
# processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
# depth_model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# # Move model to GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# depth_model.to(device)

# # Load YOLOv8n model
# yolo_model = YOLO("yolov8n.pt")

# # YOLO Class Labels
# CLASS_NAMES = {
#     0: "Person",
#     39: "Bottle",
#     60: "Cup",
#     75: "Chair",
#     9: 'traffic light',
#     10: 'fire hydrant', 
#     11: 'stop sign',
#     67: "cell phone", 
#     68: 'microwave', 
#     69: 'oven'
# }

# # Camera parameters
# IMAGE_WIDTH_PIXELS = 1280
# IMAGE_HEIGHT_PIXELS = 720
# FOCAL_LENGTH_MM = 3.5
# SENSOR_WIDTH_MM = 3.6
# SENSOR_HEIGHT_MM = 2.7
# FOCAL_LENGTH_PIXELS = (FOCAL_LENGTH_MM * IMAGE_WIDTH_PIXELS) / SENSOR_WIDTH_MM

# # Field of View (for real-world dimension calculation)
# HFOV_rad = 2 * np.arctan((SENSOR_WIDTH_MM / 2) / FOCAL_LENGTH_MM)
# VFOV_rad = 2 * np.arctan((SENSOR_HEIGHT_MM / 2) / FOCAL_LENGTH_MM)
# HFOV_deg = np.degrees(HFOV_rad)
# VFOV_deg = np.degrees(VFOV_rad)

# depth_correction_factor = 0.61 / 1.29
# def resize_with_padding(image, target_size=(640, 480)):
#         return ImageOps.pad(image, target_size, method=Image.LANCZOS, color=(0, 0, 0), centering=(0.5, 0.5))
    
# def process_image(image):
#     """
#     Takes a PIL image as input and returns processed output image with bounding boxes, depth, width, height info.
#     """
#     # Save original
#     or_img = image
#     original_image_size = image.size

#     # Resize image for YOLO + Depth
#     image = resize_with_padding(or_img, target_size=(640, 480))
#     image_np = np.array(image)  # NumPy for YOLO
#     or_np = np.array(or_img)    # Original image

#     # Prepare image for depth model
#     inputs = processor(images=image, return_tensors="pt").to(device)

#     # Predict depth
#     with torch.no_grad():
#         outputs = depth_model(**inputs)
#         predicted_depth = outputs.predicted_depth

#     # Interpolate to original size
#     depth_map = torch.nn.functional.interpolate(
#         predicted_depth.unsqueeze(1),
#         size=image.size[::-1],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze().cpu().numpy()

#     # Run YOLO object detection
#     results = yolo_model(image_np)

#     # Compute scaling factors
#     orig_w, orig_h = original_image_size
#     resized_w, resized_h = image.size
#     scale_x = orig_w / resized_w
#     scale_y = orig_h / resized_h

#     heights = []
#     widths = []
#     object_depths = []
#     class_names = []

#     # Extract bounding boxes and labels
#     for result in results:
#         boxes = result.boxes.xyxy.cpu().numpy()
#         labels = result.boxes.cls.cpu().numpy()

#     # Draw results
#     image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

#     for i, box in enumerate(boxes):
#         x_min, y_min, x_max, y_max = map(int, box)

#         # Scale bounding box
#         scaled_x_min = int(x_min * scale_x)
#         scaled_y_min = int(y_min * scale_y)
#         scaled_x_max = int(x_max * scale_x)
#         scaled_y_max = int(y_max * scale_y)

#         # Width and height in pixels
#         scaled_pixel_width = scaled_x_max - scaled_x_min
#         scaled_pixel_height = scaled_y_max - scaled_y_min

#         # Depth inside bbox
#         object_depth = depth_map[y_min:y_max, x_min:x_max]

#         if object_depth.size > 0:
#             object_depth = object_depth[(object_depth > 0) & (object_depth < 10)]
#             median_depth = np.median(object_depth) if object_depth.size > 0 else 0
#         else:
#             median_depth = 0

#         median_depth *= depth_correction_factor
#         object_depths.append(median_depth)

#         real_width = 2 * median_depth * np.tan(np.radians(HFOV_deg / 2)) * (scaled_pixel_width / orig_w)
#         real_height = 2 * median_depth * np.tan(np.radians(VFOV_deg / 2)) * (scaled_pixel_height / orig_h)

#         widths.append(real_width)
#         heights.append(real_height)

#         # Class label
#         class_name = CLASS_NAMES.get(int(labels[i]), f"Class {labels[i]}")
#         class_names.append(class_name)

#         # Draw box and text
#         text = f"{class_name} D:{median_depth:.2f}m H:{real_height:.2f}m W:{real_width:.2f}m"
#         text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
#         text_x, text_y = x_min, y_min - 10
#         if text_y - text_size[1] < 0:
#             text_y = y_min + 20

#         cv2.rectangle(image_np_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#         cv2.rectangle(
#             image_np_bgr,
#             (text_x, text_y - text_size[1] - 2),
#             (text_x + text_size[0] + 2, text_y + 2),
#             (0, 255, 0),
#             -1
#         )
#         cv2.putText(
#             image_np_bgr,
#             text,
#             (text_x, text_y),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (0, 0, 0),
#             2
#         )

#     # Convert back to RGB
#     output_image = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
#     output_image_pil = Image.fromarray(output_image)

#     return output_image_pil, object_depths, widths, heights, class_names















from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image, ImageOps

# Load depth model
processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
depth_model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

device = "cuda" if torch.cuda.is_available() else "cpu"
depth_model.to(device)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# YOLO class labels (filtered set)
CLASS_NAMES = {
    0: "Person",
    39: "Bottle",
    60: "Cup",
    75: "Chair",
    9: 'Traffic Light',
    10: 'Fire Hydrant', 
    11: 'Stop Sign',
    67: "Cell Phone", 
    68: 'Microwave', 
    69: 'Oven'
}

# Camera + sensor parameters
IMAGE_WIDTH_PIXELS = 1280
IMAGE_HEIGHT_PIXELS = 720
FOCAL_LENGTH_MM = 3.5
SENSOR_WIDTH_MM = 3.6
SENSOR_HEIGHT_MM = 2.7

FOCAL_LENGTH_PIXELS = (FOCAL_LENGTH_MM * IMAGE_WIDTH_PIXELS) / SENSOR_WIDTH_MM
HFOV_rad = 2 * np.arctan((SENSOR_WIDTH_MM / 2) / FOCAL_LENGTH_MM)
VFOV_rad = 2 * np.arctan((SENSOR_HEIGHT_MM / 2) / FOCAL_LENGTH_MM)
HFOV_deg = np.degrees(HFOV_rad)
VFOV_deg = np.degrees(VFOV_rad)

depth_correction_factor = 0.61 / 1.29

def resize_with_padding(image, target_size=(640, 480)):
    return ImageOps.pad(image, target_size, method=Image.LANCZOS, color=(0, 0, 0), centering=(0.5, 0.5))

def process_image(pil_image):
    """
    Input:
        pil_image (PIL.Image): Any RGB image (from IP cam or local)
    
    Returns:
        output_image_pil (PIL.Image): Image with YOLO boxes + info
        object_depths (list): depths per object
        widths (list): estimated width (m)
        heights (list): estimated height (m)
        class_names (list): detected object names
    """
    original_size = pil_image.size
    resized_img = resize_with_padding(pil_image, target_size=(640, 480))

    # Convert to NumPy for YOLO
    image_np = np.array(resized_img)

    # Prepare for GLPN depth
    inputs = processor(images=resized_img, return_tensors="pt").to(device)
    with torch.no_grad():
        depth_output = depth_model(**inputs).predicted_depth

    # Resize depth map to original image size
    depth_map = torch.nn.functional.interpolate(
        depth_output.unsqueeze(1),
        size=resized_img.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # Run YOLO
    yolo_results = yolo_model(image_np)
    image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    orig_w, orig_h = original_size
    resized_w, resized_h = resized_img.size
    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    object_depths, widths, heights, class_names = [], [], [], []

    for result in yolo_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()

        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = map(int, box)

            # Get scaled box for original dimensions
            scaled_x_min = int(x_min * scale_x)
            scaled_y_min = int(y_min * scale_y)
            scaled_x_max = int(x_max * scale_x)
            scaled_y_max = int(y_max * scale_y)

            scaled_pixel_width = scaled_x_max - scaled_x_min
            scaled_pixel_height = scaled_y_max - scaled_y_min

            object_region = depth_map[y_min:y_max, x_min:x_max]
            if object_region.size > 0:
                object_region = object_region[(object_region > 0) & (object_region < 10)]
                median_depth = np.median(object_region) if object_region.size > 0 else 0
            else:
                median_depth = 0

            median_depth *= depth_correction_factor
            object_depths.append(median_depth)

            real_width = 2 * median_depth * np.tan(np.radians(HFOV_deg / 2)) * (scaled_pixel_width / orig_w)
            real_height = 2 * median_depth * np.tan(np.radians(VFOV_deg / 2)) * (scaled_pixel_height / orig_h)

            widths.append(real_width)
            heights.append(real_height)

            cls_id = int(labels[i])
            class_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            class_names.append(class_name)

            # Draw annotated box and label
            label_text = f"{class_name} D:{median_depth:.2f}m H:{real_height:.2f}m W:{real_width:.2f}m"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x, text_y = x_min, y_min - 10 if y_min - 10 > 0 else y_min + 20

            cv2.rectangle(image_np_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.rectangle(image_np_bgr,
                          (text_x, text_y - text_size[1] - 2),
                          (text_x + text_size[0] + 2, text_y + 2),
                          (0, 255, 0), -1)
            cv2.putText(image_np_bgr, label_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    output_image_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
    output_image_pil = Image.fromarray(output_image_rgb)

    return output_image_pil, object_depths, widths, heights, class_names
