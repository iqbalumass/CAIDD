import os
import cv2
import torch
import face_alignment
from PIL import Image
from tqdm import tqdm

# Paths
input_dir = 'video/frames'               # or 'id_folder' if you're doing just the identity
output_dir = 'video/aligned_frames'      # output directory
os.makedirs(output_dir, exist_ok=True)

# Face alignment model
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
output_size = 256
padding = 0.3  # 30% padding around the detected face box

def crop_face(image):
    preds = fa.get_landmarks(image)
    if preds is None or len(preds) == 0:
        return None  # No face detected

    landmarks = preds[0]  # Only consider the first face
    x_min = int(landmarks[:, 0].min())
    y_min = int(landmarks[:, 1].min())
    x_max = int(landmarks[:, 0].max())
    y_max = int(landmarks[:, 1].max())

    h, w, _ = image.shape
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    x_pad = int(bbox_w * padding)
    y_pad = int(bbox_h * padding)

    # Add margin and clip to image bounds
    x1 = max(x_min - x_pad, 0)
    y1 = max(y_min - y_pad, 0)
    x2 = min(x_max + x_pad, w)
    y2 = min(y_max + y_pad, h)

    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(resized)

# Process all images in the folder
for file in tqdm(sorted(os.listdir(input_dir))):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(input_dir, file)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_img = crop_face(image_rgb)

        if face_img is not None:
            face_img.save(os.path.join(output_dir, file))
        else:
            print(f"[WARNING] No face detected in {file}")
