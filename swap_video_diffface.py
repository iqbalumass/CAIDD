import os
from tqdm import tqdm
from PIL import Image
from optimization.arguments import get_arguments
from optimization.image_editor import ImageEditor

# --- CONFIGURATION ---
SOURCE_FACE_PATH = 'video/target/id.jpg'
FRAMES_DIR = 'video/aligned_frames'
OUTPUT_DIR = 'video/swapped_frames'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- SETUP ONCE ---
args = get_arguments()
args.source_path = SOURCE_FACE_PATH
image_editor = ImageEditor(args)

# --- SWAP AND SAVE EACH FRAME ---
frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(('.png', '.jpg'))])

for frame_file in tqdm(frame_files):
    frame_path = os.path.join(FRAMES_DIR, frame_file)
    output_path = os.path.join(OUTPUT_DIR, frame_file)

    args.target_path = frame_path
    args.output_path = output_path
    image_editor.args = args

    try:
        image_editor.edit_image_by_prompt()
    except Exception as e:
        print(f"[ERROR] Skipping {frame_file} due to: {e}")
