
import cv2 as cv
import numpy as np
import os
import json
#import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark


from pythonosc import udp_client

# ── Pose connections (33 landmarks, same topology as BlazePose) ──────────────
# Each tuple is (start_index, end_index)
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Landmark names by index for reference
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# ── Drawing ──────────────────────────────────────────────────────────────────
def draw_pose(image: np.ndarray, landmarks: list[NormalizedLandmark]) -> np.ndarray:
    """Draw landmarks and skeleton on image using only OpenCV."""
    h, w = image.shape[:2]
    out = image.copy()

    # Convert normalised coords → pixel coords
    # visibility < 0.5 means the point is likely not visible
    pts = []
    for lm in landmarks:
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        pts.append((cx, cy, lm.visibility))

    # Draw connections first (so dots appear on top)
    for start_idx, end_idx in POSE_CONNECTIONS:
        start_vis = pts[start_idx][2]
        end_vis   = pts[end_idx][2]
        if start_vis < 0.5 or end_vis < 0.5:
            continue  # skip invisible joints
        cv.line(
            out,
            (pts[start_idx][0], pts[start_idx][1]),
            (pts[end_idx][0],   pts[end_idx][1]),
            color=(0, 255, 255),   # cyan skeleton
            thickness=2,
            lineType=cv.LINE_AA,
        )

    # Draw landmark dots
    for cx, cy, vis in pts:
        if vis < 0.5:
            continue
        cv.circle(out, (cx, cy), radius=4, color=(0, 0, 255), thickness=-1)   # red fill
        cv.circle(out, (cx, cy), radius=4, color=(255, 255, 255), thickness=1) # white border

    return out


# OSC Configuration
OSC_IP = "127.0.0.1"
OSC_PORT = 57120
osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# 1. Define the path to your folder
folder_path = '/home/pemb/python/mediapipe/poses'
#out_path = '/home/pemb/python/mediapipe/annotate'
out_path =  Path(folder_path) / "annotated"
images = []
images_name = []

OUTPUT_FILE = 'all_landmarks.json'
all_results = {}
#all_images_result = []
all_images_result = {}

# list to save the landmarks data
#landmarks_list = []

# 2. Iterate through files in the folder
for filename in os.listdir(folder_path):
    # Check for common image extensions
    #if filename.endswith((".jpg", ".png", ".jpeg")):
    img_path = os.path.join(folder_path, filename)
        
    # 3. Load the image
    bgr = cv.imread(img_path)
    #timg = mp.Image.create_from_file(img_path)
        
    if bgr is not None:
        images.append(bgr)
        images_name.append(filename)
        # Optional: Display to verify
        # cv.imshow('Image', img)
        # cv.waitKey(500) # Wait 500ms

model_path = '/home/pemb/python/mediapipe/pose_landmarker_heavy.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = model_path),
    num_poses = 1,
    output_segmentation_masks = True,
    running_mode = VisionRunningMode.IMAGE)


with PoseLandmarker.create_from_options(options) as landmarker:
    for index, img in enumerate(images):
        name = images_name[index]
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = img)
        #mp_image = mp.Image.create_from_file(img)

        result = landmarker.detect(mp_image)
        n_poses = len(result.pose_landmarks)

        
        print('pose landmarker result: {}'.format(result))
        
        #annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), result)
        #cv.imshow('frame', cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
        
        landmark_data = []
        for id, landmarkResult in enumerate(result.pose_landmarks):
            for i, lm in enumerate(landmarkResult):
                landmark_data.append(lm.x)
                landmark_data.append(lm.y)
                landmark_data.append(lm.z)

            annotated = img.copy()
            annotated = draw_pose(img, landmarkResult)
            cv.imwrite(str(out_path / f"{index:04d}.jpg"), annotated)

                
        all_images_result[index] = {"landmarks": landmark_data}


# Save all results to a JSON file
with open('landmarks_output.json', 'w') as f:
    json.dump(all_images_result, f, indent = 4)

print(f"\nFinished! Data for {len(all_images_result)} images saved to {OUTPUT_FILE}")

print(len(images))
