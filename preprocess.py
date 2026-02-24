import cv2
import mediapipe as mp
from mediapipe.python.solutions import holistic
import numpy as np
from pathlib import Path
from tqdm import tqdm

class SignLangVidPreprocessor:
    def __init__(self, video_dir: str, npy_dir: str):
        self.video_dir = Path(video_dir)
        self.npy_dir = Path(npy_dir)
        self.npy_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe Solutions
        self.holistic = holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks(self, results) -> np.ndarray:
        # Extract 33 Pose pts, 21 LH pts, 21 RH pts (all x, y, z)
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        
        return np.concatenate([pose, lh, rh]) # Total: 225 features

    def process_all(self):
        video_files = list(self.video_dir.glob("*.mp4"))
        
        for video_path in tqdm(video_files, desc="Preprocessing WLASL"):
            output_path = self.npy_dir / f"{video_path.stem}.npy"
            
            # Skip if already processed
            if output_path.exists():
                continue
                
            cap = cv2.VideoCapture(str(video_path))
            sequence = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Inference
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(image)
                
                # Save frame keypoints
                sequence.append(self.extract_landmarks(results))
                
            cap.release()
            
            if sequence:
                np.save(output_path, np.array(sequence))

        self.holistic.close()

if __name__ == "__main__":
    processor = SignLangVidPreprocessor(video_dir="G:/Projects/Python/Sign Language/SL-Data/videos", npy_dir="G:/Projects/Python/Sign Language/SL-Data/keypoints")
    processor.process_all()