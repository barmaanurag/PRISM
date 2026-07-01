import sys
import os

# Add the parent directory to sys.path so we can import mediapipe_inference
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mediapipe_inference import process_video_with_action_recognition
from finalfacerecognition import Track

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_inference.py <video_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    Track._count = 0
    
    print(f"Starting processing for video: {video_path}")
    
    try:
        output, top5_action_log, action_logger = process_video_with_action_recognition(
            video_path      = video_path,
            output_path     = 'output_action.mp4',
            reid_model_path = 'best_model.onnx',
            conf_threshold  = 0.25,
            max_frames      = None,
            buffer_len      = 90,
            bbox_scale      = 1.0,
            iou_thresh      = 0.20,
            dist_thresh     = 150,
            persist_frames  = 10,
            inference_every = 8,
            dataset_dir     = 'dataset',
        )
        print("\nINFERENCE_COMPLETE")
    except Exception as e:
        print(f"\nERROR: {e}")
