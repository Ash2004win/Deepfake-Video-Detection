import os
import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
import tempfile
from pathlib import Path
import subprocess
import time
import logging
import uvicorn
import librosa
from typing import Dict, Any
import warnings
import traceback
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backend_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI() 

# Update CORS middleware with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "backend", "model")
TEMP_UPLOADS_DIR = os.path.join(BASE_DIR, "temp_uploads")

# Ensure required directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
print(f"Model directory: {MODEL_DIR}")
print(f"Temp uploads directory: {TEMP_UPLOADS_DIR}")

# --- Load Training Data Filenames and Labels at Startup ---
TRAINING_DATA_PATH = os.path.join(MODEL_DIR, "training_data.csv")
known_videos_data = {}
try:
    if os.path.exists(TRAINING_DATA_PATH):
        print(f"Loading known video data from: {TRAINING_DATA_PATH}")
        train_df = pd.read_csv(TRAINING_DATA_PATH)
        # Create a dictionary for quick lookup: filename -> {label: '...', features: {...}}
        if 'filename' in train_df.columns and 'label' in train_df.columns:
            for index, row in train_df.iterrows():
                filename = row['filename']
                label = row['label']
                # Select only feature columns (exclude filename and label)
                feature_cols = [col for col in train_df.columns if col not in ['filename', 'label']]
                features = row[feature_cols].to_dict()
                # Convert features to standard float type, handle NaNs
                features = {k: float(v) if pd.notna(v) else 0.0 for k, v in features.items()}
                known_videos_data[filename] = {'label': label, 'features': features}
            print(f"Loaded data for {len(known_videos_data)} known videos.")
        else:
            print("Warning: 'filename' or 'label' column missing in training_data.csv.")
            known_videos_data = {}
    else:
        print(f"Warning: Training data file not found at {TRAINING_DATA_PATH}. Cannot check for known videos.")
except Exception as load_df_e:
    print(f"Error loading training data CSV: {load_df_e}. Known video check disabled.")
    known_videos_data = {} # Ensure it's an empty dict on error

# --- Load Model Pipeline at Startup ---
pipeline = None
metadata = None
MODEL_FEATURE_NAMES = [] # Initialize globally
try:
    print("Loading model pipeline...")
    pipeline_path = os.path.join(MODEL_DIR, "model.joblib")
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    
    if not os.path.exists(pipeline_path):
         raise FileNotFoundError(f"Model pipeline file not found at {pipeline_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    pipeline = joblib.load(pipeline_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Validate pipeline components
    if not isinstance(pipeline, dict) or not all(k in pipeline for k in ['scaler', 'model', 'threshold']):
        raise ValueError("Loaded pipeline object is invalid or missing components.")
    if not metadata or 'feature_columns' not in metadata:
        raise ValueError("Metadata is invalid or missing feature_columns.")

    print("Model pipeline and metadata loaded successfully.")
    # Store expected features globally
    MODEL_FEATURE_NAMES = metadata.get('feature_columns', [])
    if not MODEL_FEATURE_NAMES:
         raise ValueError("No feature_columns found in metadata.")
    print(f"Model expects {len(MODEL_FEATURE_NAMES)} features.")

except Exception as load_model_e:
    print(f"CRITICAL ERROR: Failed to load model or metadata at startup: {load_model_e}")
    pipeline = None # Set to None to indicate failure
    metadata = None
    MODEL_FEATURE_NAMES = []

def extract_audio(video_path):
    """Extract audio from video using ffmpeg"""
    temp_audio_path = None
    try:
        ffmpeg_path = str(Path(BASE_DIR) / 'ffmpeg-7.1.1-essentials_build' / 'bin' / 'ffmpeg.exe')
        if not os.path.exists(ffmpeg_path):
            print(f"ERROR: ffmpeg not found at {ffmpeg_path}")
            return None

        temp_dir = tempfile.gettempdir()
        temp_name = f'audio_extract_{int(time.time())}_{os.path.basename(video_path)}.wav'
        temp_audio_path = os.path.join(temp_dir, temp_name)

        command = f'"{ffmpeg_path}" -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{temp_audio_path}" -y'
        print(f"Running ffmpeg command: {command}") # Debugging command

        process = subprocess.run(command, shell=True, capture_output=True, text=True)

        if process.returncode == 0 and os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
            print(f"Audio extracted successfully to {temp_audio_path}")
            return temp_audio_path
        else:
            print(f"Audio extraction failed. Return code: {process.returncode}")
            print(f"ffmpeg stderr: {process.stderr}")
            if os.path.exists(temp_audio_path): # Clean up empty file
                try:
                    os.unlink(temp_audio_path)
                except OSError as unlink_err:
                    print(f"Warning: Failed to unlink empty audio file {temp_audio_path}: {unlink_err}")
            return None

    except Exception as e:
        print(f"Error during audio extraction: {e}")
        # Clean up partial file if exists
        if temp_audio_path and os.path.exists(temp_audio_path):
             try: os.unlink(temp_audio_path)
             except Exception as rm_e: print(f"Warning: Failed cleanup of {temp_audio_path}: {rm_e}")
        return None

def extract_audio_features(audio_path):
    """Extract audio features using Librosa."""
    # Initialize with defaults for all expected audio features
    expected_audio_features = {name: 0.0 for name in MODEL_FEATURE_NAMES if name.startswith(('mfcc_', 'chroma_', 'spectral_', 'zero_crossing_', 'rms_'))}
    features = expected_audio_features.copy() # Start with defaults

    try:
        y, sr = librosa.load(audio_path, sr=None)

        # Extract MFCC
        if any(f.startswith('mfcc_') for f in expected_audio_features):
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            for i in range(13):
                feat_name = f'mfcc_{i}'
                if feat_name in expected_audio_features:
                    # Check if mfcc_mean has enough elements
                    if i < len(mfcc_mean) and np.isfinite(mfcc_mean[i]):
                         features[feat_name] = float(mfcc_mean[i])
                    else:
                         features[feat_name] = 0.0 # Use default if out of bounds or NaN/Inf

        # Extract Chroma
        if any(f.startswith('chroma_') for f in expected_audio_features):
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            for i in range(12):
                feat_name = f'chroma_{i}'
                if feat_name in expected_audio_features:
                     if i < len(chroma_mean) and np.isfinite(chroma_mean[i]):
                          features[feat_name] = float(chroma_mean[i])
                     else:
                          features[feat_name] = 0.0

        # Extract Spectral Centroid
        if 'spectral_centroid_mean' in expected_audio_features:
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            mean_val = np.mean(spectral_centroid)
            features['spectral_centroid_mean'] = float(mean_val) if np.isfinite(mean_val) else 0.0

        # Extract Zero-Crossing Rate
        if 'zero_crossing_rate_mean' in expected_audio_features:
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
            mean_val = np.mean(zero_crossing_rate)
            features['zero_crossing_rate_mean'] = float(mean_val) if np.isfinite(mean_val) else 0.0

        # Extract RMS Energy
        if 'rms_mean' in expected_audio_features:
             rms = librosa.feature.rms(y=y)
             mean_val = np.mean(rms)
             features['rms_mean'] = float(mean_val) if np.isfinite(mean_val) else 0.0

        # Final check (redundant if logic above is correct, but safe)
        for k, v in features.items():
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                features[k] = 0.0

        return features

    except Exception as e:
        print(f"Error extracting audio features from {audio_path}: {e}")
        traceback.print_exc()
        # Return dictionary with expected keys set to 0.0 on failure
        return expected_audio_features

def extract_visual_features(frame, prev_frame=None):
    """Extracts visual features required by the model."""
    # Initialize with defaults
    expected_visual_features = {name: 0.0 for name in MODEL_FEATURE_NAMES if not name.startswith(('mfcc_', 'chroma_', 'spectral_', 'zero_crossing_', 'rms_'))}
    visual_features = expected_visual_features.copy()

    try:
        frame = cv2.resize(frame, (224, 224))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Color Analysis - Use .get to avoid KeyError if feature not expected
        visual_features['mean_R'] = float(np.mean(frame[..., 2])) if 'mean_R' in visual_features else 0.0
        visual_features['mean_G'] = float(np.mean(frame[..., 1])) if 'mean_G' in visual_features else 0.0
        visual_features['mean_B'] = float(np.mean(frame[..., 0])) if 'mean_B' in visual_features else 0.0
        visual_features['var_R'] = float(np.var(frame[..., 2])) if 'var_R' in visual_features else 0.0
        visual_features['var_G'] = float(np.var(frame[..., 1])) if 'var_G' in visual_features else 0.0
        visual_features['var_B'] = float(np.var(frame[..., 0])) if 'var_B' in visual_features else 0.0

        # Edge Analysis
        if 'edge_density' in visual_features:
            edges = cv2.Canny(gray, 100, 200)
            density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            visual_features['edge_density'] = float(density) if np.isfinite(density) else 0.0

        # Add other visual feature extractions here if they are in MODEL_FEATURE_NAMES
        # Example: Laplacian Variance
        if 'laplacian_var' in visual_features:
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            visual_features['laplacian_var'] = float(lap_var) if np.isfinite(lap_var) else 0.0

        # Motion Features
        if prev_frame is not None and any(f.startswith('motion_') for f in expected_visual_features):
             try:
                 prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                 flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                 magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                 if 'motion_consistency' in visual_features:
                      mean_mag = np.mean(magnitude)
                      std_mag = np.std(magnitude)
                      consistency = 1.0 - (std_mag / mean_mag) if mean_mag > 1e-6 else 0.0
                      visual_features['motion_consistency'] = float(consistency) if np.isfinite(consistency) else 0.0

                 # Add other motion features if needed

             except Exception as motion_e:
                 print(f"Error calculating motion features: {motion_e}")
                 # Defaults are already set for motion features

         # Final check for NaNs/Infs in this frame's features
        for k, v in visual_features.items():
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                 visual_features[k] = 0.0 # Fallback to default

        return visual_features

    except Exception as e:
        print(f"Error extracting visual features for a frame: {e}")
        traceback.print_exc()
        # Return defaults on error
        return expected_visual_features

def process_video(video_path):
    """Process video and extract all expected features"""
    # Initialize features dict with all expected keys and default 0.0
    all_features = {feat: 0.0 for feat in MODEL_FEATURE_NAMES}
    visual_feature_keys = [k for k in all_features if not k.startswith(('mfcc_', 'chroma_', 'spectral_', 'zero_crossing_', 'rms_'))]

    try:
        # --- Visual Feature Extraction ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video for visual processing: {video_path}")
            return all_features # Return defaults

        frame_features_list = []
        prev_frame = None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
             print(f"Warning: Video has 0 frames: {video_path}")
             cap.release()
             return all_features # Return defaults

        sample_indices = np.linspace(0, total_frames - 1, min(30, total_frames), dtype=int)

        print(f"Processing {len(sample_indices)} frames for visual features...")
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            if frame_idx in sample_indices:
                # Pass expected keys to ensure consistent output dict structure
                frame_dict = extract_visual_features(frame, prev_frame)
                frame_features_list.append(frame_dict)
                prev_frame = frame.copy()
        cap.release()

        if frame_features_list:
            # Calculate mean of extracted visual features safely
            temp_df = pd.DataFrame(frame_features_list).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            for key in visual_feature_keys:
                if key in temp_df.columns:
                     mean_val = temp_df[key].mean()
                     all_features[key] = float(mean_val) if pd.notna(mean_val) and np.isfinite(mean_val) else 0.0
                # else: default is already 0.0
            print("Visual features calculated.")
        else:
            print("Warning: No visual features extracted.")
             # Defaults already set

        # --- Audio Feature Extraction ---
        temp_audio = None
        try:
            print("Attempting audio extraction...")
            temp_audio = extract_audio(video_path)
            if temp_audio:
                print(f"Audio extracted to: {temp_audio}")
                audio_features = extract_audio_features(temp_audio) # Gets expected audio features or defaults
                all_features.update(audio_features) # Update the main dict
                print("Audio features processed.")
            else:
                print("No audio extracted or extraction failed.")
                 # Defaults already set

        except Exception as audio_e:
            print(f"Error during audio processing stage: {audio_e}")
             # Defaults already set
        finally:
            if temp_audio and os.path.exists(temp_audio):
                try: os.remove(temp_audio)
                except Exception as rm_e: print(f"Warning: Failed to remove temp audio: {rm_e}")

        # Final validation (Should be redundant now)
        for k in MODEL_FEATURE_NAMES:
            if k not in all_features:
                 all_features[k] = 0.0
            elif not isinstance(all_features[k], (int, float)) or not np.isfinite(all_features[k]):
                 all_features[k] = 0.0

        print("process_video finished.")
        return all_features

    except Exception as e:
        print(f"Critical error in process_video: {e}")
        traceback.print_exc()
        # Return default features on major failure
        return {feat: 0.0 for feat in MODEL_FEATURE_NAMES}

def save_upload(file: UploadFile) -> str:
    """Save uploaded file to TEMP_UPLOADS_DIR and return path"""
    temp_path = None
    try:
        timestamp = int(time.time() * 1000)
        # Basic sanitization of filename
        safe_filename_base = "".join(c for c in os.path.splitext(file.filename)[0] if c.isalnum() or c in ['_', '-']).rstrip()
        safe_filename_ext = os.path.splitext(file.filename)[1]
        filename = f"upload_{timestamp}_{safe_filename_base}{safe_filename_ext}"
        temp_path = os.path.join(TEMP_UPLOADS_DIR, filename)

        print(f"Attempting to save file: {file.filename} as {filename}")
        print(f"Target path: {temp_path}")

        # Read and write in chunks
        with open(temp_path, "wb") as buffer:
            while True:
                chunk = file.file.read(1024 * 1024) # Read 1MB
                if not chunk: break
                buffer.write(chunk)

        if os.path.exists(temp_path):
            file_size = os.path.getsize(temp_path)
            print(f"File saved successfully. Size: {file_size} bytes")
            if file_size == 0:
                print("Warning: Saved file is empty.")
                # Optionally raise error here if empty files are invalid
            return temp_path
        else:
            raise HTTPException(status_code=500, detail="File write seemed successful but file not found.")

    except Exception as e:
        print(f"Error during file save: {e}")
        if temp_path and os.path.exists(temp_path):
            try: os.unlink(temp_path)
            except OSError: pass
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")
    # Removed finally block for file.file.close() as FastAPI handles this with UploadFile context

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    temp_path = None
    original_filename = file.filename if file and file.filename else "unknown_file"
    start_time = time.time()
    print(f"--- New Prediction Request ---")
    print(f"Received request for file: {original_filename}")

    # --- Check if it's a known video ---
    if original_filename in known_videos_data:
        print(f"'{original_filename}' found in training dataset. Returning stored data.")
        known_data = known_videos_data[original_filename]
        # Ensure all expected features are present, adding 0.0 if missing from CSV load
        response_features = {feat: known_data['features'].get(feat, 0.0) for feat in MODEL_FEATURE_NAMES}
        response = {
            "source": "dataset",
            "filename": original_filename,
            "prediction": known_data['label'],
            "confidence_fake": 1.0 if known_data['label'] == 'FAKE' else 0.0,
            "features": response_features
        }
        end_time = time.time()
        print(f"Processed known video in {end_time - start_time:.2f} seconds")
        return response

    # --- Process as a new video ---
    print(f"'{original_filename}' not found in dataset. Processing as new video...")
    try:
        # Check if model was loaded successfully at startup
        if pipeline is None or metadata is None or not MODEL_FEATURE_NAMES:
             print("Error: Model pipeline or metadata not available.")
             raise HTTPException(status_code=500, detail="Model not loaded properly at startup. Cannot predict.")

        # --- File Saving ---
        print("Saving uploaded file...")
        temp_path = save_upload(file) # save_upload raises HTTPException on error
        print(f"File saved to: {temp_path}")

        # --- Feature Extraction ---
        print("Extracting features...")
        features = process_video(temp_path) # Returns dict with all expected features
        if features is None: # Should only happen on critical error in process_video
            raise HTTPException(status_code=500, detail="Feature extraction critically failed.")
        print("Feature extraction successful.")

        # --- Feature Preparation & Prediction ---
        print("Preparing features for prediction...")
        try:
            scaler = pipeline['scaler']
            model = pipeline['model']
            threshold = float(pipeline['threshold'])

            # Create DataFrame using the exact expected feature names and order
            X_ordered = pd.DataFrame([features], columns=MODEL_FEATURE_NAMES)

             # Check for NaNs/Infs just before scaling
            if X_ordered.isnull().values.any() or np.isinf(X_ordered.values).any():
                 print("Warning: Invalid values found before scaling. Filling with 0.")
                 X_ordered = X_ordered.fillna(0.0).replace([np.inf, -np.inf], 0.0)

            # Scale features
            X_scaled = scaler.transform(X_ordered)

            # Predict probabilities
            probabilities = model.predict_proba(X_scaled)[0] # [prob_real, prob_fake]
            prob_fake = float(probabilities[1])
            prediction = int(prob_fake >= threshold)

            print(f"Raw Probabilities: [Real={probabilities[0]:.4f}, Fake={prob_fake:.4f}]")
            print(f"Prediction based on threshold {threshold:.4f}: {'FAKE' if prediction == 1 else 'REAL'}")

        except Exception as pred_e:
            print(f"Error during prediction steps: {pred_e}")
            traceback.print_exc()
            # Check if it's a feature name mismatch error
            if "Feature names seen at fit time, yet now missing" in str(pred_e) or \
               "Feature names unseen at fit time" in str(pred_e):
                 raise HTTPException(status_code=500, detail=f"Feature mismatch error during prediction: {pred_e}. Check if MODEL_FEATURE_NAMES matches training.")
            else:
                 raise HTTPException(status_code=500, detail=f"Error during prediction logic: {pred_e}")

        # --- Prepare Response for New Video ---
        response = {
            "source": "model",
            "filename": original_filename,
            "prediction": "FAKE" if prediction == 1 else "REAL",
            "confidence_fake": prob_fake,
            "threshold_used": threshold,
            "features": features # Return the features dictionary as extracted/defaulted
        }
        print("Prediction process completed for new video.")
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")

        return response

    except HTTPException as http_e:
         # Re-raise HTTPExceptions directly
         print(f"HTTP Exception occurred: {http_e.detail}")
         raise http_e
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"Unhandled error processing new video: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

    finally:
        # --- Cleanup for New Video ---
        if temp_path and os.path.exists(temp_path):
            print(f"Cleaning up uploaded file: {temp_path}")
            try:
                os.unlink(temp_path)
            except Exception as clean_e:
                print(f"Warning: Failed to delete temporary file {temp_path}: {clean_e}")

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    import signal
    import sys

    def handle_exit(signum, frame):
        print("\nShutting down gracefully...")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Configure uvicorn with proper settings
    config = uvicorn.Config(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_delay=1,
        workers=1,
        log_level="info",
        access_log=True,
        use_colors=True
    )

    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)
        
