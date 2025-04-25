import os
import cv2
import json
import numpy as np
import pandas as pd
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
from pathlib import Path
import tempfile
import time
import subprocess
import librosa
from moviepy import VideoFileClip
from tqdm import tqdm

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(BASE_DIR, "train_sample_videos")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def extract_audio(video_path):
    """Extract audio from video if available"""
    try:
        ffmpeg_path = str(Path(BASE_DIR) / 
                         'ffmpeg-7.1.1-essentials_build' / 
                         'bin' / 
                         'ffmpeg.exe')
        
        # First try to extract audio directly without checking
        temp_dir = tempfile.gettempdir()
        temp_name = f'audio_extract_{int(time.time())}.wav'
        temp_path = os.path.join(temp_dir, temp_name)
        
        # Extract audio with improved parameters
        command = f'"{ffmpeg_path}" -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{temp_path}" -y'
        
        process = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode == 0 and os.path.exists(temp_path):
            # Verify the extracted audio file
            if os.path.getsize(temp_path) > 0:
                print(f"Audio extracted successfully to {temp_path}")
                return temp_path
            else:
                print("Audio file was created but is empty")
                os.unlink(temp_path)
                return None
        
        print(f"Audio extraction failed with error: {process.stderr}")
        return None
            
    except Exception as e:
        print(f"Error in audio extraction: {str(e)}")
        return None

def extract_audio_features(audio_path):
    """Extract actual audio features using Librosa"""
    try:
        print(f"\nLoading audio file: {audio_path}")
        # Load the audio file with detailed parameters
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        print(f"Audio loaded - Duration: {len(y)/sr:.2f} seconds, Sample rate: {sr} Hz")
        
        # Verify audio data
        if len(y) == 0:
            print("Error: Empty audio data")
            return None
            
        print(f"Audio data range: [{np.min(y):.4f}, {np.max(y):.4f}]")
        
        # Extract MFCC features with detailed parameters
        print("\nExtracting MFCC features...")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc_mean = np.mean(mfccs, axis=1)
        print(f"MFCC means: {[f'{x:.2f}' for x in mfcc_mean]}")
        
        # Extract spectral centroid
        print("\nExtracting spectral centroid...")
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)
        spectral_centroid_mean = np.mean(spectral_centroid)
        print(f"Spectral centroid: {spectral_centroid_mean:.2f} Hz")
        
        # Extract zero-crossing rate
        print("\nExtracting zero-crossing rate...")
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        print(f"Zero-crossing rate: {zero_crossing_rate_mean:.4f}")
        
        # Extract chroma features
        print("\nExtracting chroma features...")
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
        chroma_mean = np.mean(chroma, axis=1)
        print(f"Chroma means: {[f'{x:.2f}' for x in chroma_mean]}")
        
        # Extract RMS energy
        print("\nExtracting RMS energy...")
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
        rms_mean = np.mean(rms)
        print(f"RMS energy: {rms_mean:.4f}")
        
        # Verify feature values
        if np.all(mfcc_mean == 0) or np.all(chroma_mean == 0):
            print("Warning: All feature values are zero!")
            return None
            
        return {
            'mfcc_mean': mfcc_mean.tolist(),
            'spectral_centroid_mean': spectral_centroid_mean,
            'zero_crossing_rate_mean': zero_crossing_rate_mean,
            'chroma_mean': chroma_mean.tolist(),
            'rms_mean': rms_mean
        }
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

def extract_nlp_features(video_path):
    """Extract NLP features from video using real audio analysis"""
    default_features = {
        'voice_consistency': 0.0,
        'speech_confidence': 0.0,
        'voice_stability': 0.0
    }
    
    try:
        print(f"\nAttempting to extract audio from {os.path.basename(video_path)}...")
        audio_path = extract_audio(video_path)
        
        if audio_path is None:
            print(f"No audio could be extracted from {os.path.basename(video_path)}")
            return default_features
        
        try:
            print("Extracting audio features using Librosa...")
            audio_features = extract_audio_features(audio_path)
            
            if audio_features is None:
                print(f"Failed to extract audio features for {os.path.basename(video_path)}")
                return default_features
            
            print("\nCalculating voice characteristics...")
            
            # Calculate voice consistency from MFCCs
            mfcc_values = np.array(audio_features['mfcc_mean'])
            print(f"Raw MFCC values: {[f'{x:.2f}' for x in mfcc_values]}")
            
            # Normalize MFCC values
            mfcc_normalized = (mfcc_values - np.mean(mfcc_values)) / np.std(mfcc_values)
            print(f"Normalized MFCC values: {[f'{x:.2f}' for x in mfcc_normalized]}")
            
            # Calculate consistency as inverse of variance
            mfcc_variance = np.var(mfcc_normalized)
            voice_consistency = 1.0 / (1.0 + mfcc_variance)
            print(f"MFCC variance: {mfcc_variance:.4f}")
            print(f"Voice consistency: {voice_consistency:.4f}")
            
            # Calculate speech confidence
            spectral_centroid = audio_features['spectral_centroid_mean']
            zero_crossing = audio_features['zero_crossing_rate_mean']
            
            normalized_centroid = min(max((spectral_centroid - 1000) / 3000, 0), 1)
            normalized_zcr = 1.0 - min(zero_crossing / 0.2, 1.0)
            speech_confidence = (normalized_centroid + normalized_zcr) / 2.0
            
            print(f"Spectral centroid: {spectral_centroid:.2f} Hz")
            print(f"Zero-crossing rate: {zero_crossing:.4f}")
            print(f"Speech confidence: {speech_confidence:.4f}")
            
            # Calculate voice stability
            chroma_values = np.array(audio_features['chroma_mean'])
            chroma_peaks = np.argmax(chroma_values, axis=0)
            pitch_stability = 1.0 - (np.std(chroma_peaks) / 12.0)
            voice_stability = max(0, min(1, pitch_stability))
            
            print(f"Chroma peaks std: {np.std(chroma_peaks):.2f}")
            print(f"Voice stability: {voice_stability:.4f}")
            
            nlp_features = {
                'voice_consistency': float(voice_consistency),
                'speech_confidence': float(speech_confidence),
                'voice_stability': float(voice_stability)
            }
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                print("Temporary audio file cleaned up")
            
            return nlp_features
            
        except Exception as e:
            print(f"Error in audio processing: {str(e)}")
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
            return default_features
            
    except Exception as e:
        print(f"Error in NLP feature extraction: {str(e)}")
        return default_features

def extract_features(frame):
    """Enhanced feature extraction with temporal analysis"""
    frame = cv2.resize(frame, (224, 224))
    
    # 1. Color Analysis
    mean_R = np.mean(frame[..., 2])
    mean_G = np.mean(frame[..., 1])
    mean_B = np.mean(frame[..., 0])
    
    var_R = np.var(frame[..., 2])
    var_G = np.var(frame[..., 1])
    var_B = np.var(frame[..., 0])
    
    # 2. Edge and Texture Analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # 3. Blur and Noise Analysis
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise_kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    noise_level = np.sum(np.abs(cv2.filter2D(gray, -1, noise_kernel)))
    
    # 4. Texture Analysis with LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    texture_entropy = entropy(lbp_hist)
    
    # 5. Motion Analysis (if previous frame is available)
    motion_features = {
        'motion_consistency': 0.0,  # Will be updated in process_video
        'motion_smoothness': 0.0    # Will be updated in process_video
    }
    
    return {
        'mean_R': mean_R,
        'mean_G': mean_G,
        'mean_B': mean_B,
        'var_R': var_R,
        'var_G': var_G,
        'var_B': var_B,
        'edge_density': edge_density,
        'laplacian_var': laplacian_var,
        'noise_level': noise_level,
        'texture_entropy': texture_entropy,
        **motion_features
    }

def process_video(video_path):
    """Process a single video file with enhanced feature extraction"""
    try:
        # Extract visual features
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None
            
        frame_features = []
        prev_frame = None
        motion_consistencies = []
        motion_smoothnesses = []
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly throughout the video
        sample_indices = np.linspace(0, total_frames-1, min(30, total_frames), dtype=int)
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx in sample_indices:
                # Calculate motion features if previous frame exists
                if prev_frame is not None:
                    # Calculate optical flow
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    
                    # Calculate motion consistency
                    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    motion_consistency = 1.0 - (np.std(magnitude) / np.mean(magnitude))
                    motion_consistencies.append(motion_consistency)
                    
                    # Calculate motion smoothness
                    motion_smoothness = 1.0 - (np.mean(np.abs(np.gradient(magnitude))) / np.mean(magnitude))
                    motion_smoothnesses.append(motion_smoothness)
                
                features = extract_features(frame)
                if prev_frame is not None:
                    features['motion_consistency'] = motion_consistencies[-1]
                    features['motion_smoothness'] = motion_smoothnesses[-1]
                
                frame_features.append(features)
                prev_frame = frame.copy()
                frame_count += 1
        
        cap.release()
        
        if not frame_features:
            print(f"No frames could be extracted from: {video_path}")
            return None
            
        # Calculate mean and temporal features
        mean_features = {}
        temporal_features = {}
        
        # Calculate mean features
        for key in frame_features[0].keys():
            values = [f[key] for f in frame_features]
            mean_features[key] = np.mean(values)
            
            # Add temporal features
            if key not in ['motion_consistency', 'motion_smoothness']:
                temporal_features[f'{key}_std'] = np.std(values)
                temporal_features[f'{key}_range'] = np.max(values) - np.min(values)
        
        # Add motion features
        if motion_consistencies:
            mean_features['motion_consistency'] = np.mean(motion_consistencies)
            mean_features['motion_smoothness'] = np.mean(motion_smoothnesses)
            temporal_features['motion_consistency_std'] = np.std(motion_consistencies)
            temporal_features['motion_smoothness_std'] = np.std(motion_smoothnesses)
        
        # Combine all features
        all_features = {**mean_features, **temporal_features}
        
        # Extract audio features
        try:
            video = VideoFileClip(video_path)
            temp_audio_path = os.path.join(MODEL_DIR, "temp_audio.wav")
            video.audio.write_audiofile(temp_audio_path)
            
            audio_features = extract_audio_features(temp_audio_path)
            
            # Clean up
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
            if audio_features:
                # Convert audio features to the format expected by the model
                for i in range(13):
                    all_features[f'mfcc_{i}'] = audio_features['mfcc_mean'][i]
                all_features['spectral_centroid_mean'] = audio_features['spectral_centroid_mean']
                all_features['zero_crossing_rate_mean'] = audio_features['zero_crossing_rate_mean']
                for i in range(12):
                    all_features[f'chroma_{i}'] = audio_features['chroma_mean'][i]
                all_features['rms_mean'] = audio_features['rms_mean']
            else:
                # Use default values if audio extraction fails
                for i in range(13):
                    all_features[f'mfcc_{i}'] = 0
                all_features['spectral_centroid_mean'] = 0
                all_features['zero_crossing_rate_mean'] = 0
                for i in range(12):
                    all_features[f'chroma_{i}'] = 0
                all_features['rms_mean'] = 0
                
        except Exception as e:
            print(f"Error processing audio for {video_path}: {e}")
            # Use default values
            for i in range(13):
                all_features[f'mfcc_{i}'] = 0
            all_features['spectral_centroid_mean'] = 0
            all_features['zero_crossing_rate_mean'] = 0
            for i in range(12):
                all_features[f'chroma_{i}'] = 0
            all_features['rms_mean'] = 0
            
        return all_features
        
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None

def preprocess_dataset():
    """Preprocess all videos in the dataset"""
    print("Starting preprocessing of all videos...")
    
    # Create output directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load metadata
    metadata_path = os.path.join(VIDEO_DIR, "metadata.json")
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print("Successfully loaded metadata.json")
    except Exception as e:
        print(f"Error loading metadata.json: {e}")
        return
    
    # Initialize lists to store features and labels
    all_features = []
    filenames = []
    labels = []
    
    # Get list of video files
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    print(f"\nFound {len(video_files)} video files to process")
    
    # Process each video with progress bar
    for filename in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(VIDEO_DIR, filename)
        
        # Get label from metadata
        if filename in metadata:
            label = metadata[filename]["label"]
        else:
            print(f"\nWarning: No metadata found for {filename}, skipping...")
            continue
        
        # Process video
        features = process_video(video_path)
        if features is not None:
            all_features.append(features)
            filenames.append(filename)
            labels.append(label)
        else:
            print(f"\nFailed to process {filename}")
            
    if not all_features:
        print("\nNo videos were processed successfully!")
        return
        
    # Create DataFrame
    df = pd.DataFrame(all_features)
    df['filename'] = filenames
    df['label'] = labels
    
    # Print label distribution
    print("\nLabel distribution:")
    label_counts = df['label'].value_counts()
    print(label_counts)
    print("\nVideos per label:")
    print(df.groupby('label')['filename'].apply(list))
    
    # Feature selection
    print("\nPerforming feature selection...")
    
    # Calculate correlation matrix
    corr_matrix = df.drop(['filename', 'label'], axis=1).corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    # Drop highly correlated features
    df = df.drop(to_drop, axis=1)
    print(f"Dropped {len(to_drop)} highly correlated features")
    
    # Save to CSV
    output_path = os.path.join(MODEL_DIR, "training_data.csv")
    df.to_csv(output_path, index=False)
    print(f"\nData saved to: {output_path}")
    
    # Save feature statistics
    feature_stats = {}
    for column in df.columns:
        if column not in ['filename', 'label']:
            feature_stats[column] = {
                'mean': float(df[column].mean()),
                'std': float(df[column].std()),
                'min': float(df[column].min()),
                'max': float(df[column].max())
            }
            
    stats_path = os.path.join(MODEL_DIR, "feature_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(feature_stats, f, indent=4)
    print(f"Feature statistics saved to: {stats_path}")
    
    print("\nPreprocessing completed!")
    print(f"Total videos processed: {len(df)}")
    print(f"Fake videos: {sum(df['label'] == 'FAKE')}")
    print(f"Real videos: {sum(df['label'] == 'REAL')}")
    print(f"Total features after selection: {len(df.columns) - 2}")  # Excluding filename and label
    
    # Print sample of the data
    print("\nSample of processed data:")
    print(df[['filename', 'label']].head())

if __name__ == "__main__":
    preprocess_dataset()