import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import joblib
import librosa
from moviepy import VideoFileClip
import speech_recognition as sr
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')  # Add this to suppress warnings

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "backend", "model")

def extract_audio_features(audio_path):
    """Extract actual audio features using Librosa."""
    try:
        # Load the audio file using Librosa
        y, sr = librosa.load(audio_path, sr=None)

        # Extract features like MFCC, Spectral Centroid, Zero-Crossing Rate, etc.
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)

        # Spectral Centroid (measures the 'brightness' of the sound)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)

        # Zero-Crossing Rate (rate at which the signal changes sign)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)

        # Chroma feature (related to harmony and chords)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Root Mean Square Error (RMS) for loudness estimation
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)

        # Combining features into a dictionary
        audio_features = {
            'mfcc_mean': mfcc_mean.tolist(),
            'spectral_centroid_mean': spectral_centroid_mean,
            'zero_crossing_rate_mean': zero_crossing_rate_mean,
            'chroma_mean': chroma_mean.tolist(),
            'rms_mean': rms_mean
        }

        print("Audio features extracted successfully")
        return audio_features
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

def extract_nlp_features(audio_path):
    """Extract NLP features from audio"""
    try:
        # Initialize speech recognizer
        recognizer = sr.Recognizer()
        
        # Load audio file
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            
        # Convert speech to text
        text = recognizer.recognize_google(audio)
        
        # Perform sentiment analysis
        blob = TextBlob(text)
        
        # Extract NLP features
        nlp_features = {
            'text_sentiment': blob.sentiment.polarity,
            'text_subjectivity': blob.sentiment.subjectivity,
            'word_count': len(text.split()),
            'sentence_count': len(blob.sentences),
            'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text else 0
        }
        
        return nlp_features
    except Exception as e:
        print(f"Error in NLP feature extraction: {e}")
        return {
            'text_sentiment': 0,
            'text_subjectivity': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0
        }

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def train_models():
    print("Loading preprocessed training data...")
    
    data_path = os.path.join(MODEL_DIR, "training_data.csv")
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {len(df)} samples")
        print("\nClass distribution:")
        print(df['label'].value_counts())
    except Exception as e:
        print(f"Error loading training data: {e}")
        return
    
    # Separate features and labels
    feature_columns = [col for col in df.columns if col not in ['filename', 'label']]
    X = df[feature_columns]
    y = (df['label'] == 'FAKE').astype(int)
    
    # Print initial feature count
    print(f"\nInitial number of features: {len(feature_columns)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights more precisely
    n_samples = len(y_train)
    n_real = sum(y_train == 0)
    n_fake = sum(y_train == 1)
    
    # Adjust weight for minority class (real videos)
    scale_pos_weight = (n_fake / n_real) * 2  # Increase weight for real class
    
    print("\nClass balance:")
    print(f"Real videos: {n_real}")
    print(f"Fake videos: {n_fake}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Initial model for feature selection
    initial_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.01,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    
    # Perform feature selection
    print("\nPerforming feature selection...")
    selector = SelectFromModel(initial_model, prefit=False, threshold='median')
    selector.fit(X_train_scaled, y_train)
    
    # Get selected feature mask and names
    selected_features_mask = selector.get_support()
    selected_features = [f for f, selected in zip(feature_columns, selected_features_mask) if selected]
    
    print(f"Selected {len(selected_features)} features out of {len(feature_columns)}")
    
    # Update training data with selected features
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Define final model with optimized parameters
    final_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.01,
        min_child_weight=5,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    
    # Perform cross-validation
    print("\nPerforming cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(final_model, X_train_selected, y_train, cv=cv, scoring='f1')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train final model
    print("\nTraining final model...")
    final_model.fit(X_train_selected, y_train)
    
    # Get predictions and probabilities
    y_pred_proba = final_model.predict_proba(X_test_selected)
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nOptimal classification threshold: {optimal_threshold:.4f}")
    
    # Make predictions using optimal threshold
    y_pred = (y_pred_proba[:, 1] >= optimal_threshold).astype(int)
    
    # Print detailed performance metrics
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    
    # Calculate and print additional metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)  # True Negative Rate
    sensitivity = tp / (tp + fn)  # True Positive Rate
    
    print("\nDetailed Metrics:")
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
    
    # Get feature importance for selected features
    feature_importance = dict(zip(selected_features, 
                                final_model.feature_importances_))
    
    # Sort and print feature importance
    sorted_importance = dict(sorted(feature_importance.items(), 
                                  key=lambda x: x[1], reverse=True))
    
    print("\nTop 10 Most Important Features:")
    for feature, importance in list(sorted_importance.items())[:10]:
        print(f"{feature}: {importance:.4f}")
    
    # Prepare metadata
    metadata = {
        'feature_columns': selected_features,
        'feature_stats': {
            k: {
                'mean': float(scaler.mean_[i]),
                'std': float(scaler.scale_[i])
            }
            for i, k in enumerate(feature_columns) if k in selected_features
        },
        'feature_importance': {k: float(v) for k, v in feature_importance.items()},
        'model_performance': {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'threshold': float(optimal_threshold)
        },
        'class_distribution': {
            'FAKE': int(sum(y == 1)),
            'REAL': int(sum(y == 0))
        }
    }
    
    # Save model and metadata
    print("\nSaving model and metadata...")
    
    # Save the complete pipeline
    pipeline = {
        'scaler': scaler,
        'selector': selector,
        'model': final_model,
        'threshold': optimal_threshold
    }
    
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "model.joblib"))
    
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    train_models()
