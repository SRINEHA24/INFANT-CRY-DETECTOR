"""
Final Optimized Inference Script for Baby Crying Sound Classification
Records 7 seconds of audio and classifies using the optimized model
"""

import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import pickle
import sounddevice as sd
import scipy.io.wavfile as wav
from datetime import datetime
import pyttsx3

# ==============================================================
# Load model + config
# ==============================================================

def load_config_and_model():
    """Load model configuration and trained model"""
    try:
        print("Loading configuration...")
        with open('/home/bvrith/Desktop/Trained_Models/model_config_fast.pkl', 'rb') as f:
            config = pickle.load(f)

        print("Loading trained model...")
        model = keras.models.load_model('/home/bvrith/Desktop/Trained_Models/baby_cry_fast_best.keras')
        print("Model loaded successfully!")

        print("Loading label encoder...")
        with open('/home/bvrith/Desktop/Trained_Models/label_encoder_fast.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("Label encoder loaded successfully!")

        return model, label_encoder, config

    except Exception as e:
        print(f"Error loading model/config: {e}")
        return None, None, None


# ==============================================================
# Audio recording
# ==============================================================

def record_audio(duration=7, sr=16000):
    """Record audio from microphone"""
    print(f"\n{'='*70}")
    print(f"Recording {duration} seconds of audio...")
    print("="*70)

    print("\n🎤 GET READY!")
    print("Recording will start in: 3... 2... 1...\n")

    import time
    time.sleep(1)

    print("🔴 Recording... Please play the baby cry sound.")

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')

    for i in range(duration, 0, -1):
        print(f"  {i} seconds remaining...", end="\r")
        time.sleep(1)

    sd.wait()
    print("\n✓ Recording complete!")

    audio = audio.flatten()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recorded_audio_{timestamp}.wav"
    wav.write(filename, sr, audio)

    print(f"Audio saved as: {filename}")
    return audio, filename


# ==============================================================
# Feature extraction (MFCC only — EXACTLY LIKE TRAINING)
# ==============================================================

def extract_features(audio, config):
    """Extract MFCC features exactly as done during training"""

    # Convert all config values to int
    sr = int(config["SAMPLE_RATE"])
    duration = int(config["DURATION"])
    n_mfcc = int(config["N_MFCC"])
    max_len = int(config["MAX_LEN"])

    # Default MFCC params used in training
    n_fft = 2048
    hop_length = 512

    target_len = int(sr * duration)

    # Fix audio length
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # Extract MFCC (ONLY MFCC – NO delta, NO delta2)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Pad or trim to match MAX_LEN (time frames)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc.shape[1] > max_len:
        mfcc = mfcc[:, :max_len]

    return mfcc


# ==============================================================
# Classification
# ==============================================================

def classify_audio(audio, model, label_encoder, config):
    print("\nExtracting features...")
    mfcc = extract_features(audio, config)

    # Model expects shape: (1, 13, MAX_LEN, 1)
    mfcc = np.expand_dims(mfcc, axis=-1)   # add channel
    mfcc = np.expand_dims(mfcc, axis=0)    # add batch

    print("Classifying...")
    predictions = model.predict(mfcc, verbose=0)

    idx = np.argmax(predictions[0])
    predicted_class = label_encoder.inverse_transform([idx])[0]
    confidence = predictions[0][idx] * 100

    return predicted_class, confidence, predictions[0]


# ==============================================================
# Text-to-speech
# ==============================================================
def TextToSpeech(text, voice_type="english_female"):
    engine = pyttsx3.init(driverName='espeak')

    engine.setProperty("rate", 160)
    engine.setProperty("volume", 0.9)

    # -----------------------------
    # Telugu + other voice options
    # -----------------------------
    voice_map = {
        "telugu": "te",        # Telugu default
        "telugu_male": "te+m1",
        "telugu_female": "te+f2",
        "english_female": "en+f3",
        "english_male": "en+m3"
    }

    selected_voice = voice_map.get(voice_type.lower(), "en+f3")

    engine.setProperty("voice", selected_voice)

    engine.say(text)
    engine.runAndWait()

    try:
        engine.stop()
    except:
        pass

    try:
        del engine._driver
    except:
        pass

    del engine



# ==============================================================
# MAIN PROGRAM
# ==============================================================

def main():
    print("="*70)
    print("BABY CRYING SOUND CLASSIFICATION - Real-time Inference")
    print("="*70)

    model, label_encoder, config = load_config_and_model()
    if model is None:
        return

    print(f"\nModel ready! Classes:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f" {i+1}. {cls}")

    audio, filename = record_audio(duration=7, sr=int(config["SAMPLE_RATE"]))

    predicted_class, confidence, all_probs = classify_audio(audio, model, label_encoder, config)

    print("\n" + "="*70)
    print("🎯 CLASSIFICATION RESULTS")
    print("="*70)
    print(f"Predicted Class : {predicted_class.upper()}")
    print(f"Confidence      : {confidence:.2f}%")

    print("\nAll Classes:")
    sorted_idx = np.argsort(all_probs)[::-1]
    for i in sorted_idx:
        prob = all_probs[i] * 100
        bar = "█" * int(prob / 2)
        print(f"{label_encoder.classes_[i]:15s}: {prob:6.2f}% {bar}")

    print("="*70)

    if confidence > 20:
        msg = f"      Baby is feeling, {predicted_class} "
    else:
        msg = "Prediction uncertain. Please record again."

    TextToSpeech(msg)
    
    
                                                                                                                  

if __name__ == "__main__":
    main()
 
