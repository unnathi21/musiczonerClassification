import librosa
import numpy as np

class AudioProcessor:
    def extract_features(self, audio_path):
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)

            # Extract features
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]  # Tempo (BPM)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)  # Chroma features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))  # Spectral centroid

            # Combine features into a vector
            features = np.concatenate([[tempo], chroma, [spectral_centroid]])
            return features
        except Exception as e:
            raise Exception(f"Error extracting audio features: {str(e)}")