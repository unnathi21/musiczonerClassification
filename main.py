import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from audio_processor import AudioProcessor
from lyrics_processor import LyricsProcessor
from genre_classifier import GenreClassifier
from image_processor import ImageProcessor

def plot_results(genre, sentiment, cover_genre):
    fig, ax = plt.subplots()
    categories = ['Audio Genre', 'Lyrics Sentiment', 'Cover Genre']
    values = [genre, sentiment, cover_genre or 'N/A']
    ax.bar(categories, [1, 1, 1], color=['blue', 'green', 'red'])
    for i, v in enumerate(values):
        ax.text(i, 0.5, v, ha='center', va='center', color='white')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Results')
    st.pyplot(fig)

def main():
    st.title("Music Genre Classifier")
    st.write("Upload a song, lyrics, and optional album cover to classify genre and analyze sentiment.")

    # Initialize components
    audio_processor = AudioProcessor()
    lyrics_processor = LyricsProcessor()
    genre_classifier = GenreClassifier()
    image_processor = ImageProcessor()

    # File uploaders
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    lyrics_file = st.file_uploader("Upload lyrics (text file)", type=["txt"])
    cover_file = st.file_uploader("Upload album cover (optional)", type=["jpg", "jpeg", "png"])

    if audio_file and lyrics_file:
        # Save uploaded files
        audio_path = "temp_audio.mp3"
        lyrics_path = "temp_lyrics.txt"
        cover_path = "temp_cover.jpg" if cover_file else None

        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        with open(lyrics_path, "wb") as f:
            f.write(lyrics_file.getbuffer())
        if cover_file:
            with open(cover_path, "wb") as f:
                f.write(cover_file.getbuffer())

        # Process audio
        try:
            audio_features = audio_processor.extract_features(audio_path)
            genre = genre_classifier.predict_genre(audio_features)
        except Exception as e:
            st.error(str(e))
            return

        # Process lyrics
        try:
            with open(lyrics_path, "r", encoding="utf-8") as f:
                lyrics = f.read()
            sentiment, _ = lyrics_processor.analyze_sentiment(lyrics)
        except Exception as e:
            st.error(str(e))
            return

        # Process album cover (optional)
        cover_genre = None
        if cover_path:
            try:
                cover_genre = image_processor.classify_cover(cover_path)
            except Exception as e:
                st.error(str(e))
                return

        # Display results
        st.success(f"Song Genre: {genre}")
        st.success(f"Lyrics Sentiment: {sentiment}")
        if cover_genre:
            st.success(f"Album Cover Genre: {cover_genre}")

        # Visualize
        plot_results(genre, sentiment, cover_genre)

if __name__ == "__main__":
    main()