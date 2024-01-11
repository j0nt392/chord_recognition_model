import joblib
from dotenv import load_dotenv
import librosa
import numpy as np

class Chord_classifier():
    '''Uses the chord_identifier.pkl model to classify any chord.'''
    def __init__(self):
        self.model = joblib.load('chord_identifier.pkl')
        self.label_encoder = joblib.load('label_encoder.pkl')
    
    def get_notes_for_chord(self, chord):
        '''takes a chord (C#) and gives you the triad notes in that chord'''
        chord_notes_mapping = {
            'Ab': ['Ab', 'Eb', 'C'],
            'A': ['A', 'Db', 'E'],
            'Am': ['A', 'C', 'E'],
            'B': ['B', 'Gb', 'Eb'],
            'Bb': ['Bb', 'D', 'F'],
            'Bdim': ['B', 'D', 'F'],
            'C': ['C', 'E', 'G'],
            'Cm': ['C', 'Eb', 'G'],
            'Db': ['Db', 'Ab', 'F'],
            'Dbm': ['Db', 'E', 'Ab'],
            'D': ['D', 'A', 'Gb'],
            'Dm': ['D', 'F', 'A'],
            'Eb': ['Eb', 'Bb', 'G'],
            'E': ['E', 'B', 'Ab'],
            'Em': ['E', 'G', 'B'],
            'F': ['F', 'A', 'C'],
            'Fm': ['F', 'Ab', 'C'],
            'Gb': ['Gb', 'Db', 'Bb'],
            'G': ['G', 'B', 'D'],
            'Gm': ['G', 'Bb', 'D'],
            'Bbm': ['Bb', 'Db', 'F'],
            'Bm': ['B', 'D', 'Gb'],
            'Gbm': ['Gb', 'Bb', 'Db']
            # You can continue to add more chords and their notes as needed
        }

        if chord in chord_notes_mapping:
            # Get the list of notes corresponding to the chord
            chord_notes = chord_notes_mapping[chord]
            
            # Return the first three notes from the list (or fewer if there are less than three)
            return chord_notes[:3]
        else:
            # Handle the case when the chord is not in the dictionary
            return []

    def _extract_features(self, audio_file, fs):
        audio = None
        if type(audio_file) == str:
            audio, fs = librosa.load(audio_file, sr = None)
        else:
            audio = audio_file
        
        #preprocessing
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Compute the constant-Q transform (CQT)
        C = librosa.cqt(y=harmonic, sr=fs, fmin=librosa.note_to_hz('C1'), hop_length=256, n_bins=36)
        
        # Convert the complex CQT output into magnitude, which represents the energy at each CQT bin
        # Summing across the time axis gives us the aggregate energy for each pitch bin
        pitch_sum = np.abs(C).sum(axis=1)
        
        return pitch_sum

    def predict_new_chord(self, audio_file_path, fs):
        # Extract features from the new audio file
        feature_vector = self._extract_features(audio_file_path, fs)
        # # Reshape the feature vector to match the model's input shape
        feature_vector = feature_vector.reshape(1, -1)
        try:
            predicted_label = self.model.predict(feature_vector)
            predicted_chord = self.label_encoder.inverse_transform(predicted_label)
            return predicted_chord[0]       
        except Exception as e:
            return "Error during prediction: %s", str(e)
        
identifier = Chord_classifier()

chord = identifier.predict_new_chord('Amajor.wav', '44100')
notes = identifier.get_notes_for_chord(chord)
print(f"chord predicted: {chord} \n triad notes: {notes}")