# Chord recognition 
A boilerplate structure for training a model on guitarchords. 
note: it is required you have your own dataset, (there are a few available on www.kaggle.com you can download). 

## Example
The repository comes with a default model that is trained on a set of major-chords from Amajor to G#major. The dataset was recorded with a nylon-string guitar
I would therefor recommend that (if you would like to use my model) use a nylonguitar. 

Record a guitarchord and place the audiofile in the project directory, and write the filepath here (at the end of main.py): 
```
chord = identifier.predict_new_chord('Amajor.wav', '44100')
```
 
Then run main.py. It will print the chord and the triad notes within the chord. 

![Skärmbild 2024-01-11 144736](https://github.com/j0nt392/chord_recognition_model/assets/25915810/5985f603-00a9-4410-9f0a-f6c838fdcec9)

# Training the model
This model does not come with a dataset, it requires you to use your own. The recommended audiosamples you need:
- One strum
- No longer than 1-3 seconds
- .wav format (or any other that works with Librosa)

## Feature extraction/Creating a dataset
Create a folder in the project directory with the following structure (change the name of the subfolders if you'd like to train the model
on other chords). Each of the subfolders should contain one-shot samples of chords. 

![Skärmbild 2024-01-11 152828](https://github.com/j0nt392/chord_recognition_model/assets/25915810/4648a0d8-7b58-4f43-a419-b79281a50b46)

Once this is done, go to utils.py and connect to your postgresql server, and run utils.py. This should extract features from your chords, and
pair them with the name of the folder they were in and store it all in SQL. 

The way this works is as follows:

### hpss

```
harmonic, percussive = librosa.effects.hpss(audio)
```
The hpss removes any percussive elements from the recording, and extracts the harmonics. This is useful to filter out loud peaks and noises, which 
is crucial since we later on are going to use a CQT transform to determine the chord.

### CQT-transform
```
C = librosa.cqt(y=harmonic, sr=fs, fmin=librosa.note_to_hz('C1'), hop_length=256, n_bins=36)
```
The CQT transform sorts the audiosignal into frequency-bins (Y-axis) over time. Now we have information on which notes had the highest 
magnitude in the sample being analyzed. 

![Skärmbild 2024-01-11 155224](https://github.com/j0nt392/chord_recognition_model/assets/25915810/5cb4cd51-bc8c-426e-bdee-cebb62dfe068)

### Final pitch_sum
```
pitch_sum = np.abs(C).sum(axis=1)
```
We finally get the pitch_sum by aggregating the bins and adding up the sums of each over time. I.e, which notes had the highest magnitude for 
the longest period of time. 

## Evaluation
After each training-set you will see a confusion-matrix for evaluation. Check the actual VS predicted chords, and use this matrix to see where
your model went wrong and consider if you need to optimize it. 

![Skärmbild 2023-11-24 001158](https://github.com/j0nt392/chord_recognition_model/assets/25915810/c3945183-3c9f-460a-8d49-c1d7c32bf95e)

