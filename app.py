from flask import Flask, render_template, request, send_from_directory, send_file, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from music21 import converter, instrument, note, chord, stream, midi
import glob
import time
import os
import tensorflow.keras.utils as utils
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import load_model
from midi2voice import renderize_voice


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


# LYRICS GENERATION MAIN FUNCTION
@app.route("/generate_lyrics", methods=["POST"])
def generate_lyrics():
    if request.method=="POST":
        start_string = request.form["user_input"]
        t = float(
            request.form["slider_value"]
        )  # Assuming the slider value is a float from 0 to 1
        generated_lyrics = None
        text = (
            open("./models/lyrics/choruses_lyrics.txt", "rb")
            .read()
            .decode(encoding="utf-8")
        )
        # The unique characters in the file
        vocab = sorted(set(text))
        print("{} unique characters".format(len(vocab)))
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)

        model = tf.keras.models.load_model("./models/lyrics/model.h5")

        num_generate = 500

        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []

        # Low temperature results in more predictable text.
        # Higher temperature results in more surprising text.
        print(t)
        temperature = t

        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)

            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        generated_text = start_string + "".join(text_generated)

        # Save the generated text to a file
        with open("./models/lyrics/generated_lyrics/lyrics.txt", "w", encoding="utf-8") as output_file:
            output_file.write(generated_text)

        return jsonify({"status":True, "generated_lyrics": generated_text})
    else:
        return jsonify({"status":False, "error":"Something went wrong"})
    # return f"Generated lyrics for '{user_input}' with slider value {slider_value}"


# MUSIC GENERATION MAIN FUNCTION
@app.route("/generate_music", methods=["POST"])
def generate_music():
    if request.method=="POST":
        generated_music = None
        music_random_value = float(request.form["slider_value1"])
        temperature = float(request.form["slider_value2"])
        # Training Hyperparameters:
        VOCABULARY_SIZE = 130  # known 0-127 notes + 128 note_off + 129 no_event
        SEQ_LEN = 30
        BATCH_SIZE = 30
        HIDDEN_UNITS = 256
        EPOCHS = 35
        SEED = 2345
        np.random.seed(SEED)

        ## Load up some melodies I prepared earlier...
        with np.load(
            "./models/music/melody_training_dataset_2.npz", allow_pickle=True
        ) as data:
            train_set = data["train"]

        # Prepare training data as X and Y.
        # This slices the melodies into sequences of length SEQ_LEN+1.
        # Then, each sequence is split into an X of length SEQ_LEN and a y of length 1.

        # Slice the sequences:
        slices = []
        for seq in train_set:
            slices += slice_sequence_examples(seq, SEQ_LEN + 1)

        # Split the sequences into Xs and ys:
        X, y = seq_to_singleton_format(slices)
        # Convert into numpy arrays.
        X = np.array(X)
        y = np.array(y)

        # Look at the size of the training corpus:
        print("Total Training Corpus:")
        print("X:", X.shape)
        print("y:", y.shape)
        print()

        # Have a look at one example:
        print("Looking at one example:")
        print("X:", X[95])
        print("y:", y[95])
        # Note: Music data is sparser than text, there's lots of 129s (do nothing)
        # and few examples of any particular note on.
        # As a result, it's a bit harder to train a melody-rnn.

        # Build a decoding model (input length 1, batch size 1, stateful)
        model_dec = Sequential()
        model_dec.add(
            Embedding(
                VOCABULARY_SIZE, HIDDEN_UNITS, input_length=1, batch_input_shape=(1, 1)
            )
        )
        # LSTM part
        model_dec.add(LSTM(HIDDEN_UNITS, stateful=True, return_sequences=True))
        model_dec.add(LSTM(HIDDEN_UNITS, stateful=True))

        # project back to vocabulary
        model_dec.add(Dense(VOCABULARY_SIZE, activation="softmax"))
        model_dec.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        model_dec.summary()
        # set weights from training model
        # model_dec.set_weights(model_train.get_weights())
        model_dec.load_weights("./models/music/model-rnn4.h5")

        model_dec.reset_states()  # Start with LSTM state blank
        o = sample_model(music_random_value, model_dec, length=255, temperature=temperature)

        melody_stream = noteArrayToStream(o)  # turn into a music21 stream
        # melody_stream.show()  # show the score.

        sp = melody_stream.write("midi", "./models/music/generated_music/output.mid")

        if sp:
            return jsonify({"status":True})
        else:   
            return jsonify({"status":False, "error":"Something went wrong"})
    else:
        return jsonify({"status":False, "error":"Something went wrong"})


@app.route("/generate_vocals", methods=["POST"])
def generate_vocals():
    if request.method=="POST":
        # Get the file paths from the local directory
        text_file_path = "./models/lyrics/generated_lyrics/lyrics.txt"
        midi_file_path = "./models/music/generated_music/output.mid"

        # # Process the files using midi2voice
        with open(text_file_path, "r") as text:
            lyrics = text.readlines()

        renderize_voice(
            lyrics,
            midi_file_path,
            tempo=80,
            lang="english",
            gender="female",
            voiceindex=0,
            out_folder="./models/vocals/",
        )

        return jsonify({"status":True})

    else:
        return jsonify({"status":False, "error":"Something went wrong"})

# MUSIC GENERATION SUB-FUNCTIONS

# Melody-RNN Format is a sequence of 8-bit integers indicating the following:
# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128  # (stop playing all previous notes)
MELODY_NO_EVENT = 129  # (no change from previous event)
# Each element in the sequence lasts for one sixteenth note.
# This can encode monophonic music only.


# def streamToNoteArray(stream):
#     """
#     Convert a Music21 sequence to a numpy array of int8s into Melody-RNN format:
#         0-127 - note on at specified pitch
#         128   - note off
#         129   - no event
#     """
#     # Part one, extract from stream
#     total_length = int(np.round(stream.flat.highestTime / 0.25))  # in semiquavers
#     stream_list = []
#     for element in stream.flat:
#         if isinstance(element, note.Note):
#             stream_list.append(
#                 [
#                     np.round(element.offset / 0.25),
#                     np.round(element.quarterLength / 0.25),
#                     element.pitch.midi,
#                 ]
#             )
#         elif isinstance(element, chord.Chord):
#             stream_list.append(
#                 [
#                     np.round(element.offset / 0.25),
#                     np.round(element.quarterLength / 0.25),
#                     element.sortAscending().pitches[-1].midi,
#                 ]
#             )
#     np_stream_list = np.array(stream_list, dtype=int)
#     df = pd.DataFrame(
#         {
#             "pos": np_stream_list.T[0],
#             "dur": np_stream_list.T[1],
#             "pitch": np_stream_list.T[2],
#         }
#     )
#     df = df.sort_values(
#         ["pos", "pitch"], ascending=[True, False]
#     )  # sort the dataframe properly
#     df = df.drop_duplicates(subset=["pos"])  # drop duplicate values
#     # part 2, convert into a sequence of note events
#     output = np.zeros(total_length + 1, dtype=np.int16) + np.int16(
#         MELODY_NO_EVENT
#     )  # set array full of no events by default.
#     # Fill in the output list
#     for i in range(total_length):
#         if not df[df.pos == i].empty:
#             n = df[df.pos == i].iloc[0]  # pick the highest pitch at each semiquaver
#             output[i] = n.pitch  # set note on
#             output[i + n.dur] = MELODY_NOTE_OFF
#     return output


def noteArrayToDataFrame(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a dataframe.
    """
    df = pd.DataFrame({"code": note_array})
    df["offset"] = df.index
    df["duration"] = df.index
    df = df[df.code != MELODY_NO_EVENT]
    df.duration = (
        df.duration.diff(-1) * -1 * 0.25
    )  # calculate durati****ons and change to quarter note fractions
    df = df.fillna(0.25)
    return df[["code", "duration"]]


def noteArrayToStream(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a music21 stream.
    """
    df = noteArrayToDataFrame(note_array)
    print(df)
    melody_stream = stream.Stream()
    for index, row in df.iterrows():
        if row.code == MELODY_NO_EVENT:
            new_note = (
                note.Rest()
            )  # bit of an oversimplification, doesn't produce long notes.
        elif row.code == MELODY_NOTE_OFF:
            new_note = note.Rest()
        else:
            new_note = note.Note(row.code)
        new_note.quarterLength = row.duration
        melody_stream.append(new_note)
    return melody_stream


def slice_sequence_examples(sequence, num_steps):
    """Slice a sequence into redundant sequences of length num_steps."""
    xs = []
    for i in range(len(sequence) - num_steps - 1):
        example = sequence[i : i + num_steps]
        xs.append(example)
    return xs


def seq_to_singleton_format(examples):
    """
    Return the examples in seq to singleton format.
    """
    xs = []
    ys = []
    for ex in examples:
        xs.append(ex[:-1])
        ys.append(ex[-1])
    return (xs, ys)


def sample(preds, temperature=1.0):
    """helper function to sample an index from a probability array"""
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


## Sampling function


def sample_model(seed, model_name, length=40, temperature=1.0):
    """Samples a musicRNN given a seed sequence."""
    generated = []
    generated.append(seed)
    next_index = seed
    for i in range(length):
        x = np.array([next_index])
        x = np.reshape(x, (1, 1))
        preds = model_name.predict(x, verbose=0)[0]
        next_index = sample(preds, temperature)
        generated.append(next_index)
    return np.array(generated)


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory("models/", filename)


@app.route("/download/<path:filename>")
def download_audio(filename):
    return send_from_directory(
        "models/", filename, as_attachment=True
    )

if __name__ == "__main__":
    app.run(debug=True)
