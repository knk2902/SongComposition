from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    generated_lyrics = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        slider_value = float(
            request.form["slider_value"]
        )  # Assuming the slider value is a float from 0 to 1
        generated_lyrics = generate_lyrics(user_input, slider_value)
    return render_template("index.html", generated_lyrics=generated_lyrics)


def generate_lyrics(start_string, t):
    text = open("choruses_lyrics.txt", "rb").read().decode(encoding="utf-8")
    # The unique characters in the file
    vocab = sorted(set(text))
    print('{} unique characters'.format(len(vocab)))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    model = tf.keras.models.load_model("./model.h5")
    
    num_generate = 1000

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    temperature = t

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])
        text_generated.append("\n")

    return (start_string + ''.join(text_generated))
    return f"Generated lyrics for '{user_input}' with slider value {slider_value}"


if __name__ == "__main__":
    app.run(debug=True)
