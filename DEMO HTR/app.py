from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Cargar el modelo
loaded_model = tf.keras.models.load_model('model.keras')

#capas de proceso
import numpy as np
import tensorflow as tf

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

characters = ['!', '"', '#', "'", '(', ')', '*', ',', '-', '.', '/',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Mapping characters to integers.
char_to_num = tf.keras.layers.StringLookup(
    vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

class TextRecognitionService():
    def __init__(self, model):
        self.image_size = (128, 32)
        self.max_len = 19
        self.image_width = 128
        self.image_height = 32
        self.model = model

    # A utility function to decode the output of the network.

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search.
        results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :self.max_len
        ]

        # Iterate over the results and get back the text.
        output_text = []
        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(
                num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def distortion_free_resize(self, image, img_size):
        w, h = img_size
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

        # Check tha amount of padding needed to be done.
        pad_height = h - tf.shape(image)[0]
        pad_width = w - tf.shape(image)[1]

        # Only necessary if you want to do same amount of padding on both sides.
        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top = height + 1
            pad_height_bottom = height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left = width + 1
            pad_width_right = width
        else:
            pad_width_left = pad_width_right = pad_width // 2

        image = tf.pad(
            image,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
        )

        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.image.flip_left_right(image)
        return image

    def preprocess_image(self, image_file, img_size=(128, 32)):
        # Convertir el archivo en una imagen
        image = Image.open(image_file).convert('L')
        image = np.array(image)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=-1)  # Añadir el canal de color
        image = self.distortion_free_resize(image, img_size)
        image = np.expand_dims(image, axis=0)   # Añadir la dimensión del lote
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def predict(self, image_file):
        image = self.preprocess_image(image_file)
        predictions = self.model.predict(image)
        pred_texts = self.decode_batch_predictions(predictions)
        return pred_texts


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file:
        service = TextRecognitionService(loaded_model)
        predicted_text = service.predict(file)
        return jsonify({'prediction': predicted_text})
    else:
        return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    app.run(debug=True)
