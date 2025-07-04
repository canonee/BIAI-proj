import sys
import os
# Wyłączone nadmiarowe logi TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from main import DataProvider, HandwritingRecognitionModel, CTCLayer

def predict_image(image_path,
                  model_path: str = "handwriting_model.h5",
                  dataset_path: str = None):
    """
    Wczytuje wytrenowany model i przewiduje napis na pojedynczym obrazie.
    Wynik jest drukowany w terminalu.
    """
    if dataset_path is None:
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "iam_words"))
    
    # Sprawdź, czy plik modelu istnieje
    if not os.path.exists(model_path):
        print(f"Nie znaleziono pliku modelu: {model_path}")
        return

    dp = DataProvider(dataset_path)
    dp.load_words_data()
    model_wrapper = HandwritingRecognitionModel(img_width=128, img_height=32, max_length=32)
    model_wrapper.build_vocab(dp.vocab)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'CTCLayer': CTCLayer},
        compile=False
    )

    # Model do predykcji (zwraca warstwę softmax bez straty CTC)
    image_input = model.inputs[0]
    dense2_output = model.get_layer(name="dense2").output
    pred_model = tf.keras.models.Model(inputs=[image_input], outputs=dense2_output)

    # Przetwórz obraz
    img = dp.preprocess_image(image_path, img_width=128, img_height=32)
    if img is None:
        print(f"Nie można przetworzyć obrazu: {image_path}")
        return
    img = np.expand_dims(img, axis=(0, -1))

    # Predykcja i dekodowanie
    preds = pred_model.predict(img)
    text = model_wrapper.decode_predictions(preds)[0]
    print(f"Predykcja dla '{image_path}': {text}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("wpisz: python testModel.py [image_path]")
    else:
        predict_image(*sys.argv[1:])
