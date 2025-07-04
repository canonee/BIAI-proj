import os
import random
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

class DataProvider:
    """Klasa do wczytywania i przetwarzania danych z IAM Handwriting Database"""
    
    def __init__(self, dataset_path: str, words_file: str = "words.txt"):
        self.dataset_path = dataset_path
        self.words_file = words_file
        self.vocab = set()
        self.max_len = 0
        
    def load_words_data(self):
        """Wczytuje dane z pliku words.txt"""
        words_data = []
        words_file_path = os.path.join(self.dataset_path, self.words_file)
        
        with open(words_file_path, 'r') as file:
            for line in file:
                if not line.startswith('#') and line.strip():
                    parts = line.strip().split(' ')
                    if len(parts) >= 9 and parts[1] == 'ok':
                        word_id = parts[0]
                        transcription = ' '.join(parts[8:])
                        
                        folder_parts = word_id.split('-')
                        folder1 = folder_parts[0]
                        folder2 = f"{folder_parts[0]}-{folder_parts[1]}"
                        image_path = os.path.join(
                            self.dataset_path, 'words', folder1, folder2, f"{word_id}.png"
                        )
                        
                        if os.path.exists(image_path):
                            words_data.append({
                                'word_id': word_id,
                                'image_path': image_path,
                                'transcription': transcription
                            })
                            for char in transcription:
                                self.vocab.add(char)
                            self.max_len = max(self.max_len, len(transcription))
        
        return words_data
    
    def preprocess_image(self, image_path: str, img_width: int = 128, img_height: int = 32):
        """Przetwarza obraz do standardowego formatu"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None

            h, w = image.shape
            ratio = img_width / w
            new_h = int(h * ratio)
            
            if new_h > img_height:
                ratio = img_height / h
                new_w = int(w * ratio)
                image = cv2.resize(image, (new_w, img_height))
            else:
                image = cv2.resize(image, (img_width, new_h))
            
            h, w = image.shape
            pad_h = img_height - h
            pad_w = img_width - w
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255)
            
            image = (255 - image) / 255.0
            return image.astype(np.float32)
        except Exception as e:
            print(f"Błąd przy przetwarzaniu obrazu {image_path}: {e}")
            return None

class CTCLayer(layers.Layer):
    """Custom CTC Layer dla TensorFlow"""
    
    def __init__(self, name=None, trainable=True, **kwargs):
        # Przyjmujemy i przekazujemy trainable oraz ewentualne inne argumenty
        super().__init__(name=name, trainable=trainable, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost


    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

class HandwritingRecognitionModel:
    """Model do rozpoznawania pisma ręcznego"""
    
    def __init__(self, img_width: int = 128, img_height: int = 32, max_length: int = 32):
        self.img_width = img_width
        self.img_height = img_height
        self.max_length = max_length
        self.char_to_num = None
        self.num_to_char = None
        self.model = None
        
    def build_vocab(self, vocab: set):
        """Buduje mapowanie znaków na liczby i odwrotnie"""
        chars = sorted(list(vocab))
        chars.insert(0, "[UNK]")
        chars.append("[BLANK]")
        
        self.char_to_num = {char: idx for idx, char in enumerate(chars)}
        self.num_to_char = {idx: char for idx, char in enumerate(chars)}
        return len(chars)
    
    def encode_text(self, text: str):
        """Koduje tekst na sekwencję liczb"""
        encoded = []
        for char in text:
            encoded.append(self.char_to_num.get(char, self.char_to_num["[UNK]"]))
        return encoded
    
    def decode_predictions(self, pred):
        """Dekoduje predykcje modelu na tekst"""
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        
        output_text = []
        for res in results:
            text = ''.join([self.num_to_char[int(num)] for num in res if int(num) != -1])
            text = text.replace("[BLANK]", "").replace("[UNK]", "")
            output_text.append(text)
        
        return output_text
    
    def build_model(self, num_classes: int):
        """
        WARIANT 2: Głębsza architektura z większymi filtrami CNN
        - Zwiększone filtry CNN z testu 3 (64, 128, 256, 256)
        - Większe LSTM z testu 2
        - Optymalizowany dropout
        """
        input_img = layers.Input(shape=(self.img_height, self.img_width, 1), name="image")
        labels = layers.Input(name="label", shape=(None,))
        
        # Większe filtry CNN
        x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(input_img)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.MaxPooling2D((2, 1))(x)
        
        x = layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        x = layers.MaxPooling2D((2, 1))(x)
        
        # Dostosowany reshape dla większych filtrów
        new_shape = (self.img_width // 4, (self.img_height // 16) * 256)  # 512 cech
        x = layers.Reshape(target_shape=new_shape)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        
        # Większe LSTM z najlepszego testu
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
        
        x = layers.Dense(num_classes, activation="softmax", name="dense2")(x)
        output = CTCLayer(name="ctc_loss")(labels, x)
        
        model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer_v2")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
        return model

class TestCallback(keras.callbacks.Callback):
    """Custom callback do testowania modelu co 5 epok, na test_data"""
    
    def __init__(self, test_data, data_provider, model_wrapper, test_frequency=5):
        super().__init__()
        self.test_data = test_data
        self.data_provider = data_provider
        self.model_wrapper = model_wrapper
        self.test_frequency = test_frequency
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.test_frequency == 0:
            print(f"\n--- Testowanie modelu po epoce {epoch + 1} (na TEST_DATA) ---")
            self.test_on_test_data()
    
    def test_on_test_data(self):
        """Testuje model na całym zbiorze test_data."""
        test_generator = DataGenerator(
            self.test_data,
            self.data_provider,
            self.model_wrapper,
            batch_size=32,
            shuffle=False
        )

        image_input   = self.model.inputs[0]
        dense2_output = self.model.get_layer(name="dense2").output
        prediction_model = keras.models.Model(inputs=[image_input], outputs=dense2_output)

        correct_predictions = 0
        total_predictions = 0

        for batch_idx in range(len(test_generator)):
            test_batch = test_generator[batch_idx]
            test_images = test_batch[0]["image"]

            preds = prediction_model.predict(test_images, verbose=0)
            pred_texts = self.model_wrapper.decode_predictions(preds)

            batch_start_idx = batch_idx * test_generator.batch_size
            for i, pred_text in enumerate(pred_texts):
                if batch_start_idx + i < len(self.test_data):
                    true_text = self.test_data[batch_start_idx + i]['transcription']
                    if pred_text.strip().lower() == true_text.strip().lower():
                        correct_predictions += 1
                    total_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Dokładność na TEST_DATA: {accuracy:.4f} ({correct_predictions}/{total_predictions})")

        print("\nPrzykłady predykcji na TEST_DATA (pierwsze 5):")
        example_subset = self.test_data[:5]
        example_generator = DataGenerator(
            example_subset,
            self.data_provider,
            self.model_wrapper,
            batch_size=len(example_subset),
            shuffle=False
        )
        example_images = example_generator[0][0]["image"]
        preds = prediction_model.predict(example_images, verbose=0)
        pred_texts = self.model_wrapper.decode_predictions(preds)

        for i in range(len(pred_texts)):
            true_text = example_subset[i]['transcription']
            pred_text = pred_texts[i]
            match = "✓" if pred_text.strip().lower() == true_text.strip().lower() else "✗"
            print(f"{match} Prawdziwy: '{true_text}' | Predykcja: '{pred_text}'")
        print("-" * 60)

class DataGenerator(keras.utils.Sequence):
    """Generator danych dla treningu/testowania."""
    
    def __init__(self, data, data_provider, model, batch_size=32,
                 img_width=128, img_height=32, max_length=32, shuffle=True):
        self.data = data
        self.data_provider = data_provider
        self.model = model
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.max_length = max_length
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.data[k] for k in indexes]
        return self.__data_generation(batch_data)
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, batch_data):
        images = np.zeros((self.batch_size, self.img_height, self.img_width, 1))
        blank = self.model.char_to_num["[BLANK]"]
        labels = np.ones((self.batch_size, self.max_length), dtype=np.int32) * blank

        for i, item in enumerate(batch_data):
            image = self.data_provider.preprocess_image(item['image_path'],
                                                        self.img_width,
                                                        self.img_height)
            if image is not None:
                images[i] = np.expand_dims(image, axis=-1)

                encoded_label = self.model.encode_text(item['transcription'])
                if len(encoded_label) <= self.max_length:
                    labels[i, :len(encoded_label)] = encoded_label

        return {"image": images, "label": labels}, np.zeros((self.batch_size,))

def main():
    """Główna funkcja treningu"""
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "DATA", "iam_words")
    dataset_path = os.path.abspath(dataset_path)

    
    print("Wczytywanie danych...")
    data_provider = DataProvider(dataset_path)
    words_data = data_provider.load_words_data()
    
    print(f"Wczytano {len(words_data)} przykładów")
    print(f"Rozmiar słownika: {len(data_provider.vocab)}")
    print(f"Maksymalna długość tekstu: {data_provider.max_len}")
    
    # 80% → train+val, 20% → test_data
    train_val_data, test_data = train_test_split(words_data, test_size=0.20, random_state=42)
    # 10% z train_val → val_data, reszta (~72%) → train_data
    train_data, val_data = train_test_split(train_val_data, test_size=0.10, random_state=42)
    
    print(f"Dane treningowe:     {len(train_data)}")
    print(f"Dane walidacyjne:    {len(val_data)}")
    print(f"Dane testowe (held-out): {len(test_data)}")
    
    model = HandwritingRecognitionModel(img_width=128, img_height=32, max_length=32)
    num_classes = model.build_vocab(data_provider.vocab)
    print(f"Liczba klas: {num_classes}")
    
    keras_model = model.build_model(num_classes)
    keras_model.summary()
    
    train_generator = DataGenerator(train_data, data_provider, model, batch_size=32)
    val_generator   = DataGenerator(val_data,   data_provider, model, batch_size=32)
    
    test_callback = TestCallback(test_data, data_provider, model, test_frequency=5)
    
    callbacks = [
        test_callback,
        keras.callbacks.ModelCheckpoint(
            "handwriting_model.h5", 
            save_best_only=True, 
            monitor='val_loss',
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=8,  
            min_lr=1e-8
        )
    ]
    

    print("Rozpoczynam trening...")
    history = keras_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=40,
        callbacks=callbacks,
        verbose=1
    )
    
    # Zapisz finalny model po 40 epokach
    keras_model.save("handwriting_recognition_final.h5")
    
    # Ostateczne testowanie na TEST_DATA po zakończeniu wszystkich epok
    print("Ostateczne testowanie na TEST_DATA...")
    test_generator = DataGenerator(test_data, data_provider, model, batch_size=32, shuffle=False)
    
    image_input   = keras_model.inputs[0]
    dense2_output = keras_model.get_layer(name="dense2").output
    prediction_model = keras.models.Model(inputs=[image_input], outputs=dense2_output)
    
    correct_predictions, total_predictions = 0, 0
    for batch_idx in range(len(test_generator)):
        test_batch = test_generator[batch_idx]
        test_images = test_batch[0]["image"]
        
        preds = prediction_model.predict(test_images, verbose=0)
        pred_texts = model.decode_predictions(preds)
        
        batch_start_idx = batch_idx * test_generator.batch_size
        for i, pred_text in enumerate(pred_texts):
            if batch_start_idx + i < len(test_data):
                true_text = test_data[batch_start_idx + i]['transcription']
                if pred_text.strip().lower() == true_text.strip().lower():
                    correct_predictions += 1
                total_predictions += 1
    
    final_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Ostateczna dokładność na TEST_DATA: {final_accuracy:.4f} ({correct_predictions}/{total_predictions})")
    
    # Przykładowe predykcje na TEST_DATA
    print("\nPrzykłady predykcji (ostatni test na TEST_DATA):")
    example_subset = test_data[:5]
    example_generator = DataGenerator(
        example_subset,
        data_provider,
        model,
        batch_size=len(example_subset),
        shuffle=False
    )
    example_images = example_generator[0][0]["image"]
    preds = prediction_model.predict(example_images, verbose=0)
    pred_texts = model.decode_predictions(preds)
    
    for i in range(len(pred_texts)):
        true_text = example_subset[i]['transcription']
        pred_text = pred_texts[i]
        match = "✓" if pred_text.strip().lower() == true_text.strip().lower() else "✗"
        print(f"{match} Prawdziwy: '{true_text}' | Predykcja: '{pred_text}'")
    print("-" * 60)

if __name__ == "__main__":
    main()