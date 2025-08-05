import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, GlobalAveragePooling2D, Reshape, Dense, Input, Dropout, Flatten
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import os

class BurnClassifier:
    def __init__(self, model_path=None):
        self.model = None
        self.class_names = ['First Degree Burn', 'Second Degree Burn', 'Third Degree Burn']
        self.img_size = (224, 224)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self):
        """Build the CNN-Transformer hybrid model"""
        inputs = Input(shape=(224, 224, 3))

        # CNN backbone
        resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        resnet.trainable = False
        x = resnet(inputs)

        # Convert CNN features to 1D
        x = GlobalAveragePooling2D()(x)         # (batch, 2048)
        x = Reshape((1, -1))(x)                 # (batch, 1, 2048)

        # Transformer block (self-attention)
        x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = LayerNormalization()(x)
        x = Flatten()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(3, activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        print("Model built successfully!")

    def load_model(self, model_path):
        """Load a pre-trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Building new model instead...")
            self.build_model()

    def save_model(self, model_path):
        """Save the trained model"""
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")

    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        try:
            # Load and preprocess image
            img = Image.open(image_path)
            img = img.convert('RGB')  # Ensure RGB format
            img = img.resize(self.img_size)  # Resize to model input size

            # Convert to numpy array and normalize
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0  # Normalize to [0,1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def predict(self, image_path):
        """Make prediction on an image"""
        if not self.model:
            return None, "Model not loaded"

        # Preprocess image
        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return None, "Error preprocessing image"

        try:
            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

            result = {
                'predicted_class': self.class_names[predicted_class],
                'class_id': int(predicted_class),
                'confidence': confidence,
                'all_probabilities': {
                    self.class_names[i]: float(predictions[0][i])
                    for i in range(len(self.class_names))
                }
            }

            return result, None
        except Exception as e:
            return None, f"Error during prediction: {e}"

def get_model_instance():
    """Get a singleton instance of the burn classifier"""
    if not hasattr(get_model_instance, 'instance'):
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'burn_classifier.h5')
        get_model_instance.instance = BurnClassifier(model_path)
    return get_model_instance.instance
