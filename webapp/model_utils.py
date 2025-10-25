import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

class BurnClassifier:
    def __init__(self, model_path=None):
        self.model = None
        self.class_names = ['First Degree Burn', 'Second Degree Burn', 'Third Degree Burn']
        self.img_size = (224, 224)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            raise ValueError(f"Model path is required and must exist: {model_path}")

    def load_model(self, model_path):
        """Load a pre-trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise Exception(f"Failed to load model from {model_path}: {e}")

    def save_model(self, model_path):
        """Save the trained model"""
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")

    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        print(f"DEBUG: Preprocessing image: {image_path}")

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

    def gradcam(self, image_path, last_conv_layer_name='conv5_block3_out', pred_index=None, save_path=None):
        """Generate Grad-CAM heatmap and overlay for the given image."""
        print(f"DEBUG: Generating Grad-CAM for image: {image_path}")
        img_array = self.preprocess_image(image_path)
        if img_array is None:
            print("Error preprocessing image for GRADCAM")
            return None, None

        model = self.model
        last_conv_layer = model.get_layer(last_conv_layer_name)

        grad_model = tf.keras.models.Model(
            [model.inputs], [last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        img = Image.open(image_path).convert('RGB').resize(self.img_size)
        img = np.array(img)

        import cv2
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

        if save_path:
            cv2.imwrite(save_path, overlay)
            print(f"DEBUG: Grad-CAM overlay saved to {save_path}")

        print(f"DEBUG: Grad-CAM overlay shape: {overlay.shape}")
        print(f"DEBUG: Grad-CAM overlay dtype: {overlay.dtype}")
        return heatmap, overlay

    def predict(self, image_path, with_gradcam=False):
        """Make prediction on an image. Optionally return Grad-CAM overlay."""
        print(f"DEBUG: Making prediction for image: {image_path}")

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

            if with_gradcam:
                print("DEBUG: Generating Grad-CAM overlay...")
                _, overlay = self.gradcam(image_path, pred_index=predicted_class)
                if overlay is not None:
                    result['gradcam_overlay'] = overlay  # numpy array
                    print("DEBUG: Grad-CAM overlay added to result")
                else:
                    print("DEBUG: Grad-CAM overlay generation failed")

            return result, None
        except Exception as e:
            return None, f"Error during prediction: {e}"


def get_model_instance():
    """Get a singleton instance of the burn classifier"""
    if not hasattr(get_model_instance, 'instance'):

        model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'PROPESI', 'hybrid_resnet_transformer_finetunedtransformerlayers.h5')
        print(f"Initializing model with file: {model_path}")
        get_model_instance.instance = BurnClassifier(model_path)
    return get_model_instance.instance
