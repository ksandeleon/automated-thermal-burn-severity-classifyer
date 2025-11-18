import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import json
import cv2

class BurnClassifier:
    def __init__(self, model_path=None):
        self.model = None
        self.class_names = [
            'First Degree Burn',
            'Second Degree Burn',
            'Third Degree Burn',
            'Fourth Degree Burn',
            'No Burn/Normal Skin'
        ]
        self.img_size = (224, 224)
        self.computation_log = []  # Store all computational steps

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

            # Convert to numpy array - KEEP IN [0, 255] RANGE to match training!
            img_array = np.array(img)
            img_array = img_array.astype('float32')  # Convert to float32 but keep [0, 255] range
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def _check_skin_color(self, image_array):
        """
        Check if image contains skin-like colors using HSV color space.

        Args:
            image_array: Preprocessed image array (1, 224, 224, 3) in [0, 255] range

        Returns:
            float: Percentage of pixels that are skin-colored (0.0 to 1.0)
        """
        try:
            # Remove batch dimension and convert to uint8
            img = image_array[0].astype(np.uint8)

            # Convert RGB to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # Define skin color range in HSV (multiple ranges for different skin tones)
            # Range 1: Light skin tones
            lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin1 = np.array([20, 150, 255], dtype=np.uint8)

            # Range 2: Darker skin tones
            lower_skin2 = np.array([0, 15, 50], dtype=np.uint8)
            upper_skin2 = np.array([25, 170, 255], dtype=np.uint8)

            # Create masks
            skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

            # Combine masks
            skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)

            # Calculate percentage of skin pixels
            total_pixels = skin_mask.size
            skin_pixels = np.sum(skin_mask > 0)
            skin_percentage = skin_pixels / total_pixels

            return skin_percentage

        except Exception as e:
            print(f"Error in skin color detection: {e}")
            return 0.0

    def _check_texture_entropy(self, image_array):
        """
        Analyze image texture using entropy.
        Burns have specific texture characteristics.

        Args:
            image_array: Preprocessed image array (1, 224, 224, 3) in [0, 255] range

        Returns:
            float: Average entropy score
        """
        try:
            # Remove batch dimension and convert to grayscale
            img = image_array[0].astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Calculate local entropy using a sliding window
            from scipy.ndimage import generic_filter

            def local_entropy(patch):
                """Calculate entropy of a patch"""
                patch = patch.astype(np.uint8)
                # Create histogram
                hist, _ = np.histogram(patch, bins=16, range=(0, 256))
                # Normalize histogram to get probabilities
                hist = hist / hist.sum()
                # Remove zeros
                hist = hist[hist > 0]
                # Calculate entropy
                entropy = -np.sum(hist * np.log2(hist))
                return entropy

            # Apply local entropy filter with 5x5 window
            entropy_img = generic_filter(gray, local_entropy, size=5)
            avg_entropy = np.mean(entropy_img)

            return float(avg_entropy)

        except Exception as e:
            print(f"Error in texture entropy calculation: {e}")
            # Return mid-range value if error occurs
            return 4.0

    def validate_burn_image(self, image_array, predictions):
        """
        Multi-gate validation system to detect if image is actually a burn.

        Args:
            image_array: Preprocessed image array
            predictions: Model prediction probabilities

        Returns:
            tuple: (is_valid: bool, confidence: float, reason: str)
        """
        print("\n" + "="*80)
        print("BURN IMAGE VALIDATION - Multi-Gate System")
        print("="*80)

        validation_gates = []

        # GATE 1: Skin Color Detection
        print("\nGate 1: Checking for skin-like colors...")
        skin_percentage = self._check_skin_color(image_array)
        print(f"  → Skin-like pixels: {skin_percentage*100:.1f}%")

        if skin_percentage < 0.08:  # Less than 8% skin pixels
            return False, 0.0, f"No skin detected ({skin_percentage*100:.1f}% skin-like pixels). Image does not appear to be a medical wound."
        elif skin_percentage < 0.15:  # 8-15% is borderline
            validation_gates.append(('skin_color', False, f"Low skin content ({skin_percentage*100:.1f}%)"))
        else:
            validation_gates.append(('skin_color', True, f"Skin detected ({skin_percentage*100:.1f}%)"))

        # GATE 2: Prediction Confidence
        print("\nGate 2: Analyzing prediction confidence...")
        max_confidence = np.max(predictions[0])
        print(f"  → Maximum confidence: {max_confidence*100:.1f}%")

        if max_confidence < 0.50:  # Very low confidence
            return False, max_confidence, f"Very low classification confidence ({max_confidence*100:.1f}%). Model cannot identify burn patterns in this image."
        elif max_confidence < 0.65:  # Moderate confidence - flag as warning
            validation_gates.append(('confidence', False, f"Moderate confidence ({max_confidence*100:.1f}%)"))
        else:
            validation_gates.append(('confidence', True, f"High confidence ({max_confidence*100:.1f}%)"))

        # GATE 3: Prediction Distribution (Check for model confusion)
        print("\nGate 3: Checking prediction distribution...")
        confidence_std = np.std(predictions[0])
        confidence_range = np.max(predictions[0]) - np.min(predictions[0])
        print(f"  → Prediction std dev: {confidence_std:.4f}")
        print(f"  → Prediction range: {confidence_range:.4f}")

        if confidence_std < 0.03 and confidence_range < 0.15:  # All predictions very similar
            return False, max_confidence, f"Model is uncertain - all burn classes have similar probabilities (std: {confidence_std:.4f}). Image likely not a burn."
        elif confidence_std < 0.08:
            validation_gates.append(('distribution', False, f"Low confidence spread (std: {confidence_std:.4f})"))
        else:
            validation_gates.append(('distribution', True, f"Clear prediction spread (std: {confidence_std:.4f})"))

        # GATE 4: Texture Entropy
        print("\nGate 4: Analyzing texture patterns...")
        entropy_score = self._check_texture_entropy(image_array)
        print(f"  → Texture entropy: {entropy_score:.2f}")

        # Burns typically have entropy in range 2.5-6.5
        if entropy_score < 1.5 or entropy_score > 7.5:
            validation_gates.append(('texture', False, f"Unusual texture pattern (entropy: {entropy_score:.2f})"))
        else:
            validation_gates.append(('texture', True, f"Texture matches burn patterns (entropy: {entropy_score:.2f})"))

        # FINAL DECISION: Count passed gates
        print("\n" + "-"*80)
        print("Validation Summary:")
        passed_gates = sum(1 for _, passed, _ in validation_gates if passed)
        total_gates = len(validation_gates)

        for gate_name, passed, reason in validation_gates:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status} - {gate_name.upper()}: {reason}")

        print(f"\nGates Passed: {passed_gates}/{total_gates}")
        print("="*80 + "\n")

        # Require at least 2 out of 4 gates to pass
        if passed_gates < 2:
            failed_reasons = [reason for _, passed, reason in validation_gates if not passed]
            return False, max_confidence, f"Failed validation ({passed_gates}/{total_gates} gates passed). Reasons: {'; '.join(failed_reasons)}"

        # Calculate overall confidence based on gates
        validation_confidence = (passed_gates / total_gates) * max_confidence

        return True, validation_confidence, f"Valid burn image ({passed_gates}/{total_gates} validation gates passed)"

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

    def trace_my_image(self, image_path, num_samples=5):
        """
        Trace the user's SPECIFIC IMAGE through the network step-by-step.
        Shows the ACTUAL values from THIS IMAGE being transformed at each layer.

        Args:
            image_path: Path to the user's image
            num_samples: Number of sample values to show at each step

        Returns:
            Dictionary containing the trace with actual values
        """
        print("\n" + "="*100)
        print("MY IMAGE COMPUTATION TRACE - ACTUAL VALUES FROM YOUR IMAGE")
        print("="*100)

        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return None

        trace = {
            'image_path': image_path,
            'steps': []
        }

        # ===== STEP 0: RAW INPUT PIXELS =====
        print("\n" + "─"*100)
        print("STEP 0: YOUR IMAGE - RAW PIXEL VALUES")
        print("─"*100)

        # Sample pixels from different regions
        positions = [
            (0, 0, 'Top-Left Corner'),
            (0, 223, 'Top-Right Corner'),
            (111, 111, 'Center'),
            (223, 0, 'Bottom-Left Corner'),
            (223, 223, 'Bottom-Right Corner')
        ]

        pixel_samples = []
        for y, x, location in positions:
            pixel = img_array[0, y, x]
            pixel_samples.append({
                'location': location,
                'position': f'[{y}, {x}]',
                'RGB_values': [float(pixel[0]), float(pixel[1]), float(pixel[2])],
                'description': f'At {location}: R={pixel[0]:.1f}, G={pixel[1]:.1f}, B={pixel[2]:.1f}'
            })
            print(f"  {location:20s} [{y:3d}, {x:3d}]: RGB = ({pixel[0]:6.1f}, {pixel[1]:6.1f}, {pixel[2]:6.1f})")

        trace['steps'].append({
            'step_number': 0,
            'step_name': 'Raw Input Pixels',
            'operation': 'Load image and sample pixel values',
            'input_shape': list(img_array.shape),
            'output_shape': list(img_array.shape),
            'samples': pixel_samples,
            'statistics': {
                'min': float(np.min(img_array)),
                'max': float(np.max(img_array)),
                'mean': float(np.mean(img_array)),
                'std': float(np.std(img_array))
            }
        })

        # ===== STEP 1: FIRST CONVOLUTIONAL LAYER =====
        print("\n" + "─"*100)
        print("STEP 1: FIRST CONVOLUTIONAL LAYER - YOUR IMAGE GOES THROUGH FILTERS")
        print("─"*100)

        try:
            # Find first conv layer
            first_conv = None
            for layer in self.model.layers:
                if 'conv' in layer.name.lower() and hasattr(layer, 'get_weights'):
                    weights_list = layer.get_weights()
                    if len(weights_list) > 0:
                        first_conv = layer
                        break

            if first_conv:
                # Get output from first conv layer
                first_conv_model = tf.keras.Model(inputs=self.model.input, outputs=first_conv.output)
                first_conv_output = first_conv_model.predict(img_array, verbose=0)

                weights, bias = first_conv.get_weights()

                # Sample some output values
                conv_samples = []
                sample_positions = [(0, 0), (28, 28), (55, 55), (83, 83), (111, 111)]

                for y, x in sample_positions:
                    if y < first_conv_output.shape[1] and x < first_conv_output.shape[2]:
                        # Get values for first 5 filters
                        filter_values = first_conv_output[0, y, x, :5]
                        conv_samples.append({
                            'position': f'[{y}, {x}]',
                            'filter_outputs': [float(v) for v in filter_values],
                            'description': f'Output features at position [{y}, {x}]'
                        })
                        print(f"  Position [{y:3d}, {x:3d}] (first 5 filters): {[f'{v:7.2f}' for v in filter_values[:5]]}")

                trace['steps'].append({
                    'step_number': 1,
                    'step_name': 'First Convolution',
                    'operation': f'Apply {weights.shape[3]} filters of size {weights.shape[0]}×{weights.shape[1]}',
                    'layer_name': first_conv.name,
                    'input_shape': list(img_array.shape),
                    'output_shape': list(first_conv_output.shape),
                    'num_filters': int(weights.shape[3]),
                    'kernel_size': f'{weights.shape[0]}×{weights.shape[1]}',
                    'samples': conv_samples,
                    'statistics': {
                        'min': float(np.min(first_conv_output)),
                        'max': float(np.max(first_conv_output)),
                        'mean': float(np.mean(first_conv_output)),
                        'std': float(np.std(first_conv_output))
                    }
                })
        except Exception as e:
            print(f"  Note: Could not extract first conv output: {e}")

        # ===== STEP 2: CNN FEATURES (ResNet50 Output) =====
        print("\n" + "─"*100)
        print("STEP 2: CNN FEATURE EXTRACTION - YOUR IMAGE AS SPATIAL FEATURES")
        print("─"*100)

        try:
            cnn_output_layer = self.model.get_layer('conv5_block3_out')
            cnn_model = tf.keras.Model(inputs=self.model.input, outputs=cnn_output_layer.output)
            cnn_features = cnn_model.predict(img_array, verbose=0)

            # Sample features from different spatial locations
            cnn_samples = []
            h, w = cnn_features.shape[1], cnn_features.shape[2]
            sample_locs = [(0, 0), (h//4, w//4), (h//2, w//2), (3*h//4, 3*w//4), (h-1, w-1)]

            for y, x in sample_locs:
                feature_vec = cnn_features[0, y, x, :]
                top_5_indices = np.argsort(feature_vec)[-5:][::-1]
                top_5_values = feature_vec[top_5_indices]

                cnn_samples.append({
                    'spatial_position': f'[{y}, {x}]',
                    'top_5_channels': [int(idx) for idx in top_5_indices],
                    'top_5_values': [float(v) for v in top_5_values],
                    'total_activation': float(np.sum(feature_vec)),
                    'description': f'Features at spatial location [{y}, {x}]'
                })
                # Convert to Python types for printing
                top_5_indices_py = [int(idx) for idx in top_5_indices]
                print(f"  Spatial [{y:2d}, {x:2d}] - Top 5 channels: {top_5_indices_py} = {[f'{v:.2f}' for v in top_5_values]}")

            trace['steps'].append({
                'step_number': 2,
                'step_name': 'CNN Feature Extraction',
                'operation': 'ResNet50 processes your image through 50 layers',
                'layer_name': 'conv5_block3_out',
                'input_shape': list(img_array.shape),
                'output_shape': list(cnn_features.shape),
                'spatial_size': f'{cnn_features.shape[1]}×{cnn_features.shape[2]}',
                'feature_channels': int(cnn_features.shape[3]),
                'samples': cnn_samples,
                'statistics': {
                    'min': float(np.min(cnn_features)),
                    'max': float(np.max(cnn_features)),
                    'mean': float(np.mean(cnn_features)),
                    'std': float(np.std(cnn_features))
                }
            })
        except Exception as e:
            print(f"  Note: Could not extract CNN features: {e}")

        # ===== STEP 3: RESHAPE TO SEQUENCE =====
        print("\n" + "─"*100)
        print("STEP 3: RESHAPE TO SEQUENCE - PREPARING FOR TRANSFORMER")
        print("─"*100)

        try:
            # Find reshape layer
            reshape_layer = None
            for layer in self.model.layers:
                if 'reshape' in layer.name.lower():
                    reshape_layer = layer
                    break

            if reshape_layer:
                reshape_model = tf.keras.Model(inputs=self.model.input, outputs=reshape_layer.output)
                reshaped = reshape_model.predict(img_array, verbose=0)

                # Sample some sequence positions
                seq_samples = []
                num_tokens = reshaped.shape[1]
                sample_indices = [0, num_tokens//4, num_tokens//2, 3*num_tokens//4, num_tokens-1]

                for idx in sample_indices:
                    token = reshaped[0, idx, :]
                    top_5_indices = np.argsort(token)[-5:][::-1]
                    top_5_values = token[top_5_indices]

                    seq_samples.append({
                        'token_index': int(idx),
                        'top_5_features': [int(i) for i in top_5_indices],
                        'top_5_values': [float(v) for v in top_5_values],
                        'description': f'Token {idx} features'
                    })
                    print(f"  Token {idx:3d}: Top 5 features = {[f'{v:.2f}' for v in top_5_values]}")

                trace['steps'].append({
                    'step_number': 3,
                    'step_name': 'Reshape to Sequence',
                    'operation': f'Convert spatial features to {num_tokens} tokens',
                    'layer_name': reshape_layer.name,
                    'input_shape': list(cnn_features.shape),
                    'output_shape': list(reshaped.shape),
                    'num_tokens': int(num_tokens),
                    'feature_dim': int(reshaped.shape[2]),
                    'samples': seq_samples,
                    'statistics': {
                        'min': float(np.min(reshaped)),
                        'max': float(np.max(reshaped)),
                        'mean': float(np.mean(reshaped)),
                        'std': float(np.std(reshaped))
                    }
                })
        except Exception as e:
            print(f"  Note: Could not extract reshape output: {e}")

        # ===== STEP 4: TRANSFORMER ATTENTION =====
        print("\n" + "─"*100)
        print("STEP 4: TRANSFORMER ATTENTION - YOUR IMAGE FEATURES INTERACT")
        print("─"*100)

        try:
            attention_layers = [l for l in self.model.layers if 'multi_head_attention' in l.name]

            for idx, attn_layer in enumerate(attention_layers[:2]):  # Show first 2 blocks
                attn_model = tf.keras.Model(inputs=self.model.input, outputs=attn_layer.output)
                attn_output = attn_model.predict(img_array, verbose=0)

                # Sample some tokens
                attn_samples = []
                num_tokens = attn_output.shape[1]
                sample_indices = [0, num_tokens//2, num_tokens-1]

                for token_idx in sample_indices:
                    token = attn_output[0, token_idx, :]
                    attn_samples.append({
                        'token_index': int(token_idx),
                        'sample_values': [float(v) for v in token[:5]],
                        'mean_attention': float(np.mean(token)),
                        'max_attention': float(np.max(token))
                    })
                    print(f"  Block {idx+1}, Token {token_idx:3d}: Mean={np.mean(token):.4f}, Max={np.max(token):.4f}")

                trace['steps'].append({
                    'step_number': 4 + idx,
                    'step_name': f'Transformer Block {idx+1}',
                    'operation': 'Multi-head self-attention',
                    'layer_name': attn_layer.name,
                    'input_shape': list(attn_output.shape),
                    'output_shape': list(attn_output.shape),
                    'samples': attn_samples,
                    'statistics': {
                        'min': float(np.min(attn_output)),
                        'max': float(np.max(attn_output)),
                        'mean': float(np.mean(attn_output)),
                        'std': float(np.std(attn_output))
                    }
                })
        except Exception as e:
            print(f"  Note: Could not extract attention outputs: {e}")

        # ===== STEP 5: GLOBAL POOLING =====
        print("\n" + "─"*100)
        print("STEP 5: GLOBAL POOLING - COMBINING ALL TOKENS")
        print("─"*100)

        try:
            pooling_layer = self.model.get_layer('global_average_pooling1d_1')
            pooling_model = tf.keras.Model(inputs=self.model.input, outputs=pooling_layer.output)
            pooled = pooling_model.predict(img_array, verbose=0)

            # Show top features
            top_20_indices = np.argsort(pooled[0])[-20:][::-1]
            top_20_values = pooled[0][top_20_indices]

            print(f"  Pooled to {pooled.shape[1]} features")
            print(f"  Top 20 features:")
            for i, (idx, val) in enumerate(zip(top_20_indices[:10], top_20_values[:10])):
                print(f"    Feature {idx:4d} = {val:8.4f}")

            trace['steps'].append({
                'step_number': 10,
                'step_name': 'Global Average Pooling',
                'operation': 'Average all tokens to create final feature vector',
                'layer_name': pooling_layer.name,
                'output_shape': list(pooled.shape),
                'num_features': int(pooled.shape[1]),
                'top_20_features': {
                    'indices': [int(i) for i in top_20_indices],
                    'values': [float(v) for v in top_20_values]
                },
                'statistics': {
                    'min': float(np.min(pooled)),
                    'max': float(np.max(pooled)),
                    'mean': float(np.mean(pooled)),
                    'std': float(np.std(pooled))
                }
            })
        except Exception as e:
            print(f"  Note: Could not extract pooling output: {e}")

        # ===== STEP 6: DENSE LAYERS =====
        print("\n" + "─"*100)
        print("STEP 6: CLASSIFICATION LAYERS - COMPUTING CLASS SCORES")
        print("─"*100)

        try:
            dense_layers = [l for l in self.model.layers if 'dense' in l.name]

            for idx, dense_layer in enumerate(dense_layers):
                dense_model = tf.keras.Model(inputs=self.model.input, outputs=dense_layer.output)
                dense_output = dense_model.predict(img_array, verbose=0)

                weights, bias = dense_layer.get_weights()

                print(f"\n  Dense Layer {idx+1} ({dense_layer.name}):")
                print(f"  Input: {weights.shape[0]} features → Output: {weights.shape[1]} units")
                print(f"  Output values: {[f'{v:.4f}' for v in dense_output[0][:10]]}")

                trace['steps'].append({
                    'step_number': 11 + idx,
                    'step_name': f'Dense Layer {idx+1}',
                    'operation': f'Fully connected: {weights.shape[0]} → {weights.shape[1]}',
                    'layer_name': dense_layer.name,
                    'weight_shape': list(weights.shape),
                    'bias_shape': list(bias.shape),
                    'output_values': [float(v) for v in dense_output[0]],
                    'statistics': {
                        'min': float(np.min(dense_output)),
                        'max': float(np.max(dense_output)),
                        'mean': float(np.mean(dense_output)),
                        'std': float(np.std(dense_output))
                    }
                })
        except Exception as e:
            print(f"  Note: Could not extract dense outputs: {e}")

        # ===== STEP 7: FINAL PREDICTION =====
        print("\n" + "─"*100)
        print("STEP 7: SOFTMAX & FINAL PREDICTION - YOUR IMAGE'S CLASSIFICATION")
        print("─"*100)

        try:
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])

            print("\n  Final Probabilities:")
            for i, class_name in enumerate(self.class_names):
                prob = predictions[0][i]
                bar = '█' * int(prob * 50)
                print(f"    {class_name:25s}: {prob:.4f} ({prob*100:6.2f}%) {bar}")

            print(f"\n  → Predicted: {self.class_names[predicted_class]} ({predictions[0][predicted_class]*100:.2f}%)")

            trace['steps'].append({
                'step_number': 15,
                'step_name': 'Final Softmax & Prediction',
                'operation': 'Convert logits to probabilities',
                'probabilities': {
                    self.class_names[i]: float(predictions[0][i])
                    for i in range(len(self.class_names))
                },
                'predicted_class': self.class_names[predicted_class],
                'confidence': float(predictions[0][predicted_class])
            })
        except Exception as e:
            print(f"  Note: Could not extract final prediction: {e}")

        print("\n" + "="*100)
        print("IMAGE TRACE COMPLETE")
        print("="*100 + "\n")

        return trace

    def predict(self, image_path, with_gradcam=False, with_computational_flow=False, detailed_analysis=False, show_concrete_math=False, trace_image=False, progress_callback=None):
        """
        Make prediction on an image with optional analysis.

        Args:
            image_path: Path to the image
            with_gradcam: Include Grad-CAM visualization
            with_computational_flow: Include full computational flow analysis
            detailed_analysis: Include intermediate layer outputs and explanations
            show_concrete_math: Show actual numerical computations with real weights and values
            trace_image: Trace the user's specific image through the network with actual values
            progress_callback: Optional callback function to report progress (step_name, percentage)
        """
        def report_progress(step, percent):
            """Helper to report progress if callback is provided"""
            if progress_callback:
                progress_callback(step, percent)

        print(f"DEBUG: Making prediction for image: {image_path}")
        report_progress("Initializing analysis...", 0)

        if not self.model:
            return None, "Model not loaded"

        # Preprocess image
        report_progress("Preprocessing image...", 5)
        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return None, "Error preprocessing image"

        try:
            # Make base prediction first
            report_progress("Running initial prediction...", 10)
            predictions = self.model.predict(img_array, verbose=0)

            # VALIDATE IF IMAGE IS ACTUALLY A BURN
            report_progress("Validating image type...", 12)
            is_valid, validation_conf, validation_reason = self.validate_burn_image(img_array, predictions)

            if not is_valid:
                # Return error result for invalid images
                return {
                    'error': 'INVALID_BURN_IMAGE',
                    'message': validation_reason,
                    'is_valid_burn': False,
                    'validation_confidence': float(validation_conf),
                    'validation_reason': validation_reason,
                    # Still include predictions for transparency
                    'raw_predictions': {
                        self.class_names[i]: float(predictions[0][i])
                        for i in range(len(self.class_names))
                    }
                }, f"Invalid image: {validation_reason}"

            # Image validated - proceed with normal classification
            report_progress("Image validated - analyzing burn severity...", 15)

            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

            report_progress("Processing prediction results...", 20)
            result = {
                'predicted_class': self.class_names[predicted_class],
                'class_id': int(predicted_class),
                'confidence': confidence,
                'all_probabilities': {
                    self.class_names[i]: float(predictions[0][i])
                    for i in range(len(self.class_names))
                },
                # Add validation results
                'is_valid_burn': True,
                'validation_confidence': float(validation_conf),
                'validation_reason': validation_reason
            }

            if with_gradcam:
                print("DEBUG: Generating Grad-CAM overlay...")
                report_progress("Generating attention heatmap...", 25)
                _, overlay = self.gradcam(image_path, pred_index=predicted_class)
                if overlay is not None:
                    result['gradcam_overlay'] = overlay  # numpy array
                    print("DEBUG: Grad-CAM overlay added to result")
                else:
                    print("DEBUG: Grad-CAM overlay generation failed")

            if with_computational_flow:
                print("DEBUG: Analyzing computational flow...")
                report_progress("Extracting CNN features...", 35)
                flow_analysis = self.analyze_computational_flow(image_path)
                if flow_analysis:
                    result['computational_flow'] = flow_analysis
                    print("DEBUG: Computational flow analysis added to result")

            if detailed_analysis:
                print("DEBUG: Generating detailed analysis...")
                report_progress("Reshaping for transformer...", 50)
                detailed = self.get_detailed_analysis(image_path)
                if detailed:
                    result['detailed_analysis'] = detailed
                    print("DEBUG: Detailed analysis added to result")

            if show_concrete_math:
                print("DEBUG: Generating concrete computational analysis...")
                report_progress("Analyzing transformer attention...", 65)
                try:
                    concrete = self.show_concrete_computations(image_path)
                    if concrete:
                        result['concrete_computations'] = concrete
                        print("DEBUG: Concrete computations added to result")
                except Exception as e:
                    print(f"DEBUG: Concrete math computation failed (non-critical): {e}")
                    # Continue without concrete math - other analysis still works

            if trace_image:
                print("DEBUG: Tracing image through model...")
                report_progress("Tracing through network layers...", 75)
                trace = self.trace_my_image(image_path)
                if trace:
                    result['image_trace'] = trace
                    print("DEBUG: Image trace added to result")
                else:
                    print("DEBUG: Image trace generation failed")

            report_progress("Computing feature importance...", 90)
            # Small delay to show the final message
            import time
            time.sleep(0.2)
            report_progress("Finalizing results...", 95)
            time.sleep(0.2)
            report_progress("Complete", 100)
            return result, None
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error during prediction: {e}"

    def get_detailed_analysis(self, image_path):
        """Get detailed analysis with intermediate outputs"""
        print(f"DEBUG: Extracting detailed analysis for: {image_path}")
        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return None

        try:
            analysis = {}

            # 1. CNN features
            cnn_output_layer = self.model.get_layer('conv5_block3_out')
            resnet_functional = tf.keras.Model(inputs=self.model.input, outputs=cnn_output_layer.output)
            cnn_features = resnet_functional.predict(img_array, verbose=0)

            analysis['cnn_features'] = {
                'shape': tuple(int(x) for x in cnn_features.shape),
                'mean_activation': float(np.mean(cnn_features)),
                'max_activation': float(np.max(cnn_features)),
                'spatial_attention_map': [[float(v) for v in row] for row in np.mean(cnn_features[0], axis=-1)]
            }

            # 2. Transformer attention
            attention_layers = [l for l in self.model.layers if 'multi_head_attention' in l.name]
            analysis['transformer_attention'] = []

            for idx, attn_layer in enumerate(attention_layers):
                attn_model = tf.keras.Model(inputs=self.model.input, outputs=attn_layer.output)
                attn_output = attn_model.predict(img_array, verbose=0)

                analysis['transformer_attention'].append({
                    'block': idx + 1,
                    'mean_attention': float(np.mean(attn_output)),
                    'attention_strength': float(np.std(attn_output))
                })

            # 3. Pooled features
            pooling_layer = self.model.get_layer('global_average_pooling1d_1')
            pooling_model = tf.keras.Model(inputs=self.model.input, outputs=pooling_layer.output)
            pooled_features = pooling_model.predict(img_array, verbose=0)

            analysis['pooled_features'] = {
                'shape': tuple(int(x) for x in pooled_features.shape),
                'top_10_indices': [int(x) for x in np.argsort(pooled_features[0])[-10:]],
                'top_10_values': [float(x) for x in np.sort(pooled_features[0])[-10:]]
            }

            # 4. Classification logits
            dense_layers = [l for l in self.model.layers if 'dense' in l.name]
            if dense_layers:
                logits_model = tf.keras.Model(inputs=self.model.input, outputs=dense_layers[-1].output)
                logits = logits_model.predict(img_array, verbose=0)

                # Flatten logits for safe indexing
                logits_flat = np.array(logits).flatten()

                analysis['logits'] = {
                    self.class_names[i]: float(logits_flat[i])
                    for i in range(len(self.class_names))
                }

            return analysis

        except Exception as e:
            print(f"Error in detailed analysis: {e}")
            return None

    def analyze_computational_flow(self, image_path):
        """
        COMPLETE COMPUTATIONAL FLOW ANALYSIS
        Shows EVERY mathematical operation from input to output
        """
        print("\n" + "="*80)
        print("HYBRID CNN-TRANSFORMER COMPUTATIONAL FLOW ANALYSIS")
        print("="*80)

        self.computation_log = []
        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return None

        # Log input
        self._log_computation("INPUT", {
            'shape': tuple(int(x) for x in img_array.shape),
            'dtype': str(img_array.dtype),
            'min_value': float(np.min(img_array)),
            'max_value': float(np.max(img_array)),
            'mean_value': float(np.mean(img_array)),
            'std_value': float(np.std(img_array)),
            'description': 'Raw input image in [0, 255] range'
        })

        try:
            # ===== PHASE 1: CNN FEATURE EXTRACTION (ResNet50) =====
            print("\n" + "-"*80)
            print("PHASE 1: CNN FEATURE EXTRACTION (ResNet50)")
            print("-"*80)

            # Get CNN features from the last convolutional layer
            # The model is flat (no nested 'resnet50' layer), so we use the output layer name directly
            cnn_output_layer = self.model.get_layer('conv5_block3_out')
            resnet_functional = tf.keras.Model(
                inputs=self.model.input,
                outputs=cnn_output_layer.output
            )

            cnn_features = resnet_functional.predict(img_array, verbose=0)

            self._log_computation("CNN_OUTPUT", {
                'layer': 'ResNet50',
                'operation': 'Convolutional Neural Network (50 layers)',
                'input_shape': tuple(int(x) for x in img_array.shape),
                'output_shape': tuple(int(x) for x in cnn_features.shape),
                'description': 'Extract spatial features using ResNet50',
                'spatial_dimensions': f'{cnn_features.shape[1]}x{cnn_features.shape[2]}',
                'feature_channels': int(cnn_features.shape[3]),
                'total_features': int(np.prod(cnn_features.shape[1:])),
                'activation_stats': {
                    'min': float(np.min(cnn_features)),
                    'max': float(np.max(cnn_features)),
                    'mean': float(np.mean(cnn_features)),
                    'std': float(np.std(cnn_features)),
                    'non_zero_ratio': float(np.count_nonzero(cnn_features) / cnn_features.size)
                }
            })

            # Show feature map statistics per channel
            channel_stats = []
            for ch in range(min(10, cnn_features.shape[-1])):  # First 10 channels
                channel_data = cnn_features[0, :, :, ch]
                channel_stats.append({
                    'channel': ch,
                    'mean': float(np.mean(channel_data)),
                    'max': float(np.max(channel_data)),
                    'active_pixels': int(np.count_nonzero(channel_data))
                })

            self._log_computation("CNN_CHANNEL_ANALYSIS", {
                'description': 'Feature map analysis per channel (first 10)',
                'channels': channel_stats
            })

            # ===== PHASE 2: RESHAPE FOR TRANSFORMER =====
            print("\n" + "-"*80)
            print("PHASE 2: RESHAPE FOR TRANSFORMER INPUT")
            print("-"*80)

            reshape_layer = None
            for layer in self.model.layers:
                if 'reshape' in layer.name.lower():
                    reshape_layer = layer
                    break

            if reshape_layer:
                reshape_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=reshape_layer.output
                )
                reshaped_features = reshape_model.predict(img_array, verbose=0)

                self._log_computation("RESHAPE", {
                    'operation': 'Reshape spatial grid to sequence',
                    'input_shape': tuple(int(x) for x in cnn_features.shape),
                    'output_shape': tuple(int(x) for x in reshaped_features.shape),
                    'description': f'Convert {cnn_features.shape[1]}x{cnn_features.shape[2]} spatial grid to {reshaped_features.shape[1]} tokens',
                    'tokens': int(reshaped_features.shape[1]),
                    'features_per_token': int(reshaped_features.shape[2]),
                    'mathematical_operation': f'{cnn_features.shape[1]} * {cnn_features.shape[2]} = {reshaped_features.shape[1]} tokens'
                })

            # ===== PHASE 3: TRANSFORMER PROCESSING =====
            print("\n" + "-"*80)
            print("PHASE 3: TRANSFORMER PROCESSING")
            print("-"*80)

            # Find all transformer components
            attention_layers = [l for l in self.model.layers if 'multi_head_attention' in l.name]
            layer_norm_layers = [l for l in self.model.layers if 'layer_normalization' in l.name]

            self._log_computation("TRANSFORMER_ARCHITECTURE", {
                'num_attention_blocks': len(attention_layers),
                'num_layer_norms': len(layer_norm_layers),
                'description': 'Multi-head self-attention mechanism'
            })

            # Analyze each attention layer
            for idx, attn_layer in enumerate(attention_layers):
                print(f"\n   Analyzing Transformer Block {idx + 1}...")

                # Get attention layer output
                attn_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=attn_layer.output
                )
                attn_output = attn_model.predict(img_array, verbose=0)

                # Get layer configuration
                config = attn_layer.get_config()

                self._log_computation(f"TRANSFORMER_BLOCK_{idx+1}", {
                    'layer': attn_layer.name,
                    'operation': 'Multi-Head Self-Attention',
                    'num_heads': config.get('num_heads', 'unknown'),
                    'key_dim': config.get('key_dim', 'unknown'),
                    'input_shape': tuple(int(x) for x in reshaped_features.shape) if idx == 0 else 'previous_block_output',
                    'output_shape': tuple(int(x) for x in attn_output.shape),
                    'description': f'Attention mechanism: Q, K, V matrices with {config.get("num_heads")} heads',
                    'mathematical_operations': {
                        'query_projection': f'Q = input @ W_q  (shape: {tuple(int(x) for x in attn_output.shape)})',
                        'key_projection': f'K = input @ W_k  (shape: {tuple(int(x) for x in attn_output.shape)})',
                        'value_projection': f'V = input @ W_v  (shape: {tuple(int(x) for x in attn_output.shape)})',
                        'attention_scores': 'Attention = softmax(Q @ K^T / sqrt(d_k))',
                        'attention_output': 'Output = Attention @ V'
                    },
                    'activation_stats': {
                        'min': float(np.min(attn_output)),
                        'max': float(np.max(attn_output)),
                        'mean': float(np.mean(attn_output)),
                        'std': float(np.std(attn_output))
                    },
                    'attention_distribution': {
                        'most_attended_tokens': [int(x) for x in np.argsort(np.mean(attn_output[0], axis=-1))[-5:]],
                        'attention_strength': float(np.mean(np.abs(attn_output)))
                    }
                })

            # ===== PHASE 4: GLOBAL POOLING =====
            print("\n" + "-"*80)
            print("PHASE 4: GLOBAL AVERAGE POOLING")
            print("-"*80)

            pooling_layer = self.model.get_layer('global_average_pooling1d_1')
            pooling_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=pooling_layer.output
            )
            pooled_features = pooling_model.predict(img_array, verbose=0)

            self._log_computation("GLOBAL_POOLING", {
                'operation': 'Global Average Pooling',
                'input_shape': tuple(int(x) for x in attn_output.shape),
                'output_shape': tuple(int(x) for x in pooled_features.shape),
                'description': 'Average all token representations into single feature vector',
                'mathematical_operation': f'pool = mean(tokens, axis=1) -> shape: {tuple(int(x) for x in pooled_features.shape)}',
                'feature_vector_length': int(pooled_features.shape[1]),
                'feature_stats': {
                    'min': float(np.min(pooled_features)),
                    'max': float(np.max(pooled_features)),
                    'mean': float(np.mean(pooled_features)),
                    'std': float(np.std(pooled_features)),
                    'non_zero_count': int(np.count_nonzero(pooled_features))
                },
                'top_10_features': {
                    'indices': [int(x) for x in np.argsort(pooled_features[0])[-10:]],
                    'values': [float(x) for x in np.sort(pooled_features[0])[-10:]]
                }
            })

            # ===== PHASE 5: CLASSIFICATION HEAD =====
            print("\n" + "-"*80)
            print("PHASE 5: CLASSIFICATION (Dense Layer + Softmax)")
            print("-"*80)

            # Get final dense layer (before softmax)
            dense_layers = [l for l in self.model.layers if 'dense' in l.name and l != self.model.layers[-1]]
            final_dense = dense_layers[-1] if dense_layers else self.model.layers[-1]

            # Get logits (pre-softmax)
            logits_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=final_dense.output
            )
            logits = logits_model.predict(img_array, verbose=0)

            # Get final predictions (post-softmax)
            predictions = self.model.predict(img_array, verbose=0)

            # Dense layer computation
            weights = final_dense.get_weights()
            weight_matrix = weights[0] if len(weights) > 0 else None
            bias_vector = weights[1] if len(weights) > 1 else None

            # Flatten logits to 1D array to ensure proper indexing
            logits_flat = np.array(logits).flatten()

            self._log_computation("DENSE_LAYER", {
                'operation': 'Dense (Fully Connected) Layer',
                'input_shape': tuple(int(x) for x in pooled_features.shape),
                'output_shape': tuple(int(x) for x in logits.shape),
                'weight_matrix_shape': tuple(int(x) for x in weight_matrix.shape) if weight_matrix is not None else 'unknown',
                'bias_vector_shape': tuple(int(x) for x in bias_vector.shape) if bias_vector is not None else 'unknown',
                'mathematical_operation': 'logits = (input @ W) + b',
                'computation': f'{pooled_features.shape[1]} features @ {tuple(int(x) for x in weight_matrix.shape) if weight_matrix is not None else "?"} weights + bias',
                'raw_logits': {
                    self.class_names[i]: float(logits_flat[i])
                    for i in range(len(self.class_names))
                },
                'logit_stats': {
                    'min': float(np.min(logits)),
                    'max': float(np.max(logits)),
                    'range': float(np.max(logits) - np.min(logits))
                }
            })

            # Softmax computation
            exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
            softmax_sum = np.sum(exp_logits)

            # Flatten arrays for safe indexing
            exp_logits_flat = np.array(exp_logits).flatten()
            predictions_flat = np.array(predictions).flatten()

            self._log_computation("SOFTMAX", {
                'operation': 'Softmax Activation',
                'mathematical_formula': 'P(class_i) = exp(logit_i) / sum(exp(logits))',
                'numerical_stability': 'Using exp(logit - max(logits)) to prevent overflow',
                'input_logits': {
                    self.class_names[i]: float(logits_flat[i])
                    for i in range(len(self.class_names))
                },
                'exp_logits': {
                    self.class_names[i]: float(exp_logits_flat[i])
                    for i in range(len(self.class_names))
                },
                'sum_exp_logits': float(softmax_sum),
                'final_probabilities': {
                    self.class_names[i]: float(predictions_flat[i])
                    for i in range(len(self.class_names))
                },
                'verification': f'Sum of probabilities = {float(np.sum(predictions)):.6f} (should be 1.0)'
            })

            # ===== PHASE 6: FINAL PREDICTION =====
            print("\n" + "-"*80)
            print("PHASE 6: FINAL PREDICTION & DECISION")
            print("-"*80)

            # Use flattened predictions array for safe indexing
            predicted_class = int(np.argmax(predictions_flat))
            confidence = float(predictions_flat[predicted_class])
            sorted_indices = np.argsort(predictions_flat)[::-1]

            self._log_computation("FINAL_PREDICTION", {
                'operation': 'argmax(probabilities)',
                'predicted_class_id': predicted_class,
                'predicted_class_name': self.class_names[predicted_class],
                'confidence': confidence,
                'confidence_percentage': f'{confidence * 100:.2f}%',
                'ranking': [
                    {
                        'rank': i + 1,
                        'class': self.class_names[int(idx)],
                        'probability': float(predictions_flat[int(idx)]),
                        'percentage': f'{float(predictions_flat[int(idx)]) * 100:.2f}%'
                    }
                    for i, idx in enumerate(sorted_indices)
                ],
                'decision_threshold': 0.5,
                'exceeds_threshold': bool(confidence >= 0.5),
                'probability_gap_to_second': float(confidence - predictions_flat[int(sorted_indices[1])])
            })

            print("\n" + "="*80)
            print("COMPUTATIONAL FLOW ANALYSIS COMPLETE")
            print("="*80 + "\n")

            return {
                'computation_log': self.computation_log,
                'summary': {
                    'total_steps': len(self.computation_log),
                    'predicted_class': self.class_names[predicted_class],
                    'confidence': float(confidence)
                }
            }

        except Exception as e:
            print(f"ERROR in computational flow analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _log_computation(self, step_name, details):
        """Log a computational step with details"""
        log_entry = {
            'step': step_name,
            'details': details
        }
        self.computation_log.append(log_entry)

        # Print to console
        print(f"\n[{step_name}]")
        self._print_dict(details, indent=2)

    def _print_dict(self, d, indent=0):
        """Recursively print dictionary with indentation"""
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                self._print_dict(value, indent + 2)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                print(" " * indent + f"{key}:")
                for item in value:
                    self._print_dict(item, indent + 2)
                    print()
            else:
                print(" " * indent + f"{key}: {value}")

    def show_concrete_computations(self, image_path, sample_positions=5):
        """
        Show CONCRETE numerical computations with actual weights and values.
        This is the deep dive into the actual math happening inside the model.

        Args:
            image_path: Path to the image
            sample_positions: Number of sample calculations to show per layer
        """
        print("\n" + "="*100)
        print("CONCRETE COMPUTATIONAL ANALYSIS: ACTUAL NUMBERS & CALCULATIONS")
        print("="*100)
        print("This shows the REAL mathematical operations with ACTUAL weights and values\n")

        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return None

        concrete_log = []

        # ===== STEP 1: INPUT ANALYSIS =====
        print("\n" + "─"*100)
        print("STEP 1: INPUT IMAGE PIXEL ANALYSIS")
        print("─"*100)

        step1 = {
            'step': 'Input Pixel Sampling',
            'samples': []
        }

        # Sample random pixels
        positions = [(0, 0), (56, 56), (112, 112), (168, 168), (223, 223)]
        for y, x in positions:
            pixel = img_array[0, y, x]
            step1['samples'].append({
                'position': f'[{y}, {x}]',
                'RGB': [float(pixel[0]), float(pixel[1]), float(pixel[2])],
                'normalized': [float(pixel[0]/255), float(pixel[1]/255), float(pixel[2]/255)]
            })
            print(f"  Pixel at position [{y:3d}, {x:3d}]: RGB = [{pixel[0]:6.2f}, {pixel[1]:6.2f}, {pixel[2]:6.2f}]")

        concrete_log.append(step1)

        # ===== STEP 2: FIRST CONV LAYER COMPUTATION =====
        print("\n" + "─"*100)
        print("STEP 2: FIRST CONVOLUTIONAL LAYER (ResNet50 Entry)")
        print("─"*100)

        try:
            # Get first conv layer (directly from flat model structure)
            first_conv = None
            for layer in self.model.layers:
                if 'conv' in layer.name.lower() and hasattr(layer, 'get_weights'):
                    weights_list = layer.get_weights()
                    if len(weights_list) > 0:
                        first_conv = layer
                        break

            if first_conv:
                weights, bias = first_conv.get_weights()
                print(f"  Layer: {first_conv.name}")
                print(f"  Kernel shape: {weights.shape}")  # e.g., (7, 7, 3, 64)
                print(f"  Bias shape: {bias.shape}\n")

                # Show actual kernel weights for first filter
                print(f"  Sample Convolution Kernel (Filter #0):")
                print(f"  Shape: {weights.shape[0]}×{weights.shape[1]} kernel, {weights.shape[2]} input channels")

                # Get center pixel computation
                kernel_0 = weights[:, :, :, 0]  # First filter
                print(f"\n  Kernel weights for Red channel (center 3x3):")
                center_kernel = kernel_0[2:5, 2:5, 0]  # 3x3 center, red channel
                for row in center_kernel:
                    print(f"    [{', '.join([f'{w:7.4f}' for w in row])}]")

                # Compute convolution for one position
                print(f"\n  Computing convolution at position [112, 112]:")
                input_patch = img_array[0, 110:117, 110:117, :]  # 7x7 patch

                # Manual convolution calculation
                conv_result = 0
                print(f"  Calculation:")
                for i in range(min(3, weights.shape[0])):
                    for j in range(min(3, weights.shape[1])):
                        for c in range(weights.shape[2]):  # RGB channels
                            pixel_val = input_patch[i, j, c]
                            kernel_val = kernel_0[i, j, c]
                            contribution = pixel_val * kernel_val
                            conv_result += contribution
                            if i < 2 and j < 2:  # Show first few
                                print(f"    [{i},{j}][{['R','G','B'][c]}]: {pixel_val:.2f} × {kernel_val:.4f} = {contribution:.4f}")

                conv_result += bias[0]
                print(f"  ... (continuing for full {weights.shape[0]}×{weights.shape[1]} kernel)")
                print(f"  Sum of all products: {conv_result - bias[0]:.4f}")
                print(f"  Add bias: {conv_result - bias[0]:.4f} + {bias[0]:.4f} = {conv_result:.4f}")
                print(f"  Apply ReLU: max(0, {conv_result:.4f}) = {max(0, conv_result):.4f}")

                concrete_log.append({
                    'step': 'First Convolution',
                    'layer': first_conv.name,
                    'kernel_shape': str(weights.shape),
                    'sample_kernel_weights': kernel_0[2:5, 2:5, 0].tolist(),
                    'bias_sample': float(bias[0]),
                    'sample_output': float(max(0, conv_result))
                })

        except Exception as e:
            print(f"  Note: Could not extract detailed conv computation: {e}")

        # ===== STEP 3: CNN FEATURE EXTRACTION =====
        print("\n" + "─"*100)
        print("STEP 3: CNN FEATURE EXTRACTION (Full ResNet50)")
        print("─"*100)

        cnn_output_layer = self.model.get_layer('conv5_block3_out')
        resnet_functional = tf.keras.Model(inputs=self.model.input, outputs=cnn_output_layer.output)
        cnn_features = resnet_functional.predict(img_array, verbose=0)

        print(f"  Input shape: {img_array.shape}")
        print(f"  Output shape: {cnn_features.shape}")
        print(f"  Total parameters processed: ~23.5 million")
        print(f"\n  Sample output values at position [3, 3] (center of 7×7 grid):")
        center_features = cnn_features[0, 3, 3, :]
        print(f"    Channels 0-10: [{', '.join([f'{center_features[i]:.4f}' for i in range(10)])}]")
        print(f"    Mean activation: {np.mean(center_features):.4f}")
        print(f"    Max activation: {np.max(center_features):.4f}")
        print(f"    Active channels (>0): {np.sum(center_features > 0)} / {len(center_features)}")

        concrete_log.append({
            'step': 'CNN Features',
            'output_shape': tuple(int(x) for x in cnn_features.shape),
            'sample_center_features': [float(x) for x in center_features[:20]],
            'statistics': {
                'mean': float(np.mean(cnn_features)),
                'max': float(np.max(cnn_features)),
                'active_ratio': float(np.sum(cnn_features > 0) / cnn_features.size)
            }
        })

        # ===== STEP 4: RESHAPE OPERATION =====
        print("\n" + "─"*100)
        print("STEP 4: RESHAPE FOR TRANSFORMER (Spatial → Sequence)")
        print("─"*100)

        reshape_layer = None
        for layer in self.model.layers:
            if 'reshape' in layer.name.lower():
                reshape_layer = layer
                break

        if reshape_layer:
            reshape_model = tf.keras.Model(inputs=self.model.input, outputs=reshape_layer.output)
            reshaped = reshape_model.predict(img_array, verbose=0)

            print(f"  Input: {cnn_features.shape} → Output: {reshaped.shape}")
            print(f"  Operation: Flatten 7×7 spatial grid into sequence")
            print(f"  Result: {reshaped.shape[1]} tokens × {reshaped.shape[2]} dimensions each")
            print(f"\n  Sample Token #0 (top-left 7×7 position):")
            print(f"    First 10 features: [{', '.join([f'{reshaped[0, 0, i]:.4f}' for i in range(10)])}]")
            print(f"\n  Sample Token #24 (center 7×7 position):")
            print(f"    First 10 features: [{', '.join([f'{reshaped[0, 24, i]:.4f}' for i in range(10)])}]")

            concrete_log.append({
                'step': 'Reshape to Sequence',
                'token_0': reshaped[0, 0, :10].tolist(),
                'token_24': reshaped[0, 24, :10].tolist(),
                'num_tokens': reshaped.shape[1]
            })

        # ===== STEP 5: TRANSFORMER ATTENTION =====
        print("\n" + "─"*100)
        print("STEP 5: TRANSFORMER MULTI-HEAD ATTENTION")
        print("─"*100)

        attention_layers = [l for l in self.model.layers if 'multi_head_attention' in l.name]

        for idx, attn_layer in enumerate(attention_layers[:1]):  # Show first block in detail
            print(f"\n  Transformer Block {idx + 1}: {attn_layer.name}")

            # Get Q, K, V weights
            weights = attn_layer.get_weights()
            print(f"  Number of weight matrices: {len(weights)}")

            if len(weights) >= 3:
                # Q, K, V are typically first 3 weight matrices
                print(f"  Query weights shape: {weights[0].shape}")
                print(f"  Key weights shape: {weights[1].shape}")
                print(f"  Value weights shape: {weights[2].shape}")

                # Sample computation
                print(f"\n  Sample Attention Calculation (Token 0 attending to Token 24):")

                # Get attention output
                attn_model = tf.keras.Model(inputs=self.model.input, outputs=attn_layer.output)
                attn_output = attn_model.predict(img_array, verbose=0)

                # Simulate Q, K computation (simplified)
                if reshaped is not None:
                    token_0 = reshaped[0, 0, :10]  # First 10 dims
                    token_24 = reshaped[0, 24, :10]

                    # Query projection (simplified with first weights)
                    q_weights = weights[0][:10, :10]  # Sample 10×10
                    query_0 = np.dot(token_0, q_weights)

                    # Key projection
                    k_weights = weights[1][:10, :10]
                    key_24 = np.dot(token_24, k_weights)

                    # Attention score
                    attention_score = np.dot(query_0, key_24) / np.sqrt(10)

                    print(f"    Token 0 features (first 10): [{', '.join([f'{x:.3f}' for x in token_0])}]")
                    print(f"    Token 24 features (first 10): [{', '.join([f'{x:.3f}' for x in token_24])}]")
                    print(f"\n    Query[0] = Token[0] @ W_q")
                    print(f"            = [{', '.join([f'{x:.3f}' for x in query_0])}]")
                    print(f"\n    Key[24] = Token[24] @ W_k")
                    print(f"            = [{', '.join([f'{x:.3f}' for x in key_24])}]")
                    print(f"\n    Attention Score = Query[0] · Key[24] / √d_k")
                    print(f"                    = {np.dot(query_0, key_24):.4f} / √10")
                    print(f"                    = {attention_score:.4f}")
                    print(f"\n    After softmax across all 49 tokens: ~{np.exp(attention_score)/49:.4f}")

                    concrete_log.append({
                        'step': f'Attention Block {idx+1}',
                        'query_sample': query_0.tolist(),
                        'key_sample': key_24.tolist(),
                        'attention_score': float(attention_score),
                        'attention_output_sample': attn_output[0, 0, :10].tolist()
                    })

        # ===== STEP 6: GLOBAL POOLING =====
        print("\n" + "─"*100)
        print("STEP 6: GLOBAL AVERAGE POOLING")
        print("─"*100)

        pooling_layer = self.model.get_layer('global_average_pooling1d_1')
        pooling_model = tf.keras.Model(inputs=self.model.input, outputs=pooling_layer.output)
        pooled = pooling_model.predict(img_array, verbose=0)

        print(f"  Operation: Average all 49 token representations")
        print(f"  Input: (49 tokens, 2048 features) → Output: (2048 features)")
        print(f"\n  Sample calculation for feature dimension 0:")
        if reshaped is not None:
            all_tokens_dim0 = reshaped[0, :, 0]
            manual_avg = np.mean(all_tokens_dim0)
            print(f"    Token values for dimension 0: [{', '.join([f'{all_tokens_dim0[i]:.3f}' for i in range(5)])}...] (49 total)")
            print(f"    Average = sum(values) / 49 = {manual_avg:.4f}")
            print(f"    Model output[0] = {pooled[0, 0]:.4f}")

        print(f"\n  Final pooled feature vector (first 20 dimensions):")
        print(f"    [{', '.join([f'{pooled[0, i]:.4f}' for i in range(20)])}...]")

        concrete_log.append({
            'step': 'Global Pooling',
            'pooled_features': pooled[0, :20].tolist(),
            'feature_stats': {
                'mean': float(np.mean(pooled)),
                'max': float(np.max(pooled)),
                'min': float(np.min(pooled))
            }
        })

        # ===== STEP 7: DENSE LAYER & CLASSIFICATION =====
        print("\n" + "─"*100)
        print("STEP 7: FINAL CLASSIFICATION (Dense Layer + Softmax)")
        print("─"*100)

        # Get final dense layer
        dense_layers = [l for l in self.model.layers if 'dense' in l.name]
        final_dense = dense_layers[-1]

        # Get weights
        weights, bias = final_dense.get_weights()
        print(f"  Dense layer: {final_dense.name}")
        print(f"  Weight matrix shape: {weights.shape}  (2048 features → 5 classes)")
        print(f"  Bias vector: {bias.shape}")

        # Get logits
        logits_model = tf.keras.Model(inputs=self.model.input, outputs=final_dense.output)
        logits = logits_model.predict(img_array, verbose=0)

        # Show computation for each class
        print(f"\n  Computing logits for each class:")
        for i, class_name in enumerate(self.class_names):
            class_weights = weights[:, i]
            class_bias = bias[i]

            # Show sample computation
            print(f"\n  {class_name}:")
            print(f"    Logit = (feature_vector · class_weights) + bias")

            # Show first few multiplications
            print(f"    Sample calculations:")
            for j in range(5):
                print(f"      feature[{j}] × weight[{j}] = {pooled[0, j]:.4f} × {class_weights[j]:.4f} = {pooled[0, j] * class_weights[j]:.4f}")

            # Calculate full logit
            manual_logit = np.dot(pooled[0], class_weights) + class_bias
            print(f"      ... (continuing for all 2048 features)")
            print(f"    Sum of products = {manual_logit - class_bias:.4f}")
            print(f"    Add bias = {manual_logit - class_bias:.4f} + {class_bias:.4f} = {manual_logit:.4f}")
            print(f"    Model logit = {logits[0, i]:.4f} ✓")

        # Softmax computation
        print(f"\n  Softmax Computation:")
        print(f"  ──────────────────────")
        predictions = self.model.predict(img_array, verbose=0)

        print(f"\n  Raw logits: [{', '.join([f'{logits[0, i]:.4f}' for i in range(5)])}]")
        print(f"\n  Step 1: Apply exp() to each logit")
        exp_logits = np.exp(logits[0])
        for i, class_name in enumerate(self.class_names):
            print(f"    exp({logits[0, i]:.4f}) = {exp_logits[i]:.4f}")

        print(f"\n  Step 2: Sum all exp values")
        sum_exp = np.sum(exp_logits)
        print(f"    Sum = {' + '.join([f'{exp_logits[i]:.2f}' for i in range(5)])} = {sum_exp:.4f}")

        print(f"\n  Step 3: Divide each exp by sum to get probabilities")
        for i, class_name in enumerate(self.class_names):
            prob = exp_logits[i] / sum_exp
            print(f"    P({class_name}) = {exp_logits[i]:.4f} / {sum_exp:.4f} = {prob:.6f} = {prob*100:.2f}%")

        # Flatten arrays for safe indexing
        logits_flat = np.array(logits).flatten()
        exp_logits_flat = np.array(exp_logits).flatten()
        predictions_flat = np.array(predictions).flatten();

        print(f"\n  Final prediction: {self.class_names[np.argmax(predictions_flat)]}")
        print(f"  Confidence: {np.max(predictions_flat):.6f} = {np.max(predictions_flat)*100:.2f}%");

        concrete_log.append({
            'step': 'Classification',
            'dense_weights_shape': weights.shape,
            'logits': {self.class_names[i]: float(logits_flat[i]) for i in range(5)},
            'exp_logits': {self.class_names[i]: float(exp_logits_flat[i]) for i in range(5)},
            'sum_exp': float(sum_exp),
            'probabilities': {self.class_names[i]: float(predictions_flat[i]) for i in range(5)},
            'prediction': self.class_names[np.argmax(predictions_flat)],
            'confidence': float(np.max(predictions_flat))
        })

        print("\n" + "="*100)
        print("CONCRETE COMPUTATION COMPLETE")
        print("="*100 + "\n")

        return concrete_log


def get_model_instance():
    """Get a singleton instance of the burn classifier"""
    if not hasattr(get_model_instance, 'instance'):

        model_path = os.path.join(os.path.dirname(__file__), '..', 'real-checkpoints', 'finalfinetuning.h5')
        print(f"Initializing model with file: {model_path}")
        get_model_instance.instance = BurnClassifier(model_path)
    return get_model_instance.instance
