import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt

class YOLOBurnClassifier:
    """
    Wrapper for YOLO11-seg model to provide segmentation-based burn classification.
    Compatible with the existing Flask app interface.
    """

    def __init__(self, model_path=None):
        """
        Initialize YOLO model for burn classification

        Args:
            model_path: Path to YOLO weights file (.pt)
        """
        if model_path is None:
            # Default path to your trained model (YOLO11m with Fourth Degree support)
            model_path = r"C:\Users\ksan\Documents\thesis\automated-thermal-burn-severity-classifyer\runs\segment\train29\weights\best.pt"

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading YOLO11m-seg model from {model_path}...")
        self.model = YOLO(str(self.model_path))

        # Class names
        self.class_names = {
            0: "First Degree Burn",
            1: "Second Degree Burn",
            2: "Third Degree Burn",
            3: "Fourth Degree Burn"
        }

        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        print(f"Model loaded successfully!")

    def predict(self, image_path, with_gradcam=False, with_computational_flow=False,
                detailed_analysis=False, show_concrete_math=False, trace_image=False,
                progress_callback=None, conf_threshold=0.25):
        """
        Predict burn severity from image using YOLO segmentation

        Args:
            image_path: Path to input image
            with_gradcam: Generate attention visualization (using segmentation masks)
            with_computational_flow: Generate computational flow diagram
            detailed_analysis: Include detailed per-class analysis
            show_concrete_math: Show mathematical computations
            trace_image: Trace image through model layers
            progress_callback: Callback function for progress updates
            conf_threshold: Confidence threshold for predictions

        Returns:
            (result_dict, error_message)
        """
        try:
            if progress_callback:
                progress_callback("Loading image...", 10)

            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None, "Failed to load image"

            if progress_callback:
                progress_callback("Preprocessing image with CLAHE...", 20)

            # Apply CLAHE preprocessing (LAB color space)
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Apply CLAHE to L channel (lightness)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel_clahe = clahe.apply(l_channel)

            # Merge channels back
            lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])

            # Convert back to BGR
            image_preprocessed = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

            # Save preprocessed image temporarily for YOLO inference
            temp_path = str(Path(image_path).parent / f"temp_clahe_{Path(image_path).name}")
            cv2.imwrite(temp_path, image_preprocessed)

            if progress_callback:
                progress_callback("Running YOLO segmentation...", 30)

            # Run YOLO inference on preprocessed image
            results = self.model.predict(
                source=temp_path,
                imgsz=640,
                conf=conf_threshold,
                iou=0.6,
                device=self.device,
                verbose=False
            )

            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass

            if progress_callback:
                progress_callback("Processing results...", 60)

            # Get first result (single image)
            result = results[0]

            # Check if any detections were made
            if len(result.boxes) == 0:
                return {
                    'error': 'INVALID_BURN_IMAGE',
                    'message': 'No burn regions detected in the image. Please ensure the image contains visible burn injuries.',
                    'validation_details': {
                        'detections_found': 0,
                        'suggestion': 'Try with a clearer image showing burn areas'
                    }
                }, None

            # Aggregate predictions across all detected regions
            class_votes = {}
            class_confidences = {}
            total_area = 0

            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Get box area as weight
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)

                # Weight vote by confidence and area
                weight = confidence * area

                if class_id not in class_votes:
                    class_votes[class_id] = 0
                    class_confidences[class_id] = []

                class_votes[class_id] += weight
                class_confidences[class_id].append(confidence)
                total_area += area

            # Determine final prediction
            if class_votes:
                # Normalize votes by total area
                class_probabilities = {
                    cls: vote / total_area if total_area > 0 else 0
                    for cls, vote in class_votes.items()
                }

                # Get predicted class
                predicted_class_id = max(class_probabilities, key=class_probabilities.get)
                confidence = class_probabilities[predicted_class_id]

                # Create probability distribution for all classes
                all_probabilities = {}
                for class_id in [0, 1, 2, 3]:
                    prob = class_probabilities.get(class_id, 0.0)
                    all_probabilities[self.class_names[class_id]] = float(prob)
            else:
                # No valid detections
                return {
                    'error': 'INVALID_BURN_IMAGE',
                    'message': 'No burn regions detected with sufficient confidence.',
                    'validation_details': {
                        'detections_found': 0
                    }
                }, None

            if progress_callback:
                progress_callback("Generating visualizations...", 80)

            # Create result dictionary
            result_dict = {
                'predicted_class': self.class_names[predicted_class_id],
                'class_id': int(predicted_class_id),
                'confidence': float(confidence),
                'all_probabilities': all_probabilities,
                'num_detections': len(result.boxes),
                'detection_details': [],
                # Add for realtime detection compatibility
                'boxes': [],
                'masks': [],
                'confidences': [],
                'class_ids': []
            }

            # Add detailed detection info
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                result_dict['detection_details'].append({
                    'region_id': i + 1,
                    'class': self.class_names[class_id],
                    'class_id': int(class_id),
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'area': float((x2 - x1) * (y2 - y1))
                })

                # Add to arrays for realtime detection
                result_dict['boxes'].append([float(x1), float(y1), float(x2), float(y2)])
                result_dict['confidences'].append(float(conf))
                result_dict['class_ids'].append(int(class_id))

                # Add mask if available
                if result.masks is not None and i < len(result.masks):
                    mask = result.masks.data[i].cpu().numpy()
                    # Convert mask to polygon points for frontend
                    mask_resized = cv2.resize(mask, (int(x2 - x1), int(y2 - y1)))
                    # Flatten mask to binary and get contour points
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Get largest contour
                        contour = max(contours, key=cv2.contourArea)
                        # Convert to list of points (offset by box position)
                        points = []
                        for point in contour:
                            points.append(float(point[0][0] + x1))
                            points.append(float(point[0][1] + y1))
                        result_dict['masks'].append(points)
                    else:
                        result_dict['masks'].append(None)
                else:
                    result_dict['masks'].append(None)

            # Generate segmentation visualizations (3 versions) - ALWAYS generate for toggle feature
            # Generate all three visualizations
            overlay_img, mask_only_img = self._generate_segmentation_visualizations(image, result)
            confidence_heatmap = self._generate_confidence_heatmap(image, result)
            pixel_intensity_analysis = self._analyze_pixel_intensity(image, result)

            result_dict['segmentation_overlay'] = overlay_img
            result_dict['segmentation_mask_only'] = mask_only_img
            result_dict['segmentation_original'] = image.copy()
            result_dict['confidence_heatmap'] = confidence_heatmap
            result_dict['pixel_intensity_analysis'] = pixel_intensity_analysis

            # Keep backward compatibility
            if with_gradcam:
                result_dict['gradcam_overlay'] = overlay_img

            # Add computational flow (if requested)
            if with_computational_flow:
                result_dict['computational_flow'] = self._generate_computational_flow(result_dict)

            # Add detailed analysis (if requested)
            if detailed_analysis:
                result_dict['detailed_analysis'] = self._generate_detailed_analysis(result_dict)

            # Add concrete computations (if requested)
            if show_concrete_math:
                result_dict['concrete_computations'] = self._generate_concrete_math(
                    class_votes, total_area, class_confidences
                )

            # Add image trace (if requested)
            if trace_image:
                result_dict['image_trace'] = self._generate_image_trace(image, result)

            if progress_callback:
                progress_callback("Complete!", 100)

            return result_dict, None

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Prediction error: {str(e)}"

    def _generate_segmentation_visualizations(self, image, result):
        """
        Generate segmentation visualizations:
        1. Overlay: masks blended with original image + boxes + labels
        2. Mask Only: colored masks on black background

        Returns:
            tuple: (overlay_image, mask_only_image)
        """
        overlay = image.copy()
        mask_only = np.zeros_like(image)  # Black background for mask-only view

        # Color map for each burn class (BGR format for OpenCV)
        colors = {
            0: (100, 200, 255),  # Light orange for First Degree
            1: (50, 150, 255),   # Orange for Second Degree
            2: (50, 50, 255),    # Red for Third Degree
            3: (0, 0, 139)       # Dark red for Fourth Degree
        }

        # Draw segmentation masks
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes

            for i, (mask, box) in enumerate(zip(masks, boxes)):
                class_id = int(box.cls[0])
                color = colors.get(class_id, (255, 255, 255))

                # Resize mask to image size
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_binary = mask_resized > 0.5

                # Create colored mask
                colored_mask = np.zeros_like(image)
                colored_mask[mask_binary] = color

                # Blend with original image for overlay
                overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

                # Add to mask-only view with full opacity
                mask_only[mask_binary] = color

        # Draw bounding boxes and labels on overlay
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            color = colors.get(class_id, (255, 255, 255))

            # Draw box on overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Add label on overlay
            label = f"{self.class_names[class_id]}: {conf:.2f}"

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Draw label background
            cv2.rectangle(
                overlay,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                overlay, label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                2
            )

            # Draw box on mask-only (thinner, for reference)
            cv2.rectangle(mask_only, (x1, y1), (x2, y2), color, 1)

        return overlay, mask_only

    def _generate_confidence_heatmap(self, image, result):
        """
        Generate a per-pixel confidence heatmap visualization.
        Maps mask confidence values to a color gradient (cool to warm).

        Returns:
            numpy.ndarray: Confidence heatmap image with colorbar
        """
        from matplotlib import cm

        # Create confidence map (start with zeros)
        confidence_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes

            for i, (mask, box) in enumerate(zip(masks, boxes)):
                conf = float(box.conf[0])  # Detection confidence

                # Resize mask to image size
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

                # For each pixel in the mask, use: mask_probability * detection_confidence
                # This gives per-pixel confidence considering both mask certainty and detection confidence
                pixel_confidence = mask_resized * conf

                # Take maximum confidence where masks overlap
                confidence_map = np.maximum(confidence_map, pixel_confidence)

        # Apply colormap (jet: blue=low, red=high confidence)
        # Normalize to 0-1 range
        confidence_normalized = confidence_map

        # Convert to colormap
        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(confidence_normalized)

        # Convert from 0-1 float to 0-255 uint8
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        heatmap_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)

        # Create figure with colorbar
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        im = ax.imshow(confidence_map, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        ax.axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Confidence', rotation=270, labelpad=20, fontsize=12)

        # Convert matplotlib figure to image using buffer
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        width, height = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(height, width, 4)

        # Convert RGBA to RGB then to BGR for OpenCV
        heatmap_with_colorbar = cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)

        plt.close(fig)

        return heatmap_with_colorbar

    def _analyze_pixel_intensity(self, image, result):
        """
        Analyze RGB pixel intensity values within segmented burn regions.

        Returns:
            dict: Statistical analysis with RGB histogram data
        """
        analysis = {
            'per_class': {},
            'overall': {},
            'histogram_data': {
                'red': [],
                'green': [],
                'blue': []
            }
        }

        if result.masks is None:
            return analysis

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes

        # Collect RGB pixel values from all burn regions
        all_red = []
        all_green = []
        all_blue = []
        all_confidences = []

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            conf = float(box.conf[0])

            # Resize mask to image size
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask_binary = mask_resized > 0.5

            # Extract RGB pixels within the mask (BGR format in OpenCV)
            masked_pixels = image[mask_binary]

            if len(masked_pixels) > 0:
                # Separate BGR channels (OpenCV format)
                blue_channel = masked_pixels[:, 0]
                green_channel = masked_pixels[:, 1]
                red_channel = masked_pixels[:, 2]

                # Calculate statistics using grayscale equivalent for overall metrics
                gray_equivalent = (0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel)

                stats = {
                    'class_name': class_name,
                    'detection_confidence': float(conf),
                    'pixel_count': int(np.sum(mask_binary)),
                    'mean_intensity': float(np.mean(gray_equivalent)),
                    'std_intensity': float(np.std(gray_equivalent)),
                    'min_intensity': float(np.min(gray_equivalent)),
                    'max_intensity': float(np.max(gray_equivalent)),
                    'median_intensity': float(np.median(gray_equivalent)),
                    'percentile_25': float(np.percentile(gray_equivalent, 25)),
                    'percentile_75': float(np.percentile(gray_equivalent, 75))
                }

                # Group by class
                if class_name not in analysis['per_class']:
                    analysis['per_class'][class_name] = []
                analysis['per_class'][class_name].append(stats)

                # Collect RGB values for histogram
                all_red.extend(red_channel.tolist())
                all_green.extend(green_channel.tolist())
                all_blue.extend(blue_channel.tolist())
                all_confidences.extend([conf] * len(masked_pixels))

        # Overall statistics
        if len(all_red) > 0:
            # Calculate grayscale equivalent for overall stats
            all_gray = [0.299 * r + 0.587 * g + 0.114 * b
                       for r, g, b in zip(all_red, all_green, all_blue)]

            analysis['overall'] = {
                'total_burn_pixels': len(all_red),
                'mean_intensity': float(np.mean(all_gray)),
                'std_intensity': float(np.std(all_gray)),
                'min_intensity': float(np.min(all_gray)),
                'max_intensity': float(np.max(all_gray)),
                'median_intensity': float(np.median(all_gray)),
                'mean_confidence': float(np.mean(all_confidences))
            }

            # Create RGB histogram data (15 bins for better visualization)
            bins = 15
            hist_r, bin_edges_r = np.histogram(all_red, bins=bins, range=(0, 255))
            hist_g, bin_edges_g = np.histogram(all_green, bins=bins, range=(0, 255))
            hist_b, bin_edges_b = np.histogram(all_blue, bins=bins, range=(0, 255))

            # Store histogram data for each channel
            analysis['histogram_data']['red'] = [
                {
                    'bin_start': float(bin_edges_r[i]),
                    'bin_end': float(bin_edges_r[i+1]),
                    'count': int(hist_r[i])
                }
                for i in range(len(hist_r))
            ]

            analysis['histogram_data']['green'] = [
                {
                    'bin_start': float(bin_edges_g[i]),
                    'bin_end': float(bin_edges_g[i+1]),
                    'count': int(hist_g[i])
                }
                for i in range(len(hist_g))
            ]

            analysis['histogram_data']['blue'] = [
                {
                    'bin_start': float(bin_edges_b[i]),
                    'bin_end': float(bin_edges_b[i+1]),
                    'count': int(hist_b[i])
                }
                for i in range(len(hist_b))
            ]

        return analysis

    def _generate_computational_flow(self, result_dict):
        """Generate computational flow information"""
        return {
            'model_architecture': 'YOLO11n-seg',
            'input_size': '640x640',
            'backbone': 'CSPDarknet',
            'neck': 'PANet',
            'head': 'Segmentation Head',
            'num_parameters': '~2.5M',
            'inference_steps': [
                'Image preprocessing and resizing',
                'Feature extraction (backbone)',
                'Multi-scale feature fusion (neck)',
                'Detection and segmentation head',
                'Non-maximum suppression (NMS)',
                'Mask generation',
                'Class aggregation'
            ],
            'num_detections': result_dict['num_detections']
        }

    def _generate_detailed_analysis(self, result_dict):
        """Generate detailed per-class analysis"""
        analysis = {
            'total_regions_detected': result_dict['num_detections'],
            'final_prediction': result_dict['predicted_class'],
            'aggregation_method': 'Confidence-weighted voting by detection area',
            'per_class_breakdown': []
        }

        for class_name, probability in result_dict['all_probabilities'].items():
            analysis['per_class_breakdown'].append({
                'class': class_name,
                'final_probability': probability,
                'confidence_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
            })

        return analysis

    def _generate_concrete_math(self, class_votes, total_area, class_confidences):
        """Generate concrete mathematical computations"""
        computations = {
            'total_detection_area': float(total_area),
            'class_weighted_votes': {},
            'normalization_process': []
        }

        for class_id, vote in class_votes.items():
            class_name = self.class_names[class_id]
            avg_conf = np.mean(class_confidences[class_id])

            computations['class_weighted_votes'][class_name] = {
                'raw_vote': float(vote),
                'normalized_probability': float(vote / total_area if total_area > 0 else 0),
                'average_confidence': float(avg_conf),
                'num_regions': len(class_confidences[class_id])
            }

            computations['normalization_process'].append(
                f"{class_name}: {vote:.2f} / {total_area:.2f} = {vote/total_area:.4f}"
            )

        return computations

    def _generate_image_trace(self, image, result):
        """Generate image processing trace"""
        return {
            'original_size': f"{image.shape[1]}x{image.shape[0]}",
            'preprocessing': [
                'Resize to 640x640',
                'Normalize pixel values',
                'Convert BGR to RGB'
            ],
            'detections': len(result.boxes),
            'has_segmentation_masks': result.masks is not None,
            'num_masks': len(result.masks) if result.masks is not None else 0
        }


# Singleton instance
_yolo_instance = None

def get_yolo_instance():
    """Get or create YOLO model instance (singleton pattern)"""
    global _yolo_instance
    if _yolo_instance is None:
        _yolo_instance = YOLOBurnClassifier()
    return _yolo_instance
