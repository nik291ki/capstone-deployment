import cv2
import numpy as np
from collections import Counter

class EnhancedFruitDetector:

    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.input_size = (299, 299)

    def detect_fruits_multiscale(self, image, confidence_threshold=0.6):
        """Main detection method using multi-scale sliding windows"""
        all_detections = []
        
        # Multiple window sizes for different fruit scales
        window_sizes = [(150, 150), (200, 200), (250, 250)]
        step_sizes = [75, 100, 125]
        
        # Run detection at each scale
        for window_size, step_size in zip(window_sizes, step_sizes):
            scale_detections = self._sliding_window_detection(
                image, window_size, step_size, confidence_threshold
            )
            all_detections.extend(scale_detections)
        
        # Remove overlapping detections
        final_detections = self._apply_nms(all_detections)
        
        return {
            'all_detections': final_detections,
            'detection_summary': self._summarize_detections(final_detections)
        }

    def _sliding_window_detection(self, image, window_size, step_size, confidence_threshold):
        """Sliding window detection for a specific window size"""
        detections = []
        window_w, window_h = window_size
        original_height, original_width = image.shape[:2]
        
        max_y = max(1, original_height - window_h + 1)
        max_x = max(1, original_width - window_w + 1)
        
        for y in range(0, max_y, step_size):
            for x in range(0, max_x, step_size):
                end_y = min(y + window_h, original_height)
                end_x = min(x + window_w, original_width)
                
                # Extract window
                window = image[y:end_y, x:end_x]
                
                # Resize to model input size if needed
                if window.shape[:2] != window_size:
                    window = cv2.resize(window, window_size)
                
                # Predict
                prediction_result = self._predict_window(window)
                
                # Check confidence threshold
                if prediction_result['confidence'] > confidence_threshold:
                    detection = {
                        'fruit_type': prediction_result['fruit_type'],
                        'freshness': prediction_result['freshness'],
                        'confidence': prediction_result['confidence'],
                        'original_class': prediction_result['original_class'],
                        'bbox': (x, y, end_x, end_y),
                        'area': (end_x - x) * (end_y - y)
                    }
                    detections.append(detection)
        
        return detections

    def _predict_window(self, window):
        """Make prediction on a single window"""
        # Resize to model's expected input size
        resized = cv2.resize(window, self.input_size)
        processed = resized.astype('float32')
        batch = np.expand_dims(processed, axis=0)
        
        # Predict
        predictions = self.model.predict(batch, verbose=0)
        confidence = np.max(predictions)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]
        
        # Parse class name
        if predicted_class.startswith('good_'):
            fruit_type = predicted_class.replace('good_', '')
            freshness = 'fresh'
        elif predicted_class.startswith('stale_'):
            fruit_type = predicted_class.replace('stale_', '')
            freshness = 'stale'
        else:
            fruit_type = predicted_class
            freshness = 'unknown'
        
        return {
            'fruit_type': fruit_type,
            'freshness': freshness,
            'confidence': float(confidence),
            'original_class': predicted_class
        }

    def _apply_nms(self, detections, iou_threshold=0.3):
        """Remove overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        
        for current_det in detections:
            should_keep = True
            
            for kept_det in final_detections:
                iou = self._calculate_iou(current_det['bbox'], kept_det['bbox'])
                
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                final_detections.append(current_det)
        
        return final_detections

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # No intersection
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        # Calculate areas
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def _summarize_detections(self, detections):
        """Create summary of detections"""
        if not detections:
            return {'total': 0}
        
        fruit_counts = Counter()
        freshness_counts = Counter()
        
        for det in detections:
            fruit_counts[det['fruit_type']] += 1
            freshness_counts[det['freshness']] += 1
        
        return {
            'total': len(detections),
            'fruit_types': dict(fruit_counts),
            'freshness_distribution': dict(freshness_counts)
        }