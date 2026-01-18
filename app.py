"""
Main application for breast cancer classification using PyTorch model
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import sys
from flask import Flask, request, jsonify, render_template_string
import io
import base64
import numpy as np
import logging
from werkzeug.utils import secure_filename


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class BreastCancerClassifier:
    def __init__(self, model_path=None):
        """
        Initialize the classifier with a trained model
        """
        if model_path is None:
            # Look for the best model in the current directory
            possible_paths = [
                "best_breast_cancer_model.pth",
                "final_breast_cancer_model.pth",
                r"C:\Users\patri\Documents\Health_ai\Multi_Cancer\Breast Cancer\best_breast_cancer_model.pth",
                r"C:\Users\patri\Documents\Health_ai\Multi_Cancer\Breast Cancer\final_breast_cancer_model.pth"
            ]

            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    logger.info(f"Found model at: {path}")
                    break

            if model_path is None:
                # Check for common model file names in the current directory and subdirectories
                import glob
                model_candidates = glob.glob("**/*model*.pth", recursive=True) + \
                                  glob.glob("**/*model*.pt", recursive=True) + \
                                  glob.glob("**/*checkpoint*.pth", recursive=True)

                if model_candidates:
                    model_path = model_candidates[0]  # Use the first found model
                    logger.info(f"Found alternative model at: {model_path}")
                else:
                    raise FileNotFoundError("No trained model file found. Expected 'best_breast_cancer_model.pth' or 'final_breast_cancer_model.pth'. "
                                          "Run 'python run_training.py' to train the model first.")

        self.model_path = model_path
        self.model = self._load_model()
        self.transform = self._get_transform()

    def _get_transform(self):
        """Define the image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        """Load the trained model from checkpoint"""
        try:
            checkpoint = torch.load(self.model_path, map_location=device)
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            raise

        # Recreate the model architecture (ResNet18 with custom classifier)
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)  # 2 classes: benign and malignant
        model = model.to(device)

        # Load the trained weights
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            logger.error(f"Error loading model state dict: {e}")
            raise

        model.eval()

        # Store class names
        self.class_names = checkpoint.get('class_names', ['benign', 'malignant'])
        logger.info(f"Model loaded successfully with classes: {self.class_names}")

        return model
    
    def predict(self, image_path_or_pil):
        """
        Predict the class of an image

        Args:
            image_path_or_pil: Either a file path or PIL Image object

        Returns:
            dict: Prediction results with class and confidence
        """
        try:
            # Handle both file paths and PIL Images
            if isinstance(image_path_or_pil, str):
                # Validate file exists
                if not os.path.exists(image_path_or_pil):
                    raise FileNotFoundError(f"Image file not found: {image_path_or_pil}")

                # Validate file extension
                valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
                if not image_path_or_pil.lower().endswith(valid_extensions):
                    raise ValueError(f"Invalid image file type. Supported types: {valid_extensions}")

                try:
                    image = Image.open(image_path_or_pil).convert('RGB')
                except Exception as e:
                    raise ValueError(f"Cannot open image file: {e}")
            elif isinstance(image_path_or_pil, Image.Image):
                image = image_path_or_pil.convert('RGB')
            else:
                raise ValueError("Input must be a file path string or PIL Image object")

            # Preprocess the image
            image_tensor = self.transform(image).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                predicted_class_idx = predicted.item()
                predicted_class = self.class_names[predicted_class_idx]
                confidence_percent = confidence.item() * 100

                # Get probabilities for all classes
                all_probabilities = probabilities[0].cpu().numpy()

                result = {
                    'predicted_class': predicted_class,
                    'confidence': confidence_percent,
                    'class_probabilities': {
                        self.class_names[i]: float(prob) * 100
                        for i, prob in enumerate(all_probabilities)
                    },
                    'predicted_class_index': predicted_class_idx
                }

                logger.info(f"Prediction successful: {predicted_class} with {confidence_percent:.2f}% confidence")
                return result
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                'error': str(e),
                'predicted_class': 'unknown',
                'confidence': 0.0,
                'class_probabilities': {}
            }


# Global classifier instance
classifier = None


def create_app():
    """Create and configure the Flask app"""
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        """Home page with upload form"""
        html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Breast Cancer Classification</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 600px; margin: 0 auto; }
                .upload-form { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
                .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
                .success { background-color: #d4edda; border: 1px solid #c3e6cb; }
                .error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
                button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
                button:hover { background-color: #0069d9; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Breast Cancer Classification</h1>
                <p>Upload an image to classify it as benign or malignant</p>
                
                <form class="upload-form" action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="image" accept="image/*" required>
                    <br><br>
                    <button type="submit">Classify Image</button>
                </form>
                
                {% if result %}
                    <div class="result {{'success' if not result.error else 'error'}}">
                        {% if result.error %}
                            <h3>Error:</h3>
                            <p>{{ result.error }}</p>
                        {% else %}
                            <h3>Prediction Result:</h3>
                            <p><strong>Predicted Class:</strong> {{ result.predicted_class }}</p>
                            <p><strong>Confidence:</strong> {{ "%.2f"|format(result.confidence) }}%</p>
                            
                            <h4>Class Probabilities:</h4>
                            <ul>
                                {% for class_name, prob in result.class_probabilities.items() %}
                                    <li>{{ class_name }}: {{ "%.2f"|format(prob) }}%</li>
                                {% endfor %}
                            </ul>
                        {% endif %}
                    </div>
                {% endif %}
            </div>
        </body>
        </html>
        '''
        return render_template_string(html)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Handle image prediction request"""
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            logger.warning("Empty filename in request")
            return jsonify({'error': 'No image selected'}), 400

        try:
            # Secure the filename
            filename = secure_filename(file.filename)
            if not filename:
                logger.warning("Invalid filename")
                return jsonify({'error': 'Invalid filename'}), 400

            # Validate file extension
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
            if not filename.lower().endswith(valid_extensions):
                logger.warning(f"Invalid file type: {filename}")
                return jsonify({'error': f'Invalid file type. Supported types: {valid_extensions}'}), 400

            # Convert file to PIL Image
            image = Image.open(file.stream).convert('RGB')

            # Initialize classifier if not already done
            initialize_classifier()
            # Make prediction
            result = classifier.predict(image)

            return render_template_string(
                '''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Prediction Result</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        .container { max-width: 600px; margin: 0 auto; }
                        .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
                        .success { background-color: #d4edda; border: 1px solid #c3e6cb; }
                        .error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
                        a { color: #007bff; text-decoration: none; }
                        a:hover { text-decoration: underline; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <a href="/">← Back to Home</a>

                        <h1>Prediction Result</h1>

                        <div class="result {{'success' if not result.error else 'error'}}">
                            {% if result.error %}
                                <h3>Error:</h3>
                                <p>{{ result.error }}</p>
                            {% else %}
                                <h3>Predicted Class: {{ result.predicted_class }}</h3>
                                <p><strong>Confidence:</strong> {{ "%.2f"|format(result.confidence) }}%</p>

                                <h4>Class Probabilities:</h4>
                                <ul>
                                    {% for class_name, prob in result.class_probabilities.items() %}
                                        <li>{{ class_name }}: {{ "%.2f"|format(prob) }}%</li>
                                    {% endfor %}
                                </ul>
                            {% endif %}
                        </div>
                    </div>
                </body>
                </html>
                ''',
                result=result
            )
        except Exception as e:
            logger.error(f"Error in /predict route: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        """API endpoint for programmatic access"""
        if 'image' not in request.files:
            logger.warning("No image file in API request")
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            logger.warning("Empty filename in API request")
            return jsonify({'error': 'No image selected'}), 400

        try:
            # Secure the filename
            filename = secure_filename(file.filename)
            if not filename:
                logger.warning("Invalid filename in API request")
                return jsonify({'error': 'Invalid filename'}), 400

            # Validate file extension
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
            if not filename.lower().endswith(valid_extensions):
                logger.warning(f"Invalid file type in API request: {filename}")
                return jsonify({'error': f'Invalid file type. Supported types: {valid_extensions}'}), 400

            # Convert file to PIL Image
            image = Image.open(file.stream).convert('RGB')

            # Initialize classifier if not already done
            initialize_classifier()
            # Make prediction
            result = classifier.predict(image)

            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in /api/predict route: {e}")
            return jsonify({'error': str(e)}), 500
    
    return app


def main():
    """Main function to run the application"""
    global classifier

    print("Loading breast cancer classifier...")
    try:
        initialize_classifier()
        print(f"Model loaded successfully from: {classifier.model_path}")
        print(f"Classes: {classifier.class_names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Check if running in development mode
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        # CLI mode - for testing individual images
        if len(sys.argv) < 3:
            print("Usage: python app.py --cli <image_path>")
            sys.exit(1)
        
        image_path = sys.argv[2]
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            sys.exit(1)
        
        # Initialize classifier if not already done
        initialize_classifier()
        print(f"Predicting image: {image_path}")
        result = classifier.predict(image_path)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print("Class probabilities:")
            for class_name, prob in result['class_probabilities'].items():
                print(f"  {class_name}: {prob:.2f}%")
    else:
        # Web mode - start Flask server
        print("Starting web server...")
        app = create_app()

        # Use PORT environment variable if available (for deployment)
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port, debug=False)


# Global classifier instance - initialize as None initially
classifier = None

def initialize_classifier():
    """Initialize the classifier - called when first needed"""
    global classifier
    if classifier is None:
        try:
            classifier = BreastCancerClassifier()
            print("Model loaded successfully for inference")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

# For production deployment with Gunicorn
app = create_app()

if __name__ == "__main__":
    main()