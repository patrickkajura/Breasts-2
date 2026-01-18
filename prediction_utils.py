import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_path, num_classes=2):
    """
    Load the trained breast cancer classification model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Define the model architecture
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)

    # Load the trained weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get class names from the checkpoint
    class_names = checkpoint.get('class_names', ['benign', 'malignant'])
    logger.info(f"Loaded model with classes: {class_names}")

    return model, class_names, device

def predict_image(image_path, model_path=None):
    """
    Predict the class of a single image

    Args:
        image_path (str): Path to the image file
        model_path (str, optional): Path to the model file. Defaults to best model.

    Returns:
        tuple: (predicted_class, confidence_percent) or (None, None) if error
    """
    if model_path is None:
        # Look for model files in the current directory
        possible_paths = [
            "best_breast_cancer_model.pth",
            "final_breast_cancer_model.pth",
            r"C:\Users\patri\Documents\Health ai\Multi Cancer\Breast Cancer\best_breast_cancer_model.pth",
            r"C:\Users\patri\Documents\Health ai\Multi Cancer\Breast Cancer\final_breast_cancer_model.pth"
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError("No trained model file found. Expected 'best_breast_cancer_model.pth' or 'final_breast_cancer_model.pth'")

    try:
        # Validate image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Validate image file type
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
        if not image_path.lower().endswith(valid_extensions):
            raise ValueError(f"Invalid image file type. Supported types: {valid_extensions}")

        # Load the model
        model, class_names, device = load_trained_model(model_path)
        model.eval()

        # Define the transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load and preprocess the image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Cannot open image file: {e}")

        image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = class_names[predicted.item()]
            confidence_percent = confidence.item() * 100

            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence_percent:.2f}%")

            # Print probabilities for each class
            print("\nClass probabilities:")
            for i, class_name in enumerate(class_names):
                prob = probabilities[0][i].item() * 100
                print(f"{class_name}: {prob:.2f}%")

            return predicted_class, confidence_percent

    except Exception as e:
        logger.error(f"Error in predict_image: {e}")
        print(f"Error predicting image: {e}")
        return None, None

def batch_predict(image_folder_path, model_path=None):
    """
    Predict classes for all images in a folder

    Args:
        image_folder_path (str): Path to the folder containing images
        model_path (str, optional): Path to the model file. Defaults to best model.

    Returns:
        dict: Dictionary with filename as key and prediction results as value
    """
    if model_path is None:
        # Look for model files in the current directory
        possible_paths = [
            "best_breast_cancer_model.pth",
            "final_breast_cancer_model.pth",
            r"C:\Users\patri\Documents\Health ai\Multi Cancer\Breast Cancer\best_breast_cancer_model.pth",
            r"C:\Users\patri\Documents\Health ai\Multi Cancer\Breast Cancer\final_breast_cancer_model.pth"
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError("No trained model file found. Expected 'best_breast_cancer_model.pth' or 'final_breast_cancer_model.pth'")

    try:
        # Validate folder exists
        if not os.path.isdir(image_folder_path):
            raise FileNotFoundError(f"Image folder not found: {image_folder_path}")

        # Load the model
        model, class_names, device = load_trained_model(model_path)
        model.eval()

        # Define the transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Supported image extensions
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

        results = {}
        failed_predictions = []

        for filename in os.listdir(image_folder_path):
            if filename.lower().endswith(supported_extensions):
                image_path = os.path.join(image_folder_path, filename)

                try:
                    # Load and preprocess the image
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(device)

                    # Make prediction
                    with torch.no_grad():
                        output = model(image_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)

                        predicted_class = class_names[predicted.item()]
                        confidence_percent = confidence.item() * 100

                        results[filename] = {
                            'predicted_class': predicted_class,
                            'confidence': confidence_percent
                        }
                except Exception as e:
                    logger.warning(f"Failed to predict {filename}: {e}")
                    failed_predictions.append((filename, str(e)))

        # Print results
        print(f"Batch prediction results for folder: {image_folder_path}")
        print("-" * 50)
        for filename, result in results.items():
            print(f"{filename}: {result['predicted_class']} ({result['confidence']:.2f}%)")

        if failed_predictions:
            print(f"\nFailed to predict {len(failed_predictions)} images:")
            for filename, error in failed_predictions:
                print(f"  {filename}: {error}")

        return results

    except Exception as e:
        logger.error(f"Error in batch_predict: {e}")
        print(f"Error in batch prediction: {e}")
        return {}

def validate_and_predict_pil(image, model_path=None):
    """
    Predict the class of a PIL Image object

    Args:
        image (PIL.Image): PIL Image object to predict
        model_path (str, optional): Path to the model file. Defaults to best model.

    Returns:
        dict: Dictionary with prediction results
    """
    if model_path is None:
        # Look for model files in the current directory
        possible_paths = [
            "best_breast_cancer_model.pth",
            "final_breast_cancer_model.pth",
            r"C:\Users\patri\Documents\Health ai\Multi Cancer\Breast Cancer\best_breast_cancer_model.pth",
            r"C:\Users\patri\Documents\Health ai\Multi Cancer\Breast Cancer\final_breast_cancer_model.pth"
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError("No trained model file found. Expected 'best_breast_cancer_model.pth' or 'final_breast_cancer_model.pth'")

    try:
        # Validate input
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image object")

        # Load the model
        model, class_names, device = load_trained_model(model_path)
        model.eval()

        # Define the transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = class_names[predicted.item()]
            confidence_percent = confidence.item() * 100

            # Return detailed results
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence_percent,
                'all_probabilities': {
                    class_names[i]: float(probabilities[0][i].item()) * 100
                    for i in range(len(class_names))
                },
                'predicted_class_index': predicted.item()
            }

            return result

    except Exception as e:
        logger.error(f"Error in validate_and_predict_pil: {e}")
        return {
            'error': str(e),
            'predicted_class': None,
            'confidence': 0.0,
            'all_probabilities': {},
            'predicted_class_index': -1
        }

if __name__ == "__main__":
    print("Breast Cancer Classification - Prediction Utility")
    print("="*50)

    # Example usage:
    # To predict a single image:
    # predict_image("path_to_your_image.jpg")

    # To predict all images in a folder:
    # batch_predict("path_to_your_image_folder")

    print("\nFunctions available:")
    print("1. predict_image(image_path, model_path) - Predict a single image")
    print("2. batch_predict(image_folder_path, model_path) - Predict all images in a folder")
    print("3. validate_and_predict_pil(image, model_path) - Predict a PIL Image object")
    print("\nModel files expected at:")
    print("- best_breast_cancer_model.pth (or final_breast_cancer_model.pth)")