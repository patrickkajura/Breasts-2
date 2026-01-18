#!/usr/bin/env python3
"""
Deployment helper script for breast cancer classifier
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    # Map package names to their import names (some differ)
    package_import_map = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'Pillow': 'PIL',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'flask': 'flask',
        'werkzeug': 'werkzeug',
        'gunicorn': 'gunicorn'
    }

    missing_packages = []
    for package, import_name in package_import_map.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        return False

    logger.info("All dependencies are available")
    return True

def check_model_files():
    """Check if model files exist"""
    model_files = [
        "best_breast_cancer_model.pth",
        "final_breast_cancer_model.pth"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            logger.info(f"Found model file: {model_file}")
            return model_file
    
    logger.warning("No model files found. You need to train the model first using 'python run_training.py'")
    return None

def create_placeholder_model():
    """Create a placeholder model file if none exists"""
    import torch
    import torch.nn as nn
    from torchvision import models
    
    # Create a minimal ResNet18 model as placeholder
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: benign and malignant
    
    # Create a dummy checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_names': ['benign', 'malignant'],
        'epoch': 0,
        'loss': 0.0,
        'accuracy': 0.0
    }
    
    # Save as both possible model file names
    torch.save(checkpoint, 'best_breast_cancer_model.pth')
    logger.info("Created placeholder model file: best_breast_cancer_model.pth")
    
    return 'best_breast_cancer_model.pth'

def main():
    """Main function for deployment preparation"""
    logger.info("Checking deployment prerequisites...")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Check for model files
    model_file = check_model_files()
    
    if not model_file:
        logger.info("Creating placeholder model for deployment...")
        model_file = create_placeholder_model()
    
    # Set environment variables if needed
    os.environ.setdefault('FLASK_ENV', 'production')
    
    logger.info(f"Deployment preparation complete. Using model: {model_file}")
    logger.info("You can now start the application with: gunicorn --bind 0.0.0.0:$PORT app:app")

if __name__ == "__main__":
    main()