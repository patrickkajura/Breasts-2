# Breast Cancer Classification - Web Application Deployment Guide

This guide provides instructions for deploying the breast cancer classification model to a web application, including deployment to Render.

## Project Structure

```
Breast Cancer/
├── app.py                 # Main Flask application
├── prediction_utils.py    # Enhanced prediction utilities
├── best_breast_cancer_model.pth    # Trained model (required)
├── final_breast_cancer_model.pth   # Alternative trained model
├── requirements.txt       # Production dependencies
├── requirements_prod.txt  # Production dependencies
├── DEPLOYMENT_README.md   # This file
├── README.md             # Original project README
├── run_training.py       # Training script
├── visualize_results.py  # Visualization utilities
├── breast_cancer_model.py # Model definition
├── breast_benign/        # Benign images (for reference)
└── breast_malignant/     # Malignant images (for reference)
```

## Local Development Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Steps
1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the trained model files:
   - `best_breast_cancer_model.pth` (or `final_breast_cancer_model.pth`)
4. Run the application:
   ```bash
   python app.py
   ```
5. Open your browser and navigate to `http://localhost:5000`

## Web API Endpoints

### Web Interface
- `GET /` - Main interface for uploading and classifying images
- `POST /predict` - Process uploaded image and return classification

### API Endpoints
- `POST /api/predict` - Programmatic access to the model
  - Accepts image file in multipart/form-data
  - Returns JSON with prediction results

## API Response Format

```json
{
  "predicted_class": "benign",
  "confidence": 92.45,
  "class_probabilities": {
    "benign": 92.45,
    "malignant": 7.55
  },
  "predicted_class_index": 0
}
```

## Deployment to Render

### Prerequisites
- A Render account (https://render.com)
- Git repository with your code

### Steps

1. **Prepare your repository:**
   - Ensure all required files are in your Git repository
   - Make sure `best_breast_cancer_model.pth` or `final_breast_cancer_model.pth` is included
   - Include `requirements.txt` in the root directory

2. **Create a Render Web Service:**
   - Log in to your Render dashboard
   - Click "New +" and select "Web Service"
   - Connect your Git repository (GitHub, GitLab, or Bitbucket)

3. **Configure the Web Service:**
   - **Environment:** Python
   - **Branch:** main (or your default branch)
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     gunicorn --bind 0.0.0.0:$PORT app:app
     ```
   - **Runtime:** Choose Python version 3.8 or higher

4. **Set Environment Variables (optional):**
   - If needed, add environment variables in the Render dashboard

5. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application
   - Monitor the build logs to ensure everything deploys correctly

## Alternative Deployment Options

### Using Docker
1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

2. Build and run:
```bash
docker build -t breast-cancer-classifier .
docker run -p 5000:5000 breast-cancer-classifier
```

### Heroku Deployment
1. Create a `Procfile`:
```
web: gunicorn app:app
```

2. Deploy using Heroku CLI:
```bash
heroku create
git push heroku main
```

## Model File Requirements

The application expects one of these model files to be present:
- `best_breast_cancer_model.pth`
- `final_breast_cancer_model.pth`

If you need to retrain the model, run:
```bash
python run_training.py
```

## Troubleshooting

### Common Issues

1. **Model file not found:**
   - Ensure the trained model file is in the root directory
   - Check the file name matches what the application expects

2. **Memory issues:**
   - The model requires GPU/CPU resources to run
   - On Render, consider upgrading to a higher-tier instance if needed

3. **Import errors:**
   - Verify all dependencies are listed in requirements.txt
   - Ensure Python version compatibility

4. **Port binding issues:**
   - The application reads the PORT environment variable
   - Make sure your deployment service sets this variable

### Logging
The application logs important events to help with debugging:
- Model loading success/failure
- Prediction requests and results
- Error conditions

## Security Considerations

- Input validation is performed on uploaded files
- File type checking prevents malicious uploads
- Use HTTPS in production deployments
- Validate and sanitize all user inputs

## Performance Optimization

- The model is optimized for inference
- Image preprocessing is handled efficiently
- Consider using a CDN for static assets in production

## Updating the Model

To update the deployed model:
1. Retrain the model using `run_training.py`
2. Upload the new `.pth` file to your repository
3. Redeploy the application

## Support

For issues with deployment:
1. Check the deployment logs in your hosting platform
2. Verify all required files are present
3. Ensure dependencies are correctly specified
4. Contact support for your hosting platform if needed