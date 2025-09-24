# Burn Severity Classifier Webapp API Documentation

## Overview
This webapp provides endpoints for burn severity classification using a trained deep learning model. You can interact via a browser or programmatically via the API.

---

## Endpoints

### 1. Home Page
- **URL:** `/`
- **Method:** `GET`
- **Description:** Displays the home page with a file upload form.
- **Response:** Renders `index.html`.

---

### 2. File Upload and Prediction
- **URL:** `/upload`
- **Method:** `POST`
- **Description:** Handles file upload from the web form, saves the image, runs prediction, and displays the result.
- **Request:**
  - Form-data with key `file` containing the image file.
- **Response:**
  - On success: Renders `result.html` with prediction and image.
  - On error: Redirects to home page with error message.
- **Errors:**
  - No file selected
  - Invalid file type
  - Prediction error
  - File too large (handled by error handler)

---

### 3. API Prediction
- **URL:** `/api/predict`
- **Method:** `POST`
- **Description:** API endpoint for programmatic predictions. Accepts an image file and returns prediction as JSON.
- **Request:**
  - Form-data with key `file` containing the image file.
- **Response:**
  - On success: JSON object with prediction result:
    ```json
    {
      "predicted_class": "First Degree Burn",
      "class_id": 0,
      "confidence": 0.95,
      "all_probabilities": {
        "First Degree Burn": 0.95,
        "Second Degree Burn": 0.03,
        "Third Degree Burn": 0.02
      }
    }
    ```
  - On error: JSON object with error message and appropriate HTTP status code.
- **Errors:**
  - No file provided (`400`)
  - Invalid file type (`400`)
  - Prediction error (`500`)
  - File too large (handled by error handler)

---

### 4. File Too Large Error Handler
- **URL:** `@app.errorhandler(413)`
- **Description:** Handles requests where the uploaded file exceeds the maximum allowed size (16MB).
- **Response:** Redirects to home page with error message.

---

## General Notes
- **Allowed file types:** PNG, JPG, JPEG, GIF, WEBP
- **Max file size:** 16MB
- **Upload folder:** `static/uploads`
- **Model:** Uses a pre-trained hybrid ResNet-Transformer model for burn severity classification.
