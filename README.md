# Alzheimer's MRI Classification Using ResNet18

## Overview

This project is an educational deep learning application that classifies brain MRI images into four dataset categories using a pretrained ResNet18 model. It uses PyTorch for model inference, Streamlit for the web interface, and Grad-CAM to visualize image regions influencing model predictions.

## Features

- MRI image upload through a web interface
- Four-class image classification
- Prediction confidence scores
- Grad-CAM heatmap visualization
- Original MRI and heatmap overlay
- PyTorch-based deep learning model

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- Streamlit
- Grad-CAM
- NumPy
- Matplotlib
- Pillow
## Dataset

This project uses the **Alzheimer MRI Disease Classification Dataset** from Hugging Face.

- **Source:** https://huggingface.co/datasets/Falah/Alzheimer_MRI
- **Total images:** 6,400
- **Training split:** 5,120 images
- **Test split:** 1,280 images
- **Format:** Parquet
- **Task:** Four-class MRI image classification
- **License listed on dataset page:** Apache-2.0

### Dataset Classes

| Label | Class |
|---|---|
| 0 | Mild_Demented |
| 1 | Moderate_Demented |
| 2 | Non_Demented |
| 3 | Very_Mild_Demented |

### Download and Usage

Download `train.parquet` and `test.parquet` from the dataset source linked above. Place them in the `/content/data/` directory when running the training notebook in Google Colab.

The application uses the trained model weights, so the original dataset files are not required to run predictions.

### Citation

Falah G. Salieh. *Alzheimer MRI Dataset*. Hugging Face, 2023, version 1.0.

Dataset URL: https://huggingface.co/datasets/Falah/Alzheimer_MRI

### Responsible Use

This dataset is intended for research and educational experimentation. The resulting model is not clinically validated and must not be used as a substitute for professional medical assessment.


## Project Structure

```text
alzheimer-mri-classification/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── best_resnet18.pth
└── screenshots/
```

The trained model file is included only if it is available and permitted to redistribute.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/vikramkamishetti/alzheimer-mri-classification.git
   cd alzheimer-mri-classification
   ```

2. Create and activate a virtual environment:

   Windows PowerShell:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place `best_resnet18.pth` in the project root directory if it is not included in the repository.
5. ## Download the Trained Model

The trained ResNet18 model weights are hosted on Google Drive.

[Download best_resnet18.pth](https://drive.google.com/file/d/1elrbkNcUzAz60QpUGl5m0HntseShsRhe/view?usp=sharing)

**Setup:** Download the model file and place it in the same directory as `app.py` before running the application.

The model is provided for educational and research purposes only and is not intended for clinical diagnosis.


6. Run the application:

   ```bash
   python -m streamlit run app.py
   ```

7. Open the local URL displayed in the terminal.

## Model and Explainability

The application uses ResNet18 to produce class predictions. Grad-CAM generates a heatmap that helps inspect which image regions influenced a prediction.

Model confidence is not a calibrated probability of disease, and Grad-CAM does not establish clinical reasoning.

## Limitations

- This project is intended for education and research.
- The model has not been clinically validated.
- Predictions may be incorrect, including on images that differ from the training data.
- Results must not be used for medical diagnosis or treatment decisions.

