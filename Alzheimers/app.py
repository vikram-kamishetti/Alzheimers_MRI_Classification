
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, models


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Alzheimer's MRI Analyzer",
    page_icon="🧠",
    layout="wide"
)

MODEL_PATH = "best_resnet18.pth"
IMAGE_SIZE = 224
NUM_CLASSES = 4

# IMPORTANT:
# Replace these names only after verifying the exact
# integer-to-class mapping used by your dataset.

CLASS_NAMES = {
    0: "Mild Demented",
    1: "Moderate Demented",
    2: "Non-Demented",
    3: "Very Mild Demented"
}



# --------------------------------------------------
# 2. LOAD THE TRAINED RESNET18 MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = models.resnet18(weights=None)

    # Adapt ResNet18 to grayscale MRI images
    old_conv = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    # Replace the final layer with four classes
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device


# --------------------------------------------------
# 3. IMAGE PREPROCESSING
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485],
        std=[0.229]
    )
])


# --------------------------------------------------
# 4. GRAD-CAM IMPLEMENTATION
# --------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        self.handle = target_layer.register_forward_hook(
            self.save_activation
        )

    def save_activation(self, module, inputs, output):
        self.activations = output
        output.register_hook(self.save_gradient)

    def save_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            output = self.model(input_tensor)

            score = output[:, class_idx].sum()
            score.backward()

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def close(self):
        self.handle.remove()


# --------------------------------------------------
# 5. APPLICATION HEADER
# --------------------------------------------------

st.title("🧠 Alzheimer's MRI Analyzer")

st.write(
    "An educational AI application that uses a trained "
    "ResNet18 model to classify MRI images and visualize "
    "model attention using Grad-CAM."
)

st.warning(
    "Research prototype only. This application is not "
    "clinically validated and must not be used to diagnose "
    "or make medical decisions."
)

st.divider()


# --------------------------------------------------
# 6. IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:

    try:
        original_image = Image.open(
            uploaded_file
        ).convert("L")

        left, right = st.columns(2)

        with left:
            st.subheader("Uploaded MRI")
            st.image(
                original_image,
                caption="Input image",
                use_container_width=True
            )

        with right:
            st.subheader("Image information")
            st.write(f"Width: {original_image.width} pixels")
            st.write(f"Height: {original_image.height} pixels")
            st.write("Image format: Grayscale")

        if st.button(
            "Analyze MRI Image",
            type="primary",
            use_container_width=True
        ):

            try:
                model, device = load_model()

                input_tensor = transform(
                    original_image
                ).unsqueeze(0).to(device)

                # ----------------------------------
                # 7. PREDICTION
                # ----------------------------------

                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(
                        logits, dim=1
                    )[0]

                    predicted_class = int(
                        probabilities.argmax().item()
                    )

                    confidence = float(
                        probabilities[predicted_class].item()
                    )

                st.divider()
                st.subheader("Prediction Results")

                st.metric(
                    "Predicted class",
                    CLASS_NAMES[predicted_class]
                )

                st.metric(
                    "Model confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.caption(
                    "Confidence is the model's softmax score, "
                    "not the probability that a patient has "
                    "Alzheimer's disease."
                )

                st.subheader("Scores for all classes")

                for class_id in range(NUM_CLASSES):
                    score = float(probabilities[class_id].item())

                    st.write(
                        f"{CLASS_NAMES[class_id]}: "
                        f"{score * 100:.2f}%"
                    )

                    st.progress(score)

                # ----------------------------------
                # 8. GENERATE GRAD-CAM
                # ----------------------------------

                st.divider()
                st.subheader("Grad-CAM Explainability")

                gradcam = GradCAM(
                    model,
                    model.layer4
                )

                try:
                    heatmap = gradcam.generate(
                        input_tensor,
                        predicted_class
                    )
                finally:
                    gradcam.close()

                # Prepare the original image for overlay
                display_image = original_image.resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                )

                image_array = np.asarray(
                    display_image,
                    dtype=np.float32
                ) / 255.0

                # Create visualizations
                fig, axes = plt.subplots(
                    1, 3, figsize=(15, 5)
                )

                axes[0].imshow(
                    image_array,
                    cmap="gray"
                )
                axes[0].set_title("Original MRI")

                axes[1].imshow(
                    heatmap,
                    cmap="jet",
                    vmin=0,
                    vmax=1
                )
                axes[1].set_title("Grad-CAM Heatmap")

                axes[2].imshow(
                    image_array,
                    cmap="gray"
                )
                axes[2].imshow(
                    heatmap,
                    cmap="jet",
                    alpha=0.4,
                    vmin=0,
                    vmax=1
                )
                axes[2].set_title("Heatmap Overlay")

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.success(
                    "Analysis completed successfully."
                )

            except FileNotFoundError:
                st.error(
                    "Model file not found. Place "
                    "'best_resnet18.pth' in the same folder "
                    "as app.py."
                )

            except Exception as error:
                st.error(
                    f"Analysis failed: {error}"
                )

    except Exception as error:
        st.error(
            f"Could not open the uploaded image: {error}"
        )

else:
    st.info(
        "Upload an MRI image above to begin."
    )


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Streamlit, PyTorch, ResNet18, and Grad-CAM. "
    "For educational and research purposes only."
)
