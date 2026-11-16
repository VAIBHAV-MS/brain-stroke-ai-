import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from huggingface_hub import hf_hub_download


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILENAME = "stroke_resnet50_deploy.h5"

LOCAL_MODEL_PATH = os.path.join(
    BASE_DIR,
    MODEL_FILENAME
)

CLASS_NAMES = [
    "hemorrhagic",
    "ischemic",
    "normal"
]

IMAGE_SIZE = (224, 224)

# Model is NOT loaded when the server starts.
# It will be loaded only when prediction is requested.
model = None


def get_model():

    global model

    if model is None:

        if os.path.exists(LOCAL_MODEL_PATH):

            MODEL_PATH = LOCAL_MODEL_PATH

        else:

            MODEL_PATH = hf_hub_download(
                repo_id="Vaibhav-MS/stroke-mri-resnet50",
                filename=MODEL_FILENAME
            )

        model = tf.keras.models.load_model(
            MODEL_PATH
        )

    return model


def predict_mri(image_path):

    model = get_model()

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = preprocess_input(
        image_array
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        predictions
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index] * 100
    )

    return (
        predicted_class,
        confidence,
        predictions
    )


if __name__ == "__main__":

    image_path = input(
        "Enter MRI image path: "
    )

    predicted_class, confidence, predictions = predict_mri(
        image_path
    )

    print("\n" + "=" * 50)
    print("STROKE MRI PREDICTION")
    print("=" * 50)

    print(
        f"Prediction : {predicted_class}"
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )

    print("\nAll probabilities:")

    for class_name, probability in zip(
        CLASS_NAMES,
        predictions
    ):

        print(
            f"{class_name:15}: "
            f"{probability * 100:.2f}%"
        )

    print("=" * 50)