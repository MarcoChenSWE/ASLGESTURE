# import os
# import tensorflow as tf
# from mediapipe_model_maker import gesture_recognizer
# import matplotlib.pyplot as plt

# # Verify TensorFlow version
# assert tf.__version__.startswith('2')

# # Path to your dataset
# dataset_path = './TrainData2'

# # Load dataset
# data = gesture_recognizer.Dataset.from_folder(
#     dirname=dataset_path,
#     hparams=gesture_recognizer.HandDataPreprocessingParams()
# )

# # Split the dataset
# train_data, rest_data = data.split(0.8)
# validation_data, test_data = rest_data.split(0.5)

# # Define hyperparameters
# hparams = gesture_recognizer.HParams(
#     learning_rate=0.003,
#     batch_size=16,
#     epochs=20,
#     export_dir="model"
# )

# # Model options
# model_options = gesture_recognizer.ModelOptions(
#     dropout_rate=0.2,
#     layer_widths=[128, 64]
# )

# # Create the GestureRecognizerOptions object
# options = gesture_recognizer.GestureRecognizerOptions(
#     model_options=model_options,
#     hparams=hparams
# )

# # Train the model
# model = gesture_recognizer.GestureRecognizer.create(
#     train_data=train_data,
#     validation_data=validation_data,
#     options=options
# )

# # Evaluate the model
# loss, accuracy = model.evaluate(test_data, batch_size=1)
# print(f"Test Loss: {loss}")
# print(f"Test Accuracy: {accuracy}")

# # Export the model
# model.export_model()

# # Verify exported model
# print("Exported Model Files:", os.listdir(hparams.export_dir))

import os
import random
import shutil
import time
import tensorflow as tf
from mediapipe_model_maker import gesture_recognizer
import matplotlib.pyplot as plt

# Function to create a subset of the dataset
def create_subset(input_dir, output_dir, max_images_per_label=50):
    """
    Create a smaller subset of the dataset by sampling a fixed number of images per label.
    """
    os.makedirs(output_dir, exist_ok=True)
    for label in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label)
        if os.path.isdir(label_path):
            output_label_dir = os.path.join(output_dir, label)
            os.makedirs(output_label_dir, exist_ok=True)
            images = os.listdir(label_path)
            sampled_images = random.sample(images, min(len(images), max_images_per_label))
            for img_file in sampled_images:
                shutil.copy(os.path.join(label_path, img_file), os.path.join(output_label_dir, img_file))

# Main script
if __name__ == "__main__":
    # Verify TensorFlow version
    assert tf.__version__.startswith('2'), "TensorFlow 2.x is required!"

    # Paths for the dataset
    dataset_path = './TrainData2'  # Original dataset
    subset_dataset_path = './SubsetTrainData'  # Subset dataset

    # Step 1: Create a smaller subset of the dataset
    print("Creating a smaller subset of the dataset...")
    create_subset(dataset_path, subset_dataset_path, max_images_per_label=500)  # Use 50 images per label

    # Step 2: Load and preprocess the subset dataset
    print("Loading and preprocessing the subset dataset...")
    start_time = time.time()
    data = gesture_recognizer.Dataset.from_folder(
        dirname=subset_dataset_path,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")

    # Step 3: Split the dataset
    train_data, rest_data = data.split(0.8)  # 80% training, 20% test/validation
    validation_data, test_data = rest_data.split(0.5)  # Split the remaining 20% equally

    # Step 4: Define hyperparameters
    hparams = gesture_recognizer.HParams(
        learning_rate=0.001,   # Reduced learning rate for better convergence                
        batch_size=32,
        epochs=15,  # Reduced epochs for faster training
        export_dir="model"
    )

    # Step 5: Define model options
    model_options = gesture_recognizer.ModelOptions(
        dropout_rate=0.3,            # Slightly higher dropout to reduce overfitting
        layer_widths=[256,128, 64]   # Wider layers for more capacity
    )

    # Hyperparameter	Recommended Value	Description
    # Learning Rate	0.001 to 0.003	Controls how much to adjust weights during training. Lower values train slower.
    # Batch Size	16 to 32	Number of samples per training step. Smaller sizes may generalize better but train slower.
    # Epochs	10 to 20	Number of passes over the dataset. Adjust based on convergence.
    # Steps Per Epoch	Automatically calculated	Leave this unset unless you have a specific need.
    # Dropout Rate	0.2 to 0.5	Fraction of units to drop to prevent overfitting.
    # Layer Widths	[128, 64]	Number of neurons in additional hidden layers for fine-tuning.
    # Learning Rate Decay	0.99	Gradually reduces the learning rate during training.
    # Gamma (for Focal Loss)	2.0	Controls the focus on harder examples.

    # Step 6: Train the model
    print("Starting model training...")
    start_time = time.time()
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=gesture_recognizer.GestureRecognizerOptions(
            model_options=model_options,
            hparams=hparams
        )
    )
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # Step 7: Evaluate the model
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(test_data, batch_size=1)
    print(f"Test Loss: {loss},Test Accuracy: {accuracy}")
    

    # Step 8: Export the model
    print("Exporting the model...")
    model.export_model()
    print("Model exported successfully!")

    # Step 9: Verify exported model
    print("Exported Model Files:", os.listdir(hparams.export_dir))
