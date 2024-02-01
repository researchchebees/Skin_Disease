import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the DeepLabv3 model
model = tf.keras.applications.DenseNet201(input_shape=(None, None, 3), include_top=False)
model.load_weights('path/to/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Function to perform segmentation on an input image
def perform_segmentation(image_path):
    # Load and preprocess the input image
    input_image = Image.open(image_path)
    input_array = tf.keras.preprocessing.image.img_to_array(input_image)
    input_array = tf.image.resize(input_array, (512, 512))  # Resize to match the model's expected sizing
    input_array = tf.expand_dims(input_array, 0)  # Create a batch

    # Perform segmentation
    predictions = model.predict(input_array)

    # Post-process the segmentation mask
    segmentation_mask = np.argmax(predictions, axis=-1)[0]

    return segmentation_mask

# Example usage
image_path = 'path/to/your/image.jpg'
segmentation_mask = perform_segmentation(image_path)

# Visualize the segmentation mask
plt.imshow(segmentation_mask, cmap='viridis', alpha=0.75)
plt.axis('off')
plt.show()
