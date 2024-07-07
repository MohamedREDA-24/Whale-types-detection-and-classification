import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf

classnames = ['Jelly', 'fish', 'shark', 'tuna', 'whale']

test_folder_path = "test"

IMGSIZE = 224
model = tf.keras.models.load_model("my_model.h5")

correct_predictions = 0
total_images = 0

for class_name in classnames:
    class_folder_path = os.path.join(test_folder_path, class_name)

    for image_name in os.listdir(class_folder_path):
        image_path = os.path.join(class_folder_path, image_name)

        try:
            test_image = load_img(image_path, target_size=(IMGSIZE, IMGSIZE))
        except OSError:
            continue

        test_image_array = img_to_array(test_image)
        test_image_array = np.expand_dims(test_image_array, axis=0)
        test_image_array = preprocess_input(test_image_array)

        predictions = model.predict(test_image_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = classnames[predicted_class_index]

        # Display the image and predicted class
        # plt.imshow(test_image)
        # plt.title(f'Actual Class: {class_name}, Predicted Class: {predicted_class_name}')
        # plt.show()
        print(f'Actual Class: {class_name}, Predicted Class: {predicted_class_name}')

        if predicted_class_name == class_name:
            correct_predictions += 1
        total_images += 1

accuracy = correct_predictions / total_images
print(f'Accuracy: {accuracy * 100:.2f}%')

