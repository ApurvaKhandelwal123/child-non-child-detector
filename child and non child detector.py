#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# In[2]:


train_dir ='C:/Users/vinayak khandelwal/Downloads/child non child facial detection/dataset'


# In[5]:


class_names = sorted(os.listdir(train_dir))
class_names


# In[10]:


# Example directory structure
train_dir = 'C:/Users/vinayak khandelwal/Downloads/child non child facial detection/dataset/'

class_names = os.listdir(train_dir)
class_dis = [len(os.listdir(os.path.join(train_dir, name))) for name in class_names]
class_dis


# In[ ]:





# In[ ]:


# Define the CNN model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'C:/Users/vinayak khandelwal/Downloads/child non child facial detection/dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Train the model
model.fit(train_generator, epochs=20)

# Save the trained model
model.save('child_detection_model.h5')


# In[2]:


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('child_detection_model.h5')

# Define image size for model input
img_size = (224, 224)

# Map class indices to class names
class_names = {0: "child", 1: "non-child"}

# GUI setup
class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Classifier")

        # Create widgets
        self.label = tk.Label(self.master, text="Select an image:")
        self.label.pack()

        self.browse_button = tk.Button(self.master, text="Browse", command=self.browse_image)
        self.browse_button.pack()

        self.classify_button = tk.Button(self.master, text="Classify", command=self.classify_image)
        self.classify_button.pack()

        self.result_label = tk.Label(self.master, text="")
        self.result_label.pack()

        self.image_label = tk.Label(self.master)
        self.image_label.pack()

        # Initialize img_tk and image_np as class attributes
        self.img_tk = None
        self.image_np = None

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.display_image(file_path)

    def classify_image(self):
        if self.image_np is not None:
            # Preprocess the image
            img = Image.fromarray(self.image_np)
            img = img.resize(img_size)
            img_array = np.expand_dims(np.array(img), axis=0) / 255.0

            # Perform image classification
            predictions = model.predict(img_array)
            class_index = np.argmax(predictions)
            confidence = predictions[0][class_index]

            # Display result with confidence score and class name
            if confidence >= 0.4:
                class_name = class_names.get(class_index, "Unknown")
                self.result_label.config(text=f"Prediction: {class_name} (Confidence: {confidence:.2f})")
            else:
                # Display the other class name
                other_class_index = 1 - class_index
                other_class_name = class_names.get(other_class_index, "Unknown")
                self.result_label.config(text=f"Prediction: {other_class_name} (Confidence: {1 - confidence:.2f})")

        else:
            self.result_label.config(text="No image selected")

    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((300, 300))
        self.img_tk = ImageTk.PhotoImage(img)

        self.image_label.config(image=self.img_tk)

        # Save numpy image for later classification
        self.image_np = np.array(img)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()


# In[ ]:




