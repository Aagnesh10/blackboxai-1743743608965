import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir, "images")
images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Load pre-trained ResNet18 model
model = torchvision.models.resnet18(weights="DEFAULT")
model.eval()

# Initialize variables
Names = []
Vectors = None

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hook for feature extraction
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.avgpool.register_forward_hook(get_activation("avgpool"))

# Generate feature vectors
print("Generating feature vectors...")
with torch.no_grad():
    for i, file in enumerate(images):
        try:
            img_path = os.path.join(images_dir, file)
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = transform(img)
            out = model(img[None, ...])
            vec = activation["avgpool"].numpy().squeeze()[None, ...]
            if Vectors is None:
                Vectors = vec
            else:
                Vectors = np.vstack([Vectors, vec])
            Names.append(file)
            if i % 100 == 0 and i != 0:
                print(f"Processed {i} images")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

# Save the vectors and names
vectors_path = os.path.join(current_dir, "Vectors.npy")
names_path = os.path.join(current_dir, "Names.npy")
np.save(vectors_path, Vectors)
np.save(names_path, Names)
print(f"Successfully saved vectors to {vectors_path}")
print(f"Successfully saved names to {names_path}")
