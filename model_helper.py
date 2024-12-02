import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import random
import base64
import io
import os
import torchvision
from tqdm import tqdm

# Load pre-trained VGG19 model and modify it to return features
from torchvision.models import VGG19_Weights

vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define style and content layers to extract features
content_layers = ['21']  # Layer where content is represented
style_layers = ['0', '5', '10', '19', '28']  # Layers where style is represented

def get_all_images_from_folder(folder_path, valid_extensions=None):
    # Set default valid extensions if none are provided
    if valid_extensions is None:
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    # List all files in the folder
    all_files = os.listdir(folder_path)
    
    # Filter files that are images based on extensions
    image_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
    
    # Get the full paths of the image files
    image_paths = [os.path.join(folder_path, img_file) for img_file in image_files]
    
    return image_paths

# Example usage
folder_path = "C:/Users/savya/OneDrive/Desktop/project_ml/templates/vgg_styles"
style_images = get_all_images_from_folder(folder_path)



from PIL import Image
import torchvision.transforms as transforms
def get_features(x, model, layers):
    features = []
    for name, layer in model.named_children():
        x = layer(x)
        if name in layers:
            features.append(x)
    return features

# Preprocessing function for VGG19
def preprocess_image(image):
    # If the image is already a tensor, convert it to PIL Image first
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Function to extract features from a specific layer of VGG19
def get_features(image, model, layers):
    features = []
    x = image
    for i, layer in enumerate(model):
        x = layer(x)
        if str(i) in layers:
            features.append(x)
    return features

# Compute the content loss
def content_loss(content_features, target_features):
    return F.mse_loss(target_features, content_features)

# Compute the style loss
def style_loss(style_features, target_features):
    loss = 0
    for sf, tf in zip(style_features, target_features):
        _, c, h, w = tf.size()
        sf = sf.view(c, h * w)
        tf = tf.view(c, h * w)
        gram_sf = torch.mm(sf, sf.t())
        gram_tf = torch.mm(tf, tf.t())
        loss += F.mse_loss(gram_sf, gram_tf)
    return loss
# Function to apply style transfer
def apply_style_transfer(content_image: torch.Tensor, style_image_path: str):
    # Load style image and preprocess it
    style_image = Image.open(style_image_path).convert("RGB")
    style_tensor = preprocess_image(style_image)
    content_image = preprocess_image(content_image)

    # Move to device (either CPU or GPU)
    style_tensor = style_tensor.to(device)
    content_image = content_image.to(device)
    
    # Ensure model is also on the same device (GPU or CPU)
    vgg19.to(device)

    # Run through the style transfer pipeline (simplified)
    target = content_image.clone().requires_grad_(True)

    # Get content and style features using the VGG19 model
    content_features = get_features(content_image, vgg19, content_layers)
    style_features = get_features(style_tensor, vgg19, style_layers)
    
    # Here, you would normally define the loss function and optimization steps
    # This step would also involve iterating through the model and applying style transfer
    
    return target.squeeze().cpu().detach()

# Convert tensor to base64 for displaying on the web
def image_to_base64(image_tensor):
    image = transforms.ToPILImage()(image_tensor)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to generate styled images from a batch
def get_images_base64(images):
    base64_images = []
    for image_tensor in images:
        style_image_path = random.choice(style_images)  # Randomly choose a style
        styled_image = apply_style_transfer(image_tensor, style_image_path)
        base64_images.append(image_to_base64(styled_image))
    return base64_images

# Load images from a folder
def load_images(folder_path, img_size=64):
    X = []
    transform = transforms.Resize((img_size, img_size))

    for img_name in tqdm(os.listdir(folder_path)):
        try:
            img_path = os.path.join(folder_path, img_name)
            img_tensor = torchvision.io.read_image(img_path)
            img_tensor = transform(img_tensor)
            X.append(img_tensor / 255.0)
            if len(X) > 100000:
                break
        except:
            pass
    return X

# Display function to visualize images
def show_images(images, N=32):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 15))
    grid = torchvision.utils.make_grid(images[:N])
    grid = grid.numpy().transpose((1, 2, 0))
    plt.imshow(grid)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Example usage
if __name__ == "__main__":
    # Assuming images are loaded from a folder
    folder_path = "path_to_your_images_folder"
    images = load_images(folder_path)

    # Generate base64 images after style transfer
    styled_images_base64 = get_images_base64(images[:10])  # Process first 10 images for example

    # Print the base64 result for display or use in a web app
    for i, img_base64 in enumerate(styled_images_base64):
        print(f"Styled Image {i+1}: {img_base64[:50]}...")  # Print a snippet of the base64 string
