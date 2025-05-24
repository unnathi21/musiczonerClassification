import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ImageProcessor:
    def __init__(self):
        # Load pretrained ResNet model
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        # Simplified genre mapping for album covers
        self.labels = {
            0: "pop",  # Placeholder indices
            1: "rock",
            2: "classical",
            3: "hip-hop"
        }
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify_cover(self, image_path):
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0)

            # Classify image
            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = torch.max(outputs, 1)
                label_idx = predicted.item() % len(self.labels)  # Simplify for demo
            return self.labels.get(label_idx, "unknown")
        except Exception as e:
            raise Exception(f"Error classifying album cover: {str(e)}")