import torch
from PIL import Image
from torchvision import transforms

from model import FlowerCNN
from config import resize_x, resize_y, model_save_path


def predict_images(model, list_of_img_paths):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor()
    ])

    images = []

    for path in list_of_img_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        images.append(img)

    batch = torch.stack(images)

    with torch.no_grad():
        outputs = model(batch)
        _, preds = torch.max(outputs, 1)

    return preds.tolist()


# Run prediction directly
if __name__ == "__main__":
    print("🔍 Running prediction...")

    # Load model
    model = FlowerCNN()
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Example test images (change paths if needed)
    test_images = [
        "data/daisy/img1.jpg",
        "data/rose/img1.jpg"
    ]

    predictions = predict_images(model, test_images)

    print("Predictions:", predictions)
