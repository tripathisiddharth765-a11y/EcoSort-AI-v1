import torch
import torchvision.transforms as transforms
from dataset import classes
from model import ResNet18
from PIL import Image
from pathlib import Path

def load_model():
    model=ResNet18()
    model.load_state_dict(
    torch.load("waste_classifier.pth", map_location=torch.device("cpu"))
)
    model.eval()
    return model

def load_image(image):
    image = image.convert("RGB")

    inference_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = inference_transformer(image)
    image = image.unsqueeze(0)   
    return image

def predict(image,model):
    with torch.no_grad():
     output=model(image)
     probability=torch.softmax(output,dim=1)
     prediction=torch.argmax(probability,dim=1)
     confidence = probability[0][prediction.item()] * 100
     predict=classes[prediction]
     return predict,confidence
 
def main():
    model=load_model()
    image=load_image(image_path=r"C:\Users\Acer\Pictures\Saved Pictures\banana_pell.jpg")   
    prediction,confidence=predict(image=image,model=model)
    print(f"Prediction : {prediction} , confidence : {confidence}")
    
if __name__ == "__main__":
    main()






























