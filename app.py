from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os

app = FastAPI(title="MNIST CNN API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

# Global model variable
model = None

# Load model at startup
@app.on_event("startup")
def load_model():
    global model
    model = CNN().to(device)
    
    model_path = "cnn_mnist_weights.pkl"
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found: {model_path}")
        return
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.get("/")
def home():
    return {"message": "MNIST CNN FastAPI is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    global model
    
    if model is None:
        return {"error": "Model not loaded. Check server logs."}
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return {
            "filename": file.filename,
            "prediction": int(predicted.item()),
            "confidence": float(confidence.item())
        }
    
    except Exception as e:
        return {"error": str(e)}