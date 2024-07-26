import os
from fastapi import FastAPI, UploadFile, File
import CNN
from CNN import idx_to_classes
import numpy as np
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import uvicorn

app = FastAPI()
model= CNN.CNN(78)
state_dict = torch.load("/home/raccoon/Desktop/development/plants/Plant-Disease-Detection/Model/plant_disease_model_100epochs.pt")
model.load_state_dict(state_dict, strict=False)

model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def single_prediction(image):
    #image = Image.open(image_path).convert('RGB')
    
    input_data = transform(image)
    
    input_data = input_data.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_data)
    
    output = output.numpy()
    
    index = np.argmax(output, axis=1)[0]
    
    disease = idx_to_classes[index]
    
    return  disease


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    predict=single_prediction(image)
    return {"prediction": predict}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
