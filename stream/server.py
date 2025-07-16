from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from process import trim_and_extract_features
from torch import device, cuda, load, tensor
from torch_model import CNN_LSTM_Model_PyTorch as cnn
import torch

try:
    chip = device("cuda" if cuda.is_available() else "cpu")
    model = cnn(num_classes=24).to(chip)
    model.load_state_dict(load('../best_model.pt', map_location=chip))
    model.eval() # kept on evaluation mode
    print("The model has been loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file 'best_model.pt' not found. Please ensure the path is correct.")
    raise RuntimeError("Model file not found. Server cannot start.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

app = FastAPI(title='server')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://attractive-achievement-production.up.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class2idx = {
    0  : 'C Major',  1  : 'C Minor',
    2  : 'C# Major', 3  : 'C# Minor',
    4  : 'D Major',  5  : 'D Minor',
    6  : 'D# Major', 7  : 'D# Minor',
    8  : 'E Major',  9  : 'E Minor',
    10 : 'F Major',  11 : 'F Minor',
    12 : 'F# Major', 13 : 'F# Minor',
    14 : 'G Major',  15 : 'G Minor',
    16 : 'G# Major', 17 : 'G# Minor',
    18 : 'A Major',  19 : 'A Minor',
    20 : 'A# Major', 21 : 'A# Minor',
    22 : 'B Major',  23 : 'B Minor',
}

@app.post('/predict')
async def analyse(file: UploadFile):
    try:
        features, tempo = trim_and_extract_features(file)
        features = tensor(features,dtype=torch.float32).unsqueeze(1)
        features = features.to(next(model.parameters()).device)
        prediction = model.predict(features) 
        label = class2idx[prediction] 

        return JSONResponse(status_code=200, content={"prediction": {"index": prediction,"label": label,"tempo":round(float(tempo))}})
                                                                                                        # tempo hehe
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


