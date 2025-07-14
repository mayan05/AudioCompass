from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from process import trim_and_extract_features
from torch import device, cuda, load
from torch_model import CNN_LSTM_Model_PyTorch as cnn

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post('/predict')
async def analyse(file: UploadFile):
    try:
        features = trim_and_extract_features(file)

        ## HANDLE RESPONSE HERE !!!
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


