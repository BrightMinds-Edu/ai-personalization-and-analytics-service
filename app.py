from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

model = genai.GenerativeModel(
    model_name="tunedModels/generate-num-7045")

model = genai.GenerativeModel(
    model_name="tunedModels/generate-num-8211")

class GenerationRequest(BaseModel):
    prompt: str

pipeline_cat_cla = joblib.load('cat_cla.pkl')
pipeline_cat_math_cla = joblib.load('math_classifier.pkl')

class TextIn(BaseModel):
    text: str

class BatchIn(BaseModel):
    texts: List[str]

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome! POST to /predict/ (single) or /predict_batch/ (batch)."}

@app.post("/predictCategory/")
def predict(item: TextIn):
    # Single-item prediction
    label = pipeline_cat_cla.predict([item.text])[0]
    return {"prediction": label}

@app.post("/predictCategory_batch/")
def predict_batch_cla(batch: BatchIn):
    labels = pipeline_cat_cla.predict(batch.texts)
    return {"predictions": labels.tolist() if hasattr(labels, "tolist") else list(labels)}

@app.post("/predictMathCategory_batch/")
def predict_batch_math_cla(batch: BatchIn):
    labels = pipeline_cat_math_cla.predict(batch.texts)
    return {"predictions": labels.tolist() if hasattr(labels, "tolist") else list(labels)}

@app.post("/generate/")
async def generate_text(req: GenerationRequest):
    try:
        resp = model.generate_content(req.prompt)
        return {"generated_text": resp.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),  # important
        reload=True
    )

