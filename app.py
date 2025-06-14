from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# import google.generativeai as genai
import os
from google import genai
from google.genai import types
# import base64

# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# model_maths = genai.GenerativeModel(
#     model_name="tunedModels/generate-num-7045")

# model_sinhala = genai.GenerativeModel(
#     model_name="tunedModels/generate-num-8211")



def generate(model, question):
  client = genai.Client(
      vertexai=True,
      project="557553409003",
      location="us-central1",
  )

  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text=question)
      ]
    )
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="BLOCK_MEDIUM_AND_ABOVE"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="BLOCK_MEDIUM_AND_ABOVE"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="BLOCK_MEDIUM_AND_ABOVE"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="BLOCK_MEDIUM_AND_ABOVE"
    )],
  )

  llm_response = "" # Initialize an empty string to accumulate the response

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    llm_response += chunk.text # Accumulate each chunk of text

  return llm_response # Return the complete response after the loop


class GenerationRequest(BaseModel):
    prompt: str

pipeline_cat_cla = joblib.load('cat_classifier_1.pkl')
pipeline_cat_math_cla = joblib.load('math_classifier_1.pkl')

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



@app.post("/predictMathCategory/")
def predict(item: TextIn):
    # Single-item prediction
    label = pipeline_cat_math_cla.predict([item.text])[0]
    return {"prediction": label}



@app.post("/predictCategory_batch/")
def predict_batch_cla(batch: BatchIn):
    labels = pipeline_cat_cla.predict(batch.texts)
    return {"predictions": labels.tolist() if hasattr(labels, "tolist") else list(labels)}



@app.post("/predictMathCategory_batch/")
def predict_batch_math_cla(batch: BatchIn):
    labels = pipeline_cat_math_cla.predict(batch.texts)
    return {"predictions": labels.tolist() if hasattr(labels, "tolist") else list(labels)}



@app.post("/MathExplanation/")
async def generate_text(req: GenerationRequest):
    model = "projects/557553409003/locations/us-central1/endpoints/724847543053189120"
    try:
        resp = generate(model, req.prompt)
        return {"generated_text": resp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/SinExplanation/")
async def generate_text(req: GenerationRequest):
    model = "projects/557553409003/locations/us-central1/endpoints/4682948665558433792"

    try:
        resp = generate(model, req.prompt)
        return {"generated_text": resp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/SinAnswer/")
async def fetch_answer(req: GenerationRequest):
    model = "projects/557553409003/locations/us-central1/endpoints/4682948665558433792"

    prompt  = """

            INSTRUCTIONS
            ------------
            1. Read the multiple-choice question that follows (it may be in Sinhala).
            2. Privately solve the problem and identify the correct choice.
            3. **Return ONLY the answer text itself – exactly as it appears in the question – with no option letter/number, no punctuation, and no explanation.**

            EXAMPLE
            Question:
            1+2?
            (1) 1
            (2) 3
            (3) 0

            Output:
            3

            Only return 3. Do not return (2), the option number.

            Now it is your turn:
            Question:
            {Question}

    """

    try:
        fetch_answer_prompt = prompt.format(Question=req.prompt)
        resp = generate(model, fetch_answer_prompt)
        return {"generated_text": resp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/MathAnswer/")
async def fetch_answer(req: GenerationRequest):
    model = "projects/557553409003/locations/us-central1/endpoints/724847543053189120"
    from langchain_google_genai import ChatGoogleGenerativeAI

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash",
    #     temperature=0,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    #     # other params...
    # )

    prompt  = """

            INSTRUCTIONS
            ------------
            1. Read the multiple-choice question that follows (it may be in Sinhala).
            2. Privately solve the problem and identify the correct choice.
            3. **Return ONLY the answer text itself – exactly as it appears in the question – with no option letter/number, no punctuation, and no explanation.**

            EXAMPLE
            Question:
            1+2?
            (1) 1
            (2) 3
            (3) 0

            Output:
            3

            Only return 3. Do not return (2), the option number.

            Now it is your turn:
            Question:
            {Question}

    """

    try:
        fetch_answer_prompt = prompt.format(Question=req.prompt)
        resp = generate(model, fetch_answer_prompt)
        # a = llm.invoke(fetch_answer_prompt)
        return {"generated_text": resp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),  # important
        reload=True
    )

# uvicorn app:app --host 0.0.0.0 --port 8000 --reload


