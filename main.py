from fastapi import FastAPI, Query
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
import nltk
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Charger les ressources de nltk
nltk.download('punkt')

# Charger le modèle
model_path = 'modele_chatbot.pkl'
pipeline = joblib.load(model_path)

# Prétraitement des données
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return ' '.join(tokens)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permettre toutes les origines, pour les tests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint pour obtenir des prédictions
@app.get("/predict/")
async def predict(question: str = Query(..., title="Question")):
    question_processed = preprocess_text(question)
    response = pipeline.predict([question_processed])[0]
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.1.100", port=8000)