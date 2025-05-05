import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
nlp = spacy.load("en_core_web_trf")

class TextRequest(BaseModel):
    text: str

class NLPResponse(BaseModel):
    entities: list
    pos_tags: list
    dependencies: list

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    try:
        doc = nlp(request.text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        pos_tags = [{"text": token.text, "pos": token.pos_} for token in doc]
        dependencies = [{"text": token.text, "head": token.head.text, "dep": token.dep_} for token in doc]
        return NLPResponse(entities=entities, pos_tags=pos_tags, dependencies=dependencies)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}