from fastapi import FastAPI
from modelIA import *

app = FastAPI()

@app.post("/api/predict/summary")
async def predictFromSummary(summary: str):
    return predictWithSummary(summary)

@app.post("/api/predict/title")
async def predictFromTitle(title: str):
    return predictWithTitle(title)

@app.post("/api/predict/author")
async def predictFromAuthor(author: str):
    return predictWithAuthor(author)