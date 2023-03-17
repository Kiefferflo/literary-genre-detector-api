from fastapi import FastAPI
from app.modelIA import *

app = FastAPI()

@app.post("/api/predict/summary")
async def predictFromSummary(summary: str):
    fileName_Model = "app/Model/predictWithSummary_TreeClassifier"
    return predictWithsummary_TreeClassifier_Predict(fileName_Model,summary)

@app.post("/api/train/summary")
async def predictFromSummary():
    fileName_Data = "app/Données/BooksDataSet.csv"
    predictWithSummary_TreeClassifier_Model(fileName_Data)
    return "Success"

@app.post("/api/predict/title")
async def predictFromTitle(title: str):
    fileName_Model = "app/Model/predictWithTitle_TreeClassifier"
    return predictWithTitle_TreeClassifier_Predict(fileName_Model,title)

@app.post("/api/train/title")
async def predictFromTitle():
    fileName_Data = "app/Données/BooksDataSet.csv"
    predictWithTitle_TreeClassifier_Model(fileName_Data)
    return "Success"

@app.post("/api/predict/author")
async def predictFromAuthor(author: str):
    nomFichier = 'app/Données/Author_generes.csv'
    return predictWithAuthor(author,nomFichier)