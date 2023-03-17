from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.modelIA import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fileName_Data = "app/Données/BooksDataSet.csv"

@app.post("/api/predict/summary/tree")
async def predictFromSummaryTree(summary: str):
    fileName_Model = "app/Model/predictWithSummary_TreeClassifier"
    return predictWithsummary_TreeClassifier_Predict(fileName_Model,summary)

@app.post("/api/train/summary/tree")
async def predictFromSummaryTree():
    predictWithSummary_TreeClassifier_Model(fileName_Data)
    return "Success"

@app.post("/api/predict/summary/tree")
async def predictFromSummaryNN(summary: str):
    fileName_Model = "app/Model/predictWithSummary_TreeClassifier"
    return predictWithSummary_NN_Predict(fileName_Model, fileName_Data, summary)

@app.post("/api/train/summary/tree")
async def predictFromSummaryNN():
    predictWithSummary_NN_Model(fileName_Data)
    return "Success"

@app.post("/api/predict/title")
async def predictFromTitle(title: str):
    fileName_Model = "app/Model/predictWithTitle_TreeClassifier"
    return predictWithTitle_TreeClassifier_Predict(fileName_Model,title)

@app.post("/api/train/title")
async def predictFromTitle():
    predictWithTitle_TreeClassifier_Model(fileName_Data)
    return "Success"

@app.post("/api/predict/author")
async def predictFromAuthor(author: str):
    nomFichier = 'app/Données/Author_generes.csv'
    return predictWithAuthor(author,nomFichier)