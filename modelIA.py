#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Groupe Florent Kieffer & Louna Capel

"""


########################################################################################################################################
################################################################ Import ################################################################
########################################################################################################################################
from pydantic import BaseModel
from typing import Optional
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas
import csv
import requests
from bs4 import BeautifulSoup
import random
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import os
import joblib
import pickle





########################################################################################################################################
############################################################### Scraping ###############################################################
########################################################################################################################################
def scraping (nomFichier):
    """
    Cette fonction permet de rajouter l'auteur du roman dans le fichier source
    
    Parameters
    ----------
        nomFichier : str
            Nom du fichier source
    
    Returns
    -------
        None
            Changement directement dans le fichier source
    """
    lectureFichier = pandas.read_csv(nomFichier)
    listBookName = lectureFichier['book_name']
    
    for i in range (len(listBookName)):
        if (pandas.isna(lectureFichier.at[i, 'Author'])):
            livre = listBookName[i]
            livre = livre.replace(" ", "_")
            titre = "https://en.wikipedia.org/wiki/" + livre
        
            r = requests.get(titre)
            soup = BeautifulSoup(r.content, 'html.parser')
        
            author_section = soup.find('th', text='Author')
            
            if author_section:
                # Récupérer la cellule de données suivante (td) qui contient l'auteur
                author = author_section.find_next_sibling('td').text
            else :
                author = ""

                
            lectureFichier.at[i, 'Author'] = author
            lectureFichier.to_csv(nomFichier, index=False)
            
    return(None)


"""
## Test pour le scrapping 

scraping("Données/BooksDataSet.csv")

"""





########################################################################################################################################
########################################################## Nettoyage du texte ##########################################################
########################################################################################################################################
def removeStopWords(summary : str) -> str:
    """
    Cette fonction sert à enlever les stopsWords
    
    Parameters
    ----------
    summary : str
        résumé

    Returns
    -------
    summaryWithoutStopWord : str
        removeStopWords

    """
    
    stop_words = set(stopwords.words('english'))
    no_stopword_text = [w for w in summary.split() if not w in stop_words]
    
    summaryWithoutStopWord = ' '.join(no_stopword_text)
    return (summaryWithoutStopWord)


def lematizing(summaryWithoutStopWord : str) -> str:
    """
    Cette fonction sert de lematizing
    
    Parameters
    ----------
    summaryWithoutStopWord : str
        résumé sans les StopWords

    Returns
    -------
    lemSummary : str
        lematizing

    """
    
    stemSentence = ""
    lemma = WordNetLemmatizer()
    
    for word in summaryWithoutStopWord.split():
        stem = lemma.lemmatize(word)
        stemSentence += stem
        stemSentence += " "
        
    lemSummary = stemSentence.strip()
    return (lemSummary)


def stemming(lemSummary : str) -> str:
    """
    Cette fonction sert de stemming
    
    Parameters
    ----------
    lemSummary : str
        résumé lemmatisé

    Returns
    -------
    stemSummary : str
        stemming 

    """
    
    stemmer = PorterStemmer()
    stemSentence = ""
    
    for word in lemSummary.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
        
    stemSummary = stemSentence.strip()
    return (stemSummary)


"""
## Test de la partie nettoyage du texte 

summary = "The book tells the story of two friends embarking on an incredible adventure."
print(summary)
Stop = removeStopWords(summary)
print(Stop)
sentence = lematizing(Stop)
print(sentence)
stem = stemming(sentence)
print(stem)

"""


def cleanData (fileName : str) -> None :
    """
    Cette fonction sert à créer un fichier identique au fichier source mais avec les titres et les résumés nettoyés
    
    Parameters
    ----------
    fileName : str
            nom du fichier source avec son chemin

    Returns
    -------
    None

    """
    
    lectureFichier = pandas.read_csv(fileName)
    listNameBook = lectureFichier['book_name']
    listGenre = lectureFichier['genre']
    listAuthor = lectureFichier['Author']
    listSummary = lectureFichier['summary']
    
    listNameBook2 = []
    listGenre2 = []
    listAuthor2 = []
    listSummary2 = []
    
    listBook = []
    
    for i in range (len (listNameBook)):
        NameBookStopWords = removeStopWords(listNameBook[i])
        NameBookLem = lematizing(NameBookStopWords)
        NameBookStem = stemming(NameBookLem)

        
        SummaryStopWords = removeStopWords(listSummary[i])
        SummaryLem = lematizing(SummaryStopWords)
        SummaryStem = stemming(SummaryLem)
        
        ## Fantasy = 1
        ## Science Fiction = 2
        ## Crime Fiction = 3
        ## Historical novel = 4
        ## Horror = 5
        ## Thriller = 6
        
        genre_mapping = {
            'Fantasy': 1,
            'Science Fiction': 2,
            'Crime Fiction': 3,
            'Historical novel': 4,
            'Horror': 5,
            'Thriller': 5
        }
        
        genre = genre_mapping.get(listGenre[i], 0)

        listBook.append([NameBookStem,genre,listAuthor[i],SummaryStem])
    
    with open('Données/BDD.csv', 'w') as f:
      headers = ["NameBook","Genre","Author", "Summary"]
      writer = csv.writer(f)
      writer.writerow(headers)
      writer.writerows(listBook)
        
    return()
    
"""
## test de la partie création ##

fileName = "Données/BooksDataSet.csv"

cleanData(fileName)

"""





########################################################################################################################################
########################################################## Création BDD Autor ##########################################################
########################################################################################################################################
def CreationBDDAutor (fileName: str) -> None:
    """
    Cette fonction permet de créer un fichier contenant les auteurs et leurs genres
    
    Parameters
    ----------
        fileName : str
            nom du fichier source avec son chemin
    
    Returns
    -------
        None
    """

    lectureFichier = pandas.read_csv(fileName)
    listAuthor = lectureFichier['Author']
    listAuthor2 = []
    
    listGenre = lectureFichier['genre']
    listGenre2 = []
    
    listAuthorGenre = []
    
    
    
    for i in range (len(listAuthor)) :
        auteur = (str(listAuthor[i]))
        listAuthor2.append(auteur)
        
        genre = (str(listGenre[i]))
        listGenre2.append(genre)
        
        authorGenre = [listAuthor2[i],listGenre2[i]]
        
        if (authorGenre not in listAuthorGenre) and (listAuthor2[i]!='nan') :
            listAuthorGenre.append((listAuthor2[i],listGenre2[i]))
     
        
    with open('Données/Author_generes.csv', 'w') as f:
      headers = ["Author", "Genres"]
      writer = csv.writer(f)
      writer.writerow(headers)
      writer.writerows(listAuthorGenre)
        
    
    f.close()
    
    return ()


"""
## Test de la création de BDD

fileName = "Données/BooksDataSet.csv"
CreationBDDAutor(fileName)

"""





########################################################################################################################################
############################################################## Class Book ##############################################################
########################################################################################################################################
class Book(BaseModel) :
    title : Optional[str] = None
    author : Optional[str] = None
    summary : Optional[str] = None
    genre : Optional[str] = None
    identifiant : Optional[int] = None





#########################################################################################################################################
########################################################## predictWithSummary ###########################################################
#########################################################################################################################################
def predictWithSummary_TreeClassifier_Model (fileName: str) -> str:
    """
    Cette fonction renvoie la prediction du genre littéraire en fonction d'un résumé
    
    Parameters
    ----------
        summary : str
            résumé
    
    Returns
    -------
        genre : str
            prediction du genre
    """

    lectureFichier = pandas.read_csv(fileName)
    lectureFichier = lectureFichier.sample(frac=1).reset_index(drop=True)
    
    

    listNameBook = lectureFichier['summary']
    listGenre = lectureFichier['genre']
    
    tailleData = len(listNameBook)
    tailleTrain = int((20*tailleData)/100)
    
    listNameBook_train = listNameBook[:tailleTrain]
    listGenre_train = listGenre[:tailleTrain]
    
    listNameBook_test = listNameBook[tailleTrain:]
    listGenre_test = listGenre[tailleTrain:]
    
    listNameBook_train = tuple(listNameBook_train)
    listGenre_train = tuple(listGenre_train)
    
    listNameBook_test = tuple(listNameBook_test)
    listGenre_test = tuple(listGenre_test)
    
    
    
    # Vectorisation des titres avec TfidfVectorizer
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(listNameBook_train)
    
    # Création d'un classifieur RandomForest
    clf = RandomForestClassifier(n_estimators=150,
                                 criterion='gini',
                                 max_depth=None,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0.0,
                                 max_features='auto',
                                 max_leaf_nodes=None,
                                 min_impurity_decrease=0.0,
                                 bootstrap=True,
                                 oob_score=False,
                                 n_jobs=None,
                                 random_state=None,
                                 verbose=0,
                                 warm_start=False,
                                 class_weight=None,
                                 ccp_alpha=0.0,
                                 max_samples=None)
    
    # Entraînement du classifieur
    clf.fit(train_features, listGenre_train)
    
    
    nombreJuste = 0
    nombreFaux = 0
    predictionList = []
    
    for i in range (len(listNameBook_test)):
        book_feature = vectorizer.transform([listNameBook_test[i]])

        # Prédiction du genre du livre
        predicted_genre = clf.predict(book_feature)
        prediction = predicted_genre[0]
        predictionList.append (prediction)
        
        if (prediction == listGenre_test[i]) :
            nombreJuste = nombreJuste + 1
        else :
            nombreFaux = nombreFaux + 1
        
        
    taux_Forest = ((nombreJuste)/(nombreJuste+nombreFaux))*100
    mat_Forest = ConfusionMatrixDisplay(confusion_matrix(list(listGenre_test),predictionList,labels = ["Fantasy","Science Fiction","Crime Fiction","Historical novel","Horror","Thriller"]))
    mat_Forest.plot()
    plt.show()
    print("On peut voir que ",nombreJuste, "ont été bien prédit contre ",nombreFaux,"mal prédit")
    print("Le taux d'accuracy est de ", round(taux_Forest,2),"%")
    
    joblib.dump(clf, "Model/predictWithSummary_TreeClassifier")
    joblib.dump(vectorizer, "Model/predictWithSummary_vectorizer.joblib")
    
    return()


def predictWithsummary_TreeClassifier_Predict (fileName : str, title : str) -> str :
    """
    Cette fonction renvoie la prediction du genre littéraire en fonction d'un titre
    
    Parameters
    ----------
        title : str
            titre
        fileName : str
            nom du fichier source avec son chemin
    
    Returns
    ------- 
        genre : str
            prediction du genre
    """

    clf = joblib.load(fileName)
        
    # Vectorisation du titre à prédire
    vectorizer = joblib.load('Model/predictWithSummary_vectorizer.joblib')
    
    book_feature = vectorizer.transform([title])

    # Prédiction du genre du livre
    predicted_genre = clf.predict(book_feature)

    return(predicted_genre[0])


"""
## Test de la prediction avec summary

fileName_Data = "Données/BooksDataSet.csv"
fileName_Model = "Model/predictWithSummary_TreeClassifier"

summary = "The book tells the story of two friends embarking on an incredible adventure."

predictWithSummary_TreeClassifier_Model(fileName_Data)

prediction = predictWithsummary_TreeClassifier_Predict(fileName_Model,summary)
print(prediction)

"""





#########################################################################################################################################
########################################################### predictWithTitle ############################################################
#########################################################################################################################################
def predictWithTitle_TreeClassifier_Model (fileName : str) -> None :
    """
    Cette fonction permet de créer un model d'arbre de décision 
    
    Parameters
    ----------
        fileName : str
            nom du fichier source avec son chemin
    
    Returns
    ------- 
        None
    """
    
    lectureFichier = pandas.read_csv(fileName)
    lectureFichier = lectureFichier.sample(frac=1).reset_index(drop=True)
    
    

    listNameBook = lectureFichier['book_name']
    listGenre = lectureFichier['genre']
    
    tailleData = len(listNameBook)
    tailleTrain = int((20*tailleData)/100)
    
    listNameBook_train = listNameBook[:tailleTrain]
    listGenre_train = listGenre[:tailleTrain]
    
    listNameBook_test = listNameBook[tailleTrain:]
    listGenre_test = listGenre[tailleTrain:]
    
    listNameBook_train = tuple(listNameBook_train)
    listGenre_train = tuple(listGenre_train)
    
    listNameBook_test = tuple(listNameBook_test)
    listGenre_test = tuple(listGenre_test)
    
    
    
    # Vectorisation des titres avec TfidfVectorizer
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(listNameBook_train)
    
    # Création d'un classifieur RandomForest
    clf = RandomForestClassifier(n_estimators=150,
                                 criterion='gini',
                                 max_depth=None,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0.0,
                                 max_features='auto',
                                 max_leaf_nodes=None,
                                 min_impurity_decrease=0.0,
                                 bootstrap=True,
                                 oob_score=False,
                                 n_jobs=None,
                                 random_state=None,
                                 verbose=0,
                                 warm_start=False,
                                 class_weight=None,
                                 ccp_alpha=0.0,
                                 max_samples=None)
    
    # Entraînement du classifieur
    clf.fit(train_features, listGenre_train)
    
    
    nombreJuste = 0
    nombreFaux = 0
    predictionList = []
    
    for i in range (len(listNameBook_test)):
        book_feature = vectorizer.transform([listNameBook_test[i]])

        # Prédiction du genre du livre
        predicted_genre = clf.predict(book_feature)
        prediction = predicted_genre[0]
        predictionList.append (prediction)
        
        if (prediction == listGenre_test[i]) :
            nombreJuste = nombreJuste + 1
        else :
            nombreFaux = nombreFaux + 1
        
        
    taux_Forest = ((nombreJuste)/(nombreJuste+nombreFaux))*100
    mat_Forest = ConfusionMatrixDisplay(confusion_matrix(list(listGenre_test),predictionList,labels = ["Fantasy","Science Fiction","Crime Fiction","Historical novel","Horror","Thriller"]))
    mat_Forest.plot()
    plt.show()
    print("On peut voir que ",nombreJuste, "ont été bien prédit contre ",nombreFaux,"mal prédit")
    print("Le taux d'accuracy est de ", round(taux_Forest,2),"%")
    
    
    joblib.dump(clf, "Model/predictWithTitle_TreeClassifier")
    joblib.dump(vectorizer, "Model/predictWithTitle_vectorizer.joblib")
    
    return()


def predictWithTitle_TreeClassifier_Predict (fileName : str, title : str) -> str :
    """
    Cette fonction renvoie la prediction du genre littéraire en fonction d'un titre
    
    Parameters
    ----------
        title : str
            titre
        fileName : str
            nom du fichier source avec son chemin
    
    Returns
    ------- 
        genre : str
            prediction du genre
    """

    clf = joblib.load(fileName)
        
    # Vectorisation du titre à prédire
    vectorizer = joblib.load('Model/predictWithTitle_vectorizer.joblib')
    
    book_feature = vectorizer.transform([title])

    # Prédiction du genre du livre
    predicted_genre = clf.predict(book_feature)

    return(predicted_genre[0])


"""
## Test de la prediction avec le titre

book_title = 'The Kill Artist'

fileName_Data = "Données/BooksDataSet.csv"
fileName_Model = "Model/predictWithTitle_TreeClassifier"

predictWithTitle_TreeClassifier_Model(fileName_Data)
prediction = predictWithTitle_TreeClassifier_Predict(fileName_Model,book_title)

print("Pour le livre :", book_title,", le genre prédit est :", prediction)

"""





#########################################################################################################################################
########################################################### predictWithAuthor ############################################################
#########################################################################################################################################
def predictWithAuthor(author: str, nameFile : str) -> list:
    """
    Cette fonction renvoie la prediction du genre littéraire en fonction d'un nom d'auteur
    
    Parameters
    ----------
        author : str
            auteur
        nameFile : str
            nom du fichier contenant les auteurs et leurs genres
    
    Returns
    ------- 
        listGenre : list
            liste contenant les genres déjà écrit par l'auteur
    """
    
    listGenre : list = []
    
    lectureFichier = pandas.read_csv(nameFile, usecols=['Author', 'Genres'])
    listAuthor = lectureFichier['Author']
    listGenreSource = lectureFichier['Genres']
    
    for i in range (len(listAuthor)):
        if (listAuthor[i] == author) and (listGenreSource[i] not in listGenre):
            listGenre.append(listGenreSource[i])
    
    return (listGenre)

"""
## Test de predictWithAuthor

author = 'Stephen King'
nomFichier = 'Données/Author_generes.csv'

print(predictWithAuthor(author,nomFichier))

"""




