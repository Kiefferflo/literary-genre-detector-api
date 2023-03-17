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
import numpy as np
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


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
     
        
    with open('app/Données/Author_generes.csv', 'w') as f:
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
    tailleTrain = int((80*tailleData)/100)
    
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

#predictWithSummary_TreeClassifier_Model(fileName_Data)

prediction = predictWithsummary_TreeClassifier_Predict(fileName_Model,summary)

print("Pour le résumé :", summary,", le genre prédit est :", prediction)
"""


#########################################################################################################################################
#########################################################################################################################################


def predictWithSummary_NN_Model (fileName: str) -> None:
    """
    Cette fonction renvoie la prediction du genre littéraire en fonction d'un résumé
    
    Parameters
    ----------
        fileName : str
            nom du fichier
    
    Returns
    -------
        None
    """
    
    # Charger le fichier csv
    df = pandas.read_csv(fileName)
    
    # Séparer les données en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(df['summary'], df['genre'], test_size=0.2, random_state=42)
    
    # Convertir les résumés en vecteurs de nombres
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    
    X_train = tokenizer.texts_to_matrix(X_train, mode='tfidf')
    X_test = tokenizer.texts_to_matrix(X_test, mode='tfidf')
    
    # Convertir les étiquettes de catégorie en vecteurs binaires
    num_classes = y_train.nunique()
    
    # Convertir les étiquettes en entiers
    label_to_int = {"Fantasy": 0, "Science Fiction": 1, "Crime Fiction": 2, "Historical novel" : 3,"Horror" : 4,"Thriller" : 5}
    y_train = y_train.map(label_to_int)
    y_test = y_test.map(label_to_int)
    
    # Convertir les étiquettes en vecteurs binaires
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    
    # Définir le modèle
    model = Sequential()
    model.add(Dense(512, input_shape=(5000,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))
    
    # Compiler le modèle
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Entraîner le modèle
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
    
    nombreJuste = 0
    nombreFaux = 0
    predictionList = []
    
    for i in range (len(X_test)):
    
        # Prédiction du genre du livre
        predicted_genre = model.predict(X_test[i].reshape(1, 5000), verbose=0)
        prediction = predicted_genre[0].argmax()
        predictionList.append(prediction)
    
        if (prediction == y_test[i].argmax()) :
            nombreJuste = nombreJuste + 1
        else :
            nombreFaux = nombreFaux + 1

        
    y_test = np.argmax(y_test, axis=1)

    taux_Forest = ((nombreJuste)/(nombreJuste+nombreFaux))*100
    mat_Forest = ConfusionMatrixDisplay(confusion_matrix(list(y_test),predictionList))
    mat_Forest.plot()
    plt.show()
    print("On peut voir que ",nombreJuste, "ont été bien prédit contre ",nombreFaux,"mal prédit")
    print("Le taux d'accuracy est de ", round(taux_Forest,2),"%")
    
    joblib.dump(model, "Model/predictWithSummary_NN")
    
    return()


def predictWithSummary_NN_Predict (fileNameModel : str, fileNameDonnes : str, summary : str) -> str :
    """
    Cette fonction renvoie la prediction du genre littéraire en fonction d'un titre
    
    Parameters
    ----------
        summary : str
            résumé
        fileNameModel : str
            nom du fichier model avec son chemin
        fileNameDonnees : str
            nom du fichier données avec son chemin
    
    Returns
    ------- 
        genre : str
            prediction du genre
    """
    
    model = joblib.load(fileNameModel)
    # Charger le fichier csv
    df = pandas.read_csv(fileNameDonnees)
    
    # Convertir le résumé en vecteur de nombres
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['summary'])
    summary = tokenizer.texts_to_matrix([summary], mode='tfidf')
    
    # Prédire les probabilités du résumé appartenant à chacune des six classes
    proba = model.predict(summary, verbose=0)
    
    # Trouver la classe ayant la plus haute probabilité
    predicted_class = proba.argmax(axis=-1)
    
    if (predicted_class==0):
        prediction = "Fantasy"
    elif (predicted_class==1):
        prediction = "Science Fiction"
    elif (predicted_class==2):
        prediction = "Crime Fiction"
    elif (predicted_class==3):
        prediction = "Historical novel"
    elif (predicted_class==4):
        prediction = "Horror" 
    else :
        prediction = "Thriller"

    return(prediction)


##test de la prediction du genre NN

summary = "The book tells the story of two friends embarking on an incredible adventure."
fileNameModel = "Model/predictWithSummary_NN"
fileNameDonnees = "Données/BooksDataSet.csv"

predictWithSummary_NN_Model('Données/BooksDataSet.csv')
prediction = predictWithSummary_NN_Predict(fileNameModel,fileNameDonnees,summary)

print("le genre prédit pour le résumé :", summary,"est :", prediction)




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
    tailleTrain = int((80*tailleData)/100)
    
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
#########################################################################################################################################


def predictWithTitle_NN_Model (fileName: str) -> None:
    """
    Cette fonction renvoie la prediction du genre littéraire en fonction d'un titre
    
    Parameters
    ----------
        fileName : str
            nom du fichier
    
    Returns
    -------
        None
    """
    
    # Charger le fichier csv
    df = pandas.read_csv(fileName)
    
    # Séparer les données en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(df['book_name'], df['genre'], test_size=0.2, random_state=42)
    
    # Convertir les résumés en vecteurs de nombres
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    
    X_train = tokenizer.texts_to_matrix(X_train, mode='tfidf')
    X_test = tokenizer.texts_to_matrix(X_test, mode='tfidf')
    
    # Convertir les étiquettes de catégorie en vecteurs binaires
    num_classes = y_train.nunique()
    
    # Convertir les étiquettes en entiers
    label_to_int = {"Fantasy": 0, "Science Fiction": 1, "Crime Fiction": 2, "Historical novel" : 3,"Horror" : 4,"Thriller" : 5}
    y_train = y_train.map(label_to_int)
    y_test = y_test.map(label_to_int)
    
    # Convertir les étiquettes en vecteurs binaires
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    
    # Définir le modèle
    model = Sequential()
    model.add(Dense(512, input_shape=(5000,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))
    
    # Compiler le modèle
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Entraîner le modèle
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))
    
    nombreJuste = 0
    nombreFaux = 0
    predictionList = []
    
    for i in range (len(X_test)):
    
        # Prédiction du genre du livre
        predicted_genre = model.predict(X_test[i].reshape(1, 5000), verbose=0)
        prediction = predicted_genre[0].argmax()
        predictionList.append(prediction)
    
        if (prediction == y_test[i].argmax()) :
            nombreJuste = nombreJuste + 1
        else :
            nombreFaux = nombreFaux + 1

        
    y_test = np.argmax(y_test, axis=1)

    taux_Forest = ((nombreJuste)/(nombreJuste+nombreFaux))*100
    mat_Forest = ConfusionMatrixDisplay(confusion_matrix(list(y_test),predictionList))
    mat_Forest.plot()
    plt.show()
    print("On peut voir que ",nombreJuste, "ont été bien prédit contre ",nombreFaux,"mal prédit")
    print("Le taux d'accuracy est de ", round(taux_Forest,2),"%")
    
    joblib.dump(model, "Model/predictWithTitle_NN")
    
    return()


def predictWithTitle_NN_Predict (fileNameModel : str, fileNameDonnes : str, title : str) -> str :
    """
    Cette fonction renvoie la prediction du genre littéraire en fonction d'un titre
    
    Parameters
    ----------
        title : str
            Titre
        fileNameModel : str
            nom du fichier model avec son chemin
        fileNameDonnees : str
            nom du fichier données avec son chemin
    
    Returns
    ------- 
        genre : str
            prediction du genre
    """
    
    model = joblib.load(fileNameModel)
    # Charger le fichier csv
    df = pandas.read_csv(fileNameDonnees)
    
    # Convertir le résumé en vecteur de nombres
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['book_name'])
    title = tokenizer.texts_to_matrix([title], mode='tfidf')
    
    # Prédire les probabilités du résumé appartenant à chacune des six classes
    proba = model.predict(title, verbose=0)
    
    # Trouver la classe ayant la plus haute probabilité
    predicted_class = proba.argmax(axis=-1)
    
    if (predicted_class==0):
        prediction = "Fantasy"
    elif (predicted_class==1):
        prediction = "Science Fiction"
    elif (predicted_class==2):
        prediction = "Crime Fiction"
    elif (predicted_class==3):
        prediction = "Historical novel"
    elif (predicted_class==4):
        prediction = "Horror" 
    else :
        prediction = "Thriller"

    return(prediction)

"""
##test de la prediction du genre NN

predictWithTitle_NN_Model('Données/BooksDataSet.csv')

title = "The Kill Artist"
fileNameModel = "Model/predictWithTitle_NN"
fileNameDonnees = "Données/BooksDataSet.csv"

prediction = predictWithTitle_NN_Predict(fileNameModel,fileNameDonnees,title)

print("Le genre prédit pour le titre :", title,"est :", prediction)
"""



#########################################################################################################################################
########################################################### predictWithAuthor ###########################################################
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

print("Les romans déjà écrit par", author,"sont :",predictWithAuthor(author,nomFichier))
"""



#########################################################################################################################################
########################################################### predictTitle ###########################################################
#########################################################################################################################################
def predictTitle(summary):
    """
    Cette fonction renvoie la prediction d'un titre en fonction de son résumé
    
    Parameters
    ----------
        summary : str
            résumé

    
    Returns
    ------- 
        Title : str
            prédiction du titre
    """
    data = pandas.read_csv('Données/BooksDataSet.csv')
    
    # Vectoriser les résumés de livres en utilisant la méthode TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(data['summary'])
    
    # Calculer les similarités cosinus entre les vecteurs de résumé de livre
    cosine_similarities = cosine_similarity(vectors)
        
    # Vectoriser le résumé
    summary_vector = vectorizer.transform([summary])
    
    # Calculer les similarités cosinus entre le résumé du livre donné et tous les autres résumés
    similarity_scores = cosine_similarity(summary_vector, vectors)[0]
    
    # Trouver l'index de la livre le plus similaire
    most_similar_book_index = np.argmax(similarity_scores)
    
    # Retourner le titre du livre le plus similaire
    title = data.loc[most_similar_book_index]['book_name']
    
    return (title)


"""
## Test de predictTitle

summary = "The story follows a man named Winston Smith, who works for the government and begins to rebel against its oppressive rule. As he becomes involved in a forbidden love affair and joins a secret resistance movement, Winston must navigate the dangerous world of surveillance and propaganda to fight for his freedom and individuality."
predicted_title = predictTitle(summary)

print("Le titre prédit pour le résumé :",summary," est :",predicted_title)
"""
