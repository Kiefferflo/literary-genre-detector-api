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


#########################################################################################################################################
######################################################### Nettoyage des données #########################################################
#########################################################################################################################################


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


def stemming(lemSummary):
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

## test de la partie nettoyage du texte ##


summary = "The book tells the story of two friends embarking on an incredible adventure."
print(summary)
Stop = removeStopWords(summary)
print(Stop)
sentence = lematizing(Stop)
print(sentence)
stem = stemming(sentence)
print(stem)
"""

########################################################################################################################################
########################################################## Création BDD Autor ##########################################################
########################################################################################################################################


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
############################################################## Predictions ##############################################################
#########################################################################################################################################
def predictWithSummary(summary: str) -> str:
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
    
    return "Fiction"


def predictWithTitle(title: str) -> str:
    """
    Cette fonction renvoie la prediction du genre littéraire en fonction d'un titre
    
    Parameters
    ----------
        title : str
            titre
    
    Returns
    ------- 
        genre : str
            prediction du genre
    """
    return "Fiction"


def predictWithAuthor(author: str) -> str:
    """
    Cette fonction renvoie la prediction du genre littéraire en fonction d'un nom d'auteur
    
    Parameters
    ----------
        author : str
            auteur
    
    Returns
    ------- 
        genre : str
            prediction du genre
    """
    return "Fiction"














































