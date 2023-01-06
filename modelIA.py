#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Groupe Florent Kieffer & Louna Capel

"""

#####################################################################################################################
#################################################### Import ####################################################
#####################################################################################################################
from pydantic import BaseModel
from typing import Optional

#####################################################################################################################
##################################################### Class Book #####################################################
#####################################################################################################################
class Book(BaseModel) :
    title : Optional[str] = None
    author : Optional[str] = None
    summary : Optional[str] = None
    genre : Optional[str] = None
    identifiant : Optional[int] = None


#####################################################################################################################
################################################### Predictions ###################################################
#####################################################################################################################
"""
Renvoie la prediction du genre littéraire en fonction d'un résumé

Args :
    summary (str) : résumé

Returns : 
    genre (str) : prediction du genre
"""
def predictWithSummary(summary: str) -> str:
    return "Fiction"

"""
Renvoie la prediction du genre littéraire en fonction d'un titre

Args :
    title (str) : titre

Returns : 
    genre (str) : prediction du genre
"""
def predictWithTitle(title: str) -> str:
    return "Fiction"

"""
Renvoie la prediction du genre littéraire en fonction d'un nom d'auteur

Args :
    author (str) : auteur

Returns : 
    genre (str) : prediction du genre
"""
def predictWithAuthor(author: str) -> str:
    return "Fiction"
