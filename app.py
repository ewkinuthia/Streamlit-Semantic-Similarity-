#!/usr/bin/env python
# -*- coding: utf-8 -*-
from msilib.schema import Directory
from operator import length_hint
import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle as pkl
from numpy.linalg import norm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title('Sematic Similarity')

def save_uploadedfile(datafiles):
    for uploaded_file in datafiles:
        with open(os.path.join("tempDir",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())

def create_dataframe(matrix, tokens): 
    dir = os.listdir( 'C:/Users/shish/Desktop/Files/SE491/tempDir' )
    data_file = [f for f in dir]
    doc_names = [ 'doc{i+1}' for i, _ in enumerate(matrix)]
    df = pd.DataFrame(data=matrix, index=data_file, columns=tokens)
    st.write(df)

maxUploadSize = 2000
datafiles = st.file_uploader("Upload Document", accept_multiple_files=True)
save_uploadedfile(datafiles)
   
if len(os.listdir( 'C:/Users/shish/Desktop/Files/SE491/tempDir') ) >= 2:
    dir = os.listdir( 'C:/Users/shish/Desktop/Files/SE491/tempDir' )
    data = [open(os.path.join("tempDir",f),errors="ignore").read() for f in dir]
    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(data)
    tokens = count_vectorizer.get_feature_names()
    vector_matrix.toarray()
    
    cosine_similarity_matrix = cosine_similarity(vector_matrix)
    data_file = [f for f in dir]
    
    create_dataframe(cosine_similarity_matrix, data_file )

if st.button('Remove Files'):
    directory= "C:/Users/shish/Desktop/Files/SE491/tempDir"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            os.remove(f)
