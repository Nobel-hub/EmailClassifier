import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open('tf'))