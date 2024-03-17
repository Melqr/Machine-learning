import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from PIL import Image


st.title('''Welcome to  Crops Insight: Integrating Machine Learning for Prediction and Suitability System''')
st.header('''What is Crops Insight: Integrating Machine Learning for Prediction and Suitability System?''')
st.write('''Crops Insight: Integrating Machine Learning for Prediction 
         and Suitability System is an innovative platform designed to empower farmers 
         with advanced machine learning techniques for optimizing crop yield and 
         recommending effective agricultural practices.''')

st.subheader("What is Machine Learning?")
st.write('''Machine learning is a branch of artificial intelligence (AI) and computer science
          which focuses on the use of data and algorithms to imitate the way that humans 
         learn, gradually improving its accuracy.''')

st.subheader("Problem Statement For This Topic")
st.write('''Farmers are keen on maximizing crop production for the benefit of all. In an ideal 
         scenario, they should have easy access to advanced prediction techniques combining 
         historical trends, climate patterns, and soil data. These tools would enable 
         farmers to input specific field data and receive accurate yield predictions.''')
st.write('''However, the reality is different. Many farmers, especially in developing regions, 
         lack the knowledge and resources for efficient crop analysis and yield prediction. 
         Relying on traditional methods hinders their productivity, leading to insufficient 
         yields and missed opportunities.''')
st.write('''To address this, a machine learning project is proposed. This research aims to 
         identify the best methods for predicting crop suitability. The resulting system will 
         empower farmers to switch to modern methods, enhancing their resources and boosting 
         productivity. Accurate predictions are crucial for adapting to market demands and 
         changing environmental conditions in agriculture.''')

st.subheader("Project's Objective")
st.write('''1) To identify a prediction technique for agricultural crop yield''')
st.write('''2) To design and develop predictive models that recommend optimized 
         agricultural techniques.''')
st.write('''3) To evaluate the accuracy of the prototype in predicting the correlation 
         with crop yields.''')

st.subheader("Project's Scope")
st.write('''1) This project is designed to reduce mistakes in agricultural practices by
          analyzing the agricultural crop yield.''')
st.write('''2) This project has the potential to classify the key factors influencing crop 
         yields including climate conditions, soil health, and farming practices.''')
st.write('''3) TThis project focuses on developing predictive models capable of recommending 
         optimized agricultural techniques.''')

