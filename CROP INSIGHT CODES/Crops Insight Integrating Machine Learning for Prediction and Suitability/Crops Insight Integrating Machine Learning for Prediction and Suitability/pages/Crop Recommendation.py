import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from PIL import Image


st.title("Crop Recommendation")
st.header("What is Crop Recommendation?")
st.write('''Crop recommendation refers to the process of providing farmers with personalized 
		 advice on the selection of crops, cultivation practices, and resource management 
		 based on various factors such as soil conditions, climate, and historical data.''')
st.write('''The goal is to help farmers make informed decisions that optimize agricultural 
		 productivity, reduce environmental impact, and promote sustainable farming practices.''')

st.subheader("Models Chosen for This Machine")
st.write("The models that are chosen for this machine are Support Vector Machine (SVM), Decision Tree and Random Forest")

st.subheader("Crop Recommendation Classification Using Three Different Models")
st.write('''The goal is to help farmers make informed decisions that optimize agricultural 
         productivity, reduce environmental impact, and promote sustainable farming practices.
         ''')

selectView = st.sidebar.selectbox("Select View:", options = ["Select to View", "Variable Input & Target", "Dataset Training and Testing"])

data = pd.read_csv('Crop_recommendation(random).csv')
data = pd.get_dummies(data,columns=['N','P','K','temperature','humidity','ph','rainfall'],drop_first=True)
X = data.drop(columns=['label'])
y = data['label']


if selectView == "Variable Input & Target":
	st.write('''The chosen dataset contained 8 attributes that suitable for variable input 
		  like nitrogen level, phosphorus level, potassium level, temperature, local humidity, 
		  pH of soil, rainfall during the season, and crop suitability. The dataset has been 
		  manually random sorted in the Microsoft Excel before input into the coding.''')

	st.subheader("Dataset Variables Input Description")
	st.write('''Here are descriptions of the dataset features:

- **Nitrogen Level (N):** The level of nitrogen in the soil.
- **Phosphorus Level (P):** The level of phosphorus in the soil.
- **Potassium Level (K):** The level of potassium in the soil.
- **Temperature:** The temperature during the season.
- **Local Humidity:** The humidity of the local environment.
- **pH of Soil:** The pH level of the soil.
- **Rainfall:** The amount of rainfall during the season.
''', unsafe_allow_html=True)

	data = pd.read_csv('Crop_recommendation(random).csv')
	X = data.drop(columns=['label'])
	X

	st.subheader("Dataset Variables Target Description")
	st.write('''<span style="font-family: 'Arial, sans-serif'; color: green;">Here are descriptions of the dataset features:
	
- **label:** Crop Suitability.
		Crop suitability is a measure of how well a particular crop can thrive in the 
		 given environmental conditions, including factors like nitrogen, phosphorus, 
		 potassium levels, temperature, local humidity, pH of soil, and rainfall during 
		 the season. It helps in determining the most suitable crops for a specific region 
		 or set of conditions</span>''', unsafe_allow_html=True)

	y = data['label']
	y


# DECLARE DATA INPUT AND DATA TARGET FOR TRAINING AND TESTING DATA		
if selectView == "Dataset Training and Testing":
	st.header("Split Data Input (X) and Data Target (y) into Training and Testing Data")
	st.write('''The dataset has been manually split into training and testing sets. 
		  80% of the data is used for training, and the 
		  remaining 20% is used for testing.''')

	st.subheader("Data Training")
	CropRec_training = pd.read_csv('CropRecommend_Training.csv')
	st.write("Data Input From Training Set:")
	data_input_CropRec_training = CropRec_training.drop(columns='label')
	data_input_CropRec_training

	st.write("Data Target From Training Set:")
	data_target_CropRec_training = CropRec_training['label']
	data_target_CropRec_training


	st.subheader("Data Testing")
	CropRec_testing = pd.read_csv('CropRecommend_Testing.csv')
	st.write("Data From Input Testing Set:")
	data_input_CropRec_testing = CropRec_testing.drop(columns='label')
	data_input_CropRec_testing

	st.write("Data From Target Testing Set:")
	data_target_CropRec_testing = CropRec_testing['label']
	data_target_CropRec_testing
		
	
# SELECT MODEL FOR CLASSIFICATION
	selectModel = st.sidebar.selectbox("Select Model", options = ["Select Model", "Support Vector Machine (SVM)", "K-Nearest Neighbors", "Random Forest", "Compare All Models"])
	
	accuracy_scores = {}

	svclassifierLinear = SVC(kernel='linear')	
	svclassifierLinear.fit(data_input_CropRec_training, data_target_CropRec_training)
	predictSVMLinear = svclassifierLinear.predict(data_input_CropRec_testing)
	score_Linear = accuracy_score(predictSVMLinear, data_target_CropRec_testing)
		
	#--------------TRAIN SVM--------------------------
	svclassifierPoly = SVC(kernel='poly')	
	svclassifierPoly.fit(data_input_CropRec_training, data_target_CropRec_training)
	predictSVMPoly = svclassifierPoly.predict(data_input_CropRec_testing)
	score_Poly = accuracy_score(predictSVMPoly, data_target_CropRec_testing)
	
	svclassifierRbf = SVC(kernel='rbf')	
	svclassifierRbf.fit(data_input_CropRec_training, data_target_CropRec_training)
	predictSVMRbf = svclassifierRbf.predict(data_input_CropRec_testing)
	score_RBF = accuracy_score(predictSVMRbf, data_target_CropRec_testing)
	
	svclassifierSigmoid = SVC(kernel='sigmoid')
	svclassifierSigmoid.fit(data_input_CropRec_training, data_target_CropRec_training)
	predictSVMSigmoid = svclassifierSigmoid.predict(data_input_CropRec_testing)
	score_Sigmoid = accuracy_score(predictSVMSigmoid, data_target_CropRec_testing)


	#--------------TRAIN KNN--------------------------
	knn5 = KNeighborsClassifier(n_neighbors=5)
	knn5.fit(data_input_CropRec_training, data_target_CropRec_training)	
	knnPredict5 = knn5.predict(data_input_CropRec_testing)
	score_KNN5 = accuracy_score(knnPredict5, data_target_CropRec_testing)
		
	knn10 = KNeighborsClassifier(n_neighbors=10)
	knn10.fit(data_input_CropRec_training, data_target_CropRec_training)	
	knnPredict10 = knn10.predict(data_input_CropRec_testing)
	score_KNN10 = accuracy_score(knnPredict10, data_target_CropRec_testing)

	knn15 = KNeighborsClassifier(n_neighbors=15)
	knn15.fit(data_input_CropRec_training, data_target_CropRec_training)	
	knnPredict15 = knn15.predict(data_input_CropRec_testing)
	score_KNN15 = accuracy_score(knnPredict15, data_target_CropRec_testing)

	knn20 = KNeighborsClassifier(n_neighbors=20)
	knn20.fit(data_input_CropRec_training, data_target_CropRec_training)	
	knnPredict20 = knn20.predict(data_input_CropRec_testing)
	score_KNN20 = accuracy_score(knnPredict20, data_target_CropRec_testing)


	#--------------TRAIN RANDOM FOREST--------------------------
	rf5 = RandomForestClassifier(n_estimators=5, random_state=42)
	rf5.fit(data_input_CropRec_training, data_target_CropRec_training)	
	rfPredict5 = rf5.predict(data_input_CropRec_testing)
	scoreRF_5 = accuracy_score(rfPredict5, data_target_CropRec_testing)

	rf10 = RandomForestClassifier(n_estimators=10, random_state=42)
	rf10.fit(data_input_CropRec_training, data_target_CropRec_training)	
	rfPredict10 = rf10.predict(data_input_CropRec_testing)
	scoreRF_10 = accuracy_score(rfPredict10, data_target_CropRec_testing)

	rf15 = RandomForestClassifier(n_estimators=15, random_state=42)
	rf15.fit(data_input_CropRec_training, data_target_CropRec_training)	
	rfPredict15 = rf15.predict(data_input_CropRec_testing)
	scoreRF_15 = accuracy_score(rfPredict15, data_target_CropRec_testing)

	rf20 = RandomForestClassifier(n_estimators=20, random_state=42)
	rf20.fit(data_input_CropRec_training, data_target_CropRec_training)	
	rfPredict20 = rf20.predict(data_input_CropRec_testing)
	scoreRF_20 = accuracy_score(rfPredict20, data_target_CropRec_testing)


	# ---------------------------------Result SVM------------------------------------------------------
	if selectModel == "Support Vector Machine (SVM)":
		st.subheader("Support Vector Machine Crop Recommendation Classification Model")
		st.subheader("Predicted Results for Dataset Testing:")

		# ACCURACY SCORE IN TABLE
		results_df = pd.DataFrame({
			'Kernel': ["Linear", "Poly", "RBF", "Sigmoid"],
			'Accuracy Score': [score_Linear, score_Poly, score_RBF, score_Sigmoid]
		})

		st.write("Accuracy Score in Table")
		centered_table = f'<div style="text-align: center;">{results_df.to_html(index=False)}</div>'
		st.markdown(centered_table, unsafe_allow_html=True)

		# Find the highest accuracy for SVM
		highest_accuracy_svm = max(score_Linear, score_Poly, score_RBF, score_Sigmoid)
		kernel_svm = 'linear' if highest_accuracy_svm == score_Linear else \
              'poly' if highest_accuracy_svm == score_Poly else \
              'rbf' if highest_accuracy_svm == score_RBF else 'sigmoid'
		
		st.write(f"Highest Accuracy for SVM (Kernel: {kernel_svm}, Accuracy: {highest_accuracy_svm})")
		
		accuracy_scores["Highest_SVM"] = highest_accuracy_svm

		st.subheader("Conclusion")
		st.write('''Table above shows the results of the crop classification dataset by using the Support Vector Machine (SVM) model. 
		   Based on the results, the Linear kernel performed the best with an accuracy of 98.64% followed by the Polynomial 
		   (poly) kernel with 97.72% and the Radial Basis Function (RBF) with 96.59%. However, the Sigmod kernel had a significantly 
		   lower accuracy of 7.27%. Therefore, the Linear is chosen as the best kernel for SVM models and will be compared to others models.''')
		
		#USER INPUT FOR SVM MODEL
		st.header("User Input for SVM Model")
		st.subheader("Enter the following details:")
		nitrogen_level = st.number_input("Nitrogen Level in the soil", min_value=0.0, max_value=150.0, value=50.0)
		phosphorus_level = st.number_input("Phosphorus Level in the soil", min_value=0.0, max_value=150.0, value=50.0)
		potassium_level = st.number_input("Potassium Level in the soil", min_value=0.0, max_value=250.0, value=50.0)
		temperature = st.number_input("Temperature during the season", min_value=-10.0, max_value=45.0, value=25.0)
		humidity = st.number_input("Local Humidity", min_value=0.0, max_value=100.0, value=100.0)
		ph_of_soil = st.number_input("pH of Soil", min_value=0.0, max_value=14.0, value=12.0)
		rainfall = st.number_input("Rainfall during the season", min_value=0.0, max_value=1000.0, value=500.0)

		# User data
		user_data = pd.DataFrame({
    		'N': [nitrogen_level],
    		'P': [phosphorus_level],
    		'K': [potassium_level],
    		'temperature': [temperature],
    		'humidity': [humidity],
    		'ph': [ph_of_soil],
    		'rainfall': [rainfall]
		})

		 # Prediction and display results for each kernel
		st.subheader("SVM Results Based on User Input:")
		for kernel in ["linear", "poly", "rbf", "sigmoid"]:
			# Train SVM with selected kernel type
			svclassifier = SVC(kernel=kernel)
			svclassifier.fit(data_input_CropRec_training, data_target_CropRec_training)
			
			# Prediction
			svm_prediction = svclassifier.predict(user_data)
			
			# Display results for each kernel
			st.write(f"SVM Prediction (Kernel: {kernel}): {svm_prediction[0]}")


	# ---------------------------------TRAIN KNN------------------------------------------------------
	elif selectModel == "K-Nearest Neighbors":
		st.subheader("K-Nearest Neighbors Crop Recommendation Classification Model")
		st.subheader("Predicted Results for Dataset Testing:")

		# ACCURACY SCORE IN TABLE
		results_df = pd.DataFrame({
			'Number of  Neighbors': [5, 10, 15, 20],
			'Accuracy Score': [score_KNN5, score_KNN10, score_KNN15, score_KNN20]
		})

		st.write("Accuracy Score in Table")
		centered_table = f'<div style="text-align: center;">{results_df.to_html(index=False)}</div>'
		st.markdown(centered_table, unsafe_allow_html=True)

		# Find the highest accuracy for SVM
		highest_accuracy_knn = max(score_KNN5, score_KNN10, score_KNN15, score_KNN20)
		bestKNN = 5 if highest_accuracy_knn == score_KNN5 else \
              10 if highest_accuracy_knn == score_KNN10 else \
              15 if highest_accuracy_knn == score_KNN15 else 20
		
		st.write(f"Highest Accuracy for KNN (Number of Neighbors: {bestKNN}, Accuracy: {highest_accuracy_knn})")
		
		accuracy_scores["Highest_KNN"] = highest_accuracy_knn

		st.subheader("Conclusion")
		st.write('''Based on the results, the use of 5 neighbors resulted in the highest accuracy of 97.50%. 
		   As the number of neighbors increased, the accuracy slightly decreased, with 96.59% for 10 neighbors, 
		   95.90% for 15 neighbors, and 96.13% for 20 neighbors. Therefore, the choice of 5 neighbors in the KNN 
		   algorithm seems to provide the best performance and will represent KNN compared with other models.''')
		
		#USER INPUT FOR KNN MODEL
		st.header("User Input for KNN Model")
		st.subheader("Enter the following details:")
		nitrogen_level = st.number_input("Nitrogen Level in the soil", min_value=0.0, max_value=150.0, value=50.0)
		phosphorus_level = st.number_input("Phosphorus Level in the soil", min_value=0.0, max_value=150.0, value=50.0)
		potassium_level = st.number_input("Potassium Level in the soil", min_value=0.0, max_value=250.0, value=50.0)
		temperature = st.number_input("Temperature during the season", min_value=-10.0, max_value=45.0, value=25.0)
		humidity = st.number_input("Local Humidity", min_value=0.0, max_value=100.0, value=100.0)
		ph_of_soil = st.number_input("pH of Soil", min_value=0.0, max_value=14.0, value=12.0)
		rainfall = st.number_input("Rainfall during the season", min_value=0.0, max_value=1000.0, value=500.0)

		# User data
		user_data = pd.DataFrame({
    		'N': [nitrogen_level],
    		'P': [phosphorus_level],
    		'K': [potassium_level],
    		'temperature': [temperature],
    		'humidity': [humidity],
    		'ph': [ph_of_soil],
    		'rainfall': [rainfall]
		})

		 # Prediction and display results for each kernel
		st.subheader("KNN Results Based on User Input:")
		for neighbors in [5, 10, 15, 20]:
			# Train KNN with selected kernel type
			knn_classifier = KNeighborsClassifier(n_neighbors=neighbors)
			knn_classifier.fit(data_input_CropRec_training, data_target_CropRec_training)
			
			# Prediction
			knn_prediction = knn_classifier.predict(user_data)
			
			# Display results for each kernel
			st.write(f"KNN Prediction (Number of Neighbors: {neighbors}): {knn_prediction[0]}")
	
	# ---------------------------------TRAIN RANDOM FOREST------------------------------------------------------
	elif selectModel == "Random Forest":
		st.subheader("Random Forest Crop Recommendation Classification Model")
		st.subheader("Predicted Results for Dataset Testing:")

		# ACCURACY SCORE IN TABLE
		results_df = pd.DataFrame({
			'Number of  Estimators': [5, 10, 15, 20],
			'Accuracy Score': [scoreRF_5, scoreRF_10, scoreRF_15, scoreRF_20]
		})

		st.write("Accuracy Score in Table")
		centered_table = f'<div style="text-align: center;">{results_df.to_html(index=False)}</div>'
		st.markdown(centered_table, unsafe_allow_html=True)

		# Find the highest accuracy for SVM
		highest_accuracy_rf = max(scoreRF_5, scoreRF_10, scoreRF_15, scoreRF_20)
		bestRF = 5 if highest_accuracy_rf == scoreRF_5 else \
              10 if highest_accuracy_rf == scoreRF_10 else \
              15 if highest_accuracy_rf == scoreRF_15 else 20
		
		st.write(f"Highest Accuracy for Random Forest (Number of Estimators: {bestRF}, Accuracy: {highest_accuracy_rf})")
		
		accuracy_scores["Highest_RF"] = highest_accuracy_rf

		st.subheader("Conclusion")
		st.write('''Based on the results, it can be concluded that the model consistently performed well. 
		   Specifically, the accuracy scores were 98.86% for 5 estimators, 99.09% for 10 estimators, 99.31% 
		   for both 15 and 20 estimators. This suggests that the KNN algorithm performed exceptionally well 
		   on the given dataset, with minimal variation in accuracy as the number of estimators changed. 
		   Therefore, the result for both 15 and 20 estimators will be chosen to be compared with the other 
		   algorithm’s models.''')
		
		#USER INPUT FOR RANDOM FOREST MODEL
		st.header("User Input for Random Forest Model")
		st.subheader("Enter the following details:")
		nitrogen_level = st.number_input("Nitrogen Level in the soil", min_value=0.0, max_value=150.0, value=50.0)
		phosphorus_level = st.number_input("Phosphorus Level in the soil", min_value=0.0, max_value=150.0, value=50.0)
		potassium_level = st.number_input("Potassium Level in the soil", min_value=0.0, max_value=250.0, value=50.0)
		temperature = st.number_input("Temperature during the season", min_value=-10.0, max_value=45.0, value=25.0)
		humidity = st.number_input("Local Humidity", min_value=0.0, max_value=100.0, value=100.0)
		ph_of_soil = st.number_input("pH of Soil", min_value=0.0, max_value=14.0, value=12.0)
		rainfall = st.number_input("Rainfall during the season", min_value=0.0, max_value=1000.0, value=500.0)

		# User data
		user_data = pd.DataFrame({
    		'N': [nitrogen_level],
    		'P': [phosphorus_level],
    		'K': [potassium_level],
    		'temperature': [temperature],
    		'humidity': [humidity],
    		'ph': [ph_of_soil],
    		'rainfall': [rainfall]
		})

		 # Prediction and display results for each kernel
		st.subheader("Random Forest Results Based on User Input:")
		for estimators in [5, 10, 15, 20]:
			# Train KNN with selected kernel type
			rf_classifier = RandomForestClassifier(n_estimators=estimators)
			rf_classifier.fit(data_input_CropRec_training, data_target_CropRec_training)
			
			# Prediction
			rf_prediction = rf_classifier.predict(user_data)
			
			# Display results for each kernel
			st.write(f"Random Forest Prediction (Number of Estimators: {estimators}): {rf_prediction[0]}")
	
	# ---------------------------------COMPARE ALL MODELS------------------------------------------------------
	elif selectModel == "Compare All Models":
		# Find the highest accuracy for SVM
		highest_accuracy_svm = max(score_Linear, score_Poly, score_RBF, score_Sigmoid)
		kernel_svm = 'linear' if highest_accuracy_svm == score_Linear else \
              'poly' if highest_accuracy_svm == score_Poly else \
              'rbf' if highest_accuracy_svm == score_RBF else 'sigmoid'		
		accuracy_scores["Highest_SVM"] = highest_accuracy_svm


		# Find the highest accuracy for KNN
		highest_accuracy_knn = max(score_KNN5, score_KNN10, score_KNN15, score_KNN20)
		bestKNN = 5 if highest_accuracy_knn == score_KNN5 else \
              10 if highest_accuracy_knn == score_KNN10 else \
              15 if highest_accuracy_knn == score_KNN15 else 20		
		accuracy_scores["Highest_KNN"] = highest_accuracy_knn

		
		# Find the highest accuracy for Random Forest
		highest_accuracy_rf = max(scoreRF_5, scoreRF_10, scoreRF_15, scoreRF_20)
		bestRF = 5 if highest_accuracy_rf == scoreRF_5 else \
              10 if highest_accuracy_rf == scoreRF_10 else \
              15 if highest_accuracy_rf == scoreRF_15 else 20
		accuracy_scores["Highest_RF"] = highest_accuracy_rf

		# ACCURACY SCORE IN TABLE
		results_df = pd.DataFrame({
			'Models': ["SVM", "KNN", "Random Forest"],
			'Accuracy Score': [highest_accuracy_svm, highest_accuracy_knn, highest_accuracy_rf]
		})

		st.header("Comparison of Accuracy Score for Each Model")
		centered_table = f'<div style="text-align: center;">{results_df.to_html(index=False)}</div>'
		st.markdown(centered_table, unsafe_allow_html=True)

		# Find the highest accuracy for among three model
		if "Highest_SVM" in accuracy_scores and "Highest_KNN" in accuracy_scores and "Highest_RF" in accuracy_scores:
			highest_accuracies = {
				"SVM": accuracy_scores["Highest_SVM"],
				"KNN": accuracy_scores["Highest_KNN"],
				"Random Forest": accuracy_scores["Highest_RF"]
				}
		
			highest_accuracy_model = max(highest_accuracies, key=highest_accuracies.get)
			st.write(f"Highest Accuracy Model: {highest_accuracy_model}, Accuracy: {highest_accuracies[highest_accuracy_model]}")
		
		st.subheader("Conclusion")
		st.write('''In conclusion, all three machine learning models, namely SVM, KNN, and Random Forest, 
		   proved to be highly effective in predicting crop recommendations. The small variations in accuracy 
		   highlight the suitability of each algorithm for the task. The Random Forest model, with the highest 
		   accuracy at 99.31%, stands out as the most reliable choice for accurate crop recommendations in the 
		   given dataset.''')
		

	
