import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn import datasets
from sklearn import preprocessing
 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from PIL import Image


st.title("Rice Pest Disease")
st.header("What is Rice Pest Disease?")
st.write('''Rice pests and diseases refer to the various organisms and pathological agents 
		 that adversely affect the growth, health, and yield of rice crops. These can include 
		 insects, pathogens, and environmental factors that pose threats to the overall 
		 well-being of rice plants.''')
st.write('''Managing and preventing rice pests and diseases are critical aspects of 
		 agricultural practices, as they can have profound implications on global food 
		 security. Farmers employ various strategies, including the use of resistant 
		 rice varieties, integrated pest management, and timely application of fungicides or 
		 insecticides, to mitigate the impact of pests and diseases on rice crops.''')

st.subheader("Models Chosen for This Machine")
st.write("The models that are chosen for this machine are Support Vector Machine (SVM), Decision Tree and Random Forest")

st.subheader("Rice Pest Disease Classification Using Three Different Models")
st.write('''The goal is to help farmers make informed decisions that optimize agricultural 
         productivity, reduce environmental impact, and promote sustainable farming practices.
         ''')

selectView = st.sidebar.selectbox("Select View:", options = ["Select to View", "Variable Input & Target", "Dataset Training and Testing"])


if selectView == "Variable Input & Target":

	st.write('''The chosen dataset contained 14 attributes that can be used to predict and 
		  classify potential threats to rice crops.''')
	st.subheader("Dataset Variables Input Description")
	st.write('''Here are descriptions of the dataset features:

- **Observation Year:** The year in which the observation was recorded.
- **Standard Week:** The standard week corresponding to the observation.
- **Pest Value:** Value associated with the presence or intensity of pests.
- **Collection Type:** Type of collection method for the data.
- **Maximum Temperature (MaxT):** Highest temperature recorded.
- **Minimum Temperature (MinT):** Lowest temperature recorded.
- **Relative Humidity 1 (MAX):** Maximum relative humidity.
- **Relative Humidity 2 (MIN):** Minimum relative humidity.
- **Rainfall:** Amount of rainfall during the observation period.
- **Wind Speed:** Speed of the wind during the observation.
- **Sunshine Hours:** Duration of sunshine during the observation.
- **Evaporation (EVP mm):** Evaporation measured in millimeters.
- **PEST NAME:** Name of the pest observed.
- **Location:** Geographic location where the observation took place.
''', unsafe_allow_html=True)

	data = pd.read_csv('RICE.csv')
	X = data.drop(columns = ["Observation Year", "PEST NAME","Location"])
	X

	st.subheader("Dataset Variables Target Description")
	st.write('''<span style="font-family: 'Arial, sans-serif'; color: green;">Here are descriptions of the dataset features:
	
- **label:** Crop Suitability.
		Crop suitabiliitions</span>''', unsafe_allow_html=True)

	y = data["PEST NAME"]
	y


# DECLARE DATA INPUT AND DATA TARGET FOR TRAINING AND TESTING DATA		
if selectView == "Dataset Training and Testing":
	st.write('''In this dataset, the total number of rows is 19.4k and 14 columns which is a 
		  lot of data. With this, the dataset will need to be split manually into the training 
		  dataset and testing dataset. A new column is to be added to the dataset which is 
		  ‘bil’ to give the dataset a number to sort by, then a new column is created to 
		  randomize the data set so it will not be biased. The function =RANDBETWEEN(1,19400) 
		  is used to create a column of randomized numbers that are then sorted''')
	st.write('''The dataset has been manually split into training and testing sets. 
		  80% of the data is used for training, and the 
		  remaining 20% is used for testing.''')

	st.header("Split Data Input (X) and Data Target (y) into Training and Testing Data")
	st.subheader("Data Training")
	dataTrng  = pd.read_csv('TrainingRiceData.csv')
	st.write("Data From Input Training Set")
	data_input_training = dataTrng.drop(columns = ["Observation Year", "PEST NAME","Location"])
	data_input_training

	st.write("Data From Target Training Set")
	data_target_training = dataTrng["PEST NAME"]
	data_target_training


	st.subheader("Data Testing")
	dataTest = pd.read_csv("TestingRiceData.csv")
	st.write("Data From Input Testing Set")
	data_input_testing = dataTest.drop(columns = ["Observation Year", "PEST NAME","Location"])
	data_input_testing

	st.write("Data From Target Testing Set")
	data_target_testing = dataTest ["PEST NAME"]
	data_target_testing
		
	
# SELECT MODEL FOR CLASSIFICATION
	selectModel = st.sidebar.selectbox("Select Model", options = ["Select Model", "Support Vector Machine (SVM)", "K-Nearest Neighbors", "Random Forest", "Compare All Models"])

	accuracy_scores  = {}

	# ---------------------------------TRAIN SVM------------------------------------------------------
	kernel_values_svm = []
	accuracy_values_svm = []
		
	kernels = ['sigmoid', 'poly', 'rbf', 'linear']
	for kernel in kernels:
		svm_model = f'model_{kernel}.joblib'
		try:
			SVCKr = joblib.load(svm_model)
		except FileNotFoundError:
			SVCKr = SVC(kernel=kernel)
			SVCKr.fit(data_input_training, data_target_training)
			joblib.dump(SVCKr, svm_model)

		ResltSVM=SVCKr.predict(data_input_testing)
		accuracy = accuracy_score(ResltSVM, data_target_testing)
		kernel_values_svm.append(kernel)
		accuracy_values_svm.append(accuracy)			
	
	# ---------------------------------TRAIN KNN------------------------------------------------------
	knn_values = []
	accuracy_values_knn = []

	n_neighbors_list = [5, 10, 15, 20]
	for n_neighbors in n_neighbors_list:
		neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
		neigh.fit(data_input_training, data_target_training)
		RsltKnn = neigh.predict(data_input_testing)
		accuracy = accuracy_score(RsltKnn, data_target_testing)
		knn_values.append(n_neighbors)
		accuracy_values_knn.append(accuracy)

	# ---------------------------------TRAIN RANDOM FOREST------------------------------------------------------
	rf_values = []
	accuracy_values_rf = []
		
	n_estimators_list = [5, 10, 15, 20]
	for n_estimators in n_estimators_list:
		random_forest_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
		random_forest_model.fit(data_input_training, data_target_training)
		RsltRf = random_forest_model.predict(data_input_testing)
		accuracy = accuracy_score(RsltRf, data_target_testing)
		rf_values.append(n_estimators)
		accuracy_values_rf.append(accuracy)



	if selectModel == "Support Vector Machine (SVM)":
		st.subheader("Support Vector Machine Wheat Variety Classification Model")
		st.subheader("Predicted Results for Dataset Testing:")

		st.write("Accuracy Score for SVM")

		# ACCURACY SCORE IN TABLE
		st.write("Accuracy Score in Table")
		results_df_svr = pd.DataFrame({
			'Kernel': kernel_values_svm,
			'Accuracy Score': accuracy_values_svm
        })
		centered_table_svr = f'<div style="text-align: center;">{results_df_svr.to_html(index=False)}</div>'
		st.markdown(centered_table_svr, unsafe_allow_html=True)

		st.subheader("Conclusion")
		st.write('''Based on the table above the Linear kernel has the highest accuracy at 24.50%. Meanwhile the Poly 
		   and RBF kernel are neck to neck at 19.25% and 20.00% respectively. The sigmoids are the lowest of the accuracy 
		   at  17.75%. Then the kernel chosen to represent the SVM model is the Linear kernel which has the highest of 
		   the four accuray scores at 24.50%. ''')
		
		#USER INPUT FOR SVM MODEL
		st.header("User Input for SVM Model")
		st.subheader("Enter the following details:")
		Standard_Week = st.number_input("Week of Year", min_value=0.0, max_value=20.0, value=15.0)
		Pest_Value = st.number_input("Pest Value", min_value=0.0, max_value=10000.0, value=500.0)
		Max_Temperature = st.number_input("Maximum Temperature", min_value=0.0, max_value=50.0, value=30.)
		Min_Temperature = st.number_input("Minimum Temperature", min_value=0.0, max_value=50.0, value=25.0)
		Max_Humidity = st.number_input("Maximum Relative Humidity", min_value=-10.0, max_value=100.0, value=90.0)
		Min_Humidity = st.number_input("Minimum Relative Humidity", min_value=0.0, max_value=100.0, value=50.0)
		Rainfall = st.number_input("Amount Rainfall", min_value=0.0, max_value=200.0, value=50.0)
		Wind_Speed = st.number_input("The Speed of Wind", min_value=0.0, max_value=20.0, value=5.0)
		Sunshine_Hours = st.number_input("The Amount of Hour Sunlight Exposure", min_value=0.0, max_value=15.0, value=8.0)
		Evaporation = st.number_input("Amount of Water Evaporation", min_value=0.0, max_value=50.0, value=25.0)

		# User data
		user_data = pd.DataFrame({
    		'Standard Week': [Standard_Week],
    		'Pest Value': [Pest_Value],
    		'MaxT': [Max_Temperature],
			'MinT': [Min_Temperature],
    		'RH1(%)': [Max_Humidity],
    		'RH2(%)': [Min_Humidity],
    		'RF(mm)': [Rainfall],
    		'WS(kmph)': [Wind_Speed],
			'SSH(hrs)': [Sunshine_Hours],
			'EVP(mm)': [Evaporation]
		})

		 # Prediction and display results for each kernel
		st.subheader("SVM Results Based on User Input:")
		for kernel in ["linear", "poly", "rbf", "sigmoid"]:
			# Train KNN with selected kernel type
			svm_model = f'model_{kernel}.joblib'
			SVCKr = joblib.load(svm_model)

			# Prediction
			svm_prediction = SVCKr.predict(user_data)
			
			# Display results for each kernel
			st.write(f"SVM Prediction (Kernel: {kernel}): {svm_prediction[0]}")


	elif selectModel == "K-Nearest Neighbors":
		st.subheader("K-Nearest Neighbors Rice Pest and Disease Classification Model")
		st.subheader("Predicted Results for Dataset Testing:")

		st.write("Accuracy Score for KNN")

		# ACCURACY SCORE IN TABLE
		st.write("Accuracy Score in Table")
		results_df_svr = pd.DataFrame({
			'Number of Neighbors': knn_values,
			'Accuracy Score': accuracy_values_knn
        })
		centered_table_svr = f'<div style="text-align: center;">{results_df_svr.to_html(index=False)}</div>'
		st.markdown(centered_table_svr, unsafe_allow_html=True)

		st.subheader("Conclusion")
		st.write('''Based on the results, the highest accuracy score between the 4 numbers of neighbors is 
		   when the n is at 15 with the accuracy score of 24.50%. When the n is at 15 and 20 is at 24.50% 
		   and 24.00% respectively. They have the same accuracy score with minor differences. The lowest of 
		   them all is when the n is at 5 which has the score of 22.75%. Hence the number of neighbors that
		    will represent the model for the KNN model is when the n is at 15..''')
		
		#USER INPUT FOR KNN MODEL
		st.header("User Input for KNN Model")
		st.subheader("Enter the following details:")
		Standard_Week = st.number_input("Week of Year", min_value=0.0, max_value=20.0, value=15.0)
		Pest_Value = st.number_input("Pest Value", min_value=0.0, max_value=10000.0, value=500.0)
		Max_Temperature = st.number_input("Maximum Temperature", min_value=0.0, max_value=50.0, value=30.)
		Min_Temperature = st.number_input("Minimum Temperature", min_value=0.0, max_value=50.0, value=25.0)
		Max_Humidity = st.number_input("Maximum Relative Humidity", min_value=-10.0, max_value=100.0, value=90.0)
		Min_Humidity = st.number_input("Minimum Relative Humidity", min_value=0.0, max_value=100.0, value=50.0)
		Rainfall = st.number_input("Amount Rainfall", min_value=0.0, max_value=200.0, value=50.0)
		Wind_Speed = st.number_input("The Speed of Wind", min_value=0.0, max_value=20.0, value=5.0)
		Sunshine_Hours = st.number_input("The Amount of Hour Sunlight Exposure", min_value=0.0, max_value=15.0, value=8.0)
		Evaporation = st.number_input("Amount of Water Evaporation", min_value=0.0, max_value=50.0, value=25.0)

		# User data
		user_data = pd.DataFrame({
    		'Standard Week': [Standard_Week],
    		'Pest Value': [Pest_Value],
    		'MaxT': [Max_Temperature],
			'MinT': [Min_Temperature],
    		'RH1(%)': [Max_Humidity],
    		'RH2(%)': [Min_Humidity],
    		'RF(mm)': [Rainfall],
    		'WS(kmph)': [Wind_Speed],
			'SSH(hrs)': [Sunshine_Hours],
			'EVP(mm)': [Evaporation]
		})

		 # Prediction and display results for each kernel
		st.subheader("KNN Results Based on User Input:")
		for neighbors in [5, 10, 15, 20]:
			# Train KNN with selected neigbors
			knn_classifier = KNeighborsClassifier(n_neighbors=neighbors)
			knn_classifier.fit(data_input_training, data_target_training)
			
			# Prediction
			knn_prediction = knn_classifier.predict(user_data)
			
			# Display results for each kernel
			st.write(f"KNN Prediction (Number of Neigbors: {neighbors}): {knn_prediction[0]}")
	

	elif selectModel == "Random Forest":
		st.subheader("Random Forest Rice Pest and Disease Classification Model")
		st.subheader("Predicted Results for Dataset Testing:")
	
		st.write("Accuracy Score for Random Forest")

		# ACCURACY SCORE IN TABLE
		st.write("Mean Square Value in Table")
		results_df_svr = pd.DataFrame({
			'Number of Estimators': rf_values,
			'Accuracy Score': accuracy_values_rf
        })
		centered_table_svr = f'<div style="text-align: center;">{results_df_svr.to_html(index=False)}</div>'
		st.markdown(centered_table_svr, unsafe_allow_html=True)

		st.subheader("Conclusion")
		st.write('''Based on the results, the highest accuracy score between the 4 numbers of estimators is 
		   when the n is at 20 with the accuracy score of 25.75%. When the estimator is at 10 and 15 the accuracy score 
		   is the same which is at 23.00 % The lowest of them all is when the n is at 5 which has the score of 21.50%. 
		   Hence the number of estimators that will represent the model for the RF model is when the n is at 20.''')
		
		#USER INPUT FOR RANDOM FOREST MODEL
		st.header("User Input for Random Forest Model")
		st.subheader("Enter the following details:")
		Standard_Week = st.number_input("Week of Year", min_value=0.0, max_value=20.0, value=15.0)
		Pest_Value = st.number_input("Pest Value", min_value=0.0, max_value=10000.0, value=500.0)
		Max_Temperature = st.number_input("Maximum Temperature", min_value=0.0, max_value=50.0, value=30.)
		Min_Temperature = st.number_input("Minimum Temperature", min_value=0.0, max_value=50.0, value=25.0)
		Max_Humidity = st.number_input("Maximum Relative Humidity", min_value=-10.0, max_value=100.0, value=90.0)
		Min_Humidity = st.number_input("Minimum Relative Humidity", min_value=0.0, max_value=100.0, value=50.0)
		Rainfall = st.number_input("Amount Rainfall", min_value=0.0, max_value=200.0, value=50.0)
		Wind_Speed = st.number_input("The Speed of Wind", min_value=0.0, max_value=20.0, value=5.0)
		Sunshine_Hours = st.number_input("The Amount of Hour Sunlight Exposure", min_value=0.0, max_value=15.0, value=8.0)
		Evaporation = st.number_input("Amount of Water Evaporation", min_value=0.0, max_value=50.0, value=25.0)

		# User data
		user_data = pd.DataFrame({
    		'Standard Week': [Standard_Week],
    		'Pest Value': [Pest_Value],
    		'MaxT': [Max_Temperature],
			'MinT': [Min_Temperature],
    		'RH1(%)': [Max_Humidity],
    		'RH2(%)': [Min_Humidity],
    		'RF(mm)': [Rainfall],
    		'WS(kmph)': [Wind_Speed],
			'SSH(hrs)': [Sunshine_Hours],
			'EVP(mm)': [Evaporation]
		})

		 # Prediction and display results for each kernel
		st.subheader("Random Forest Results Based on User Input:")
		for estimators in [5, 10, 15, 20]:
			# Train KNN with selected neigbors
			rf_classifier = RandomForestClassifier(n_estimators=estimators)
			rf_classifier.fit(data_input_training, data_target_training)
			
			# Prediction
			rf_prediction = rf_classifier.predict(user_data)
			
			# Display results for each kernel
			st.write(f"Random Forest Prediction (Estimators: {estimators}): {rf_prediction[0]}")
		
	elif selectModel == "Compare All Models":
		
		accuracy_scores["SVM"] = max(accuracy_values_svm)
			
		accuracy_scores["KNN"] = max(accuracy_values_knn)
			
		accuracy_scores["Random Forest"] = max(accuracy_values_rf)
		
		results_df = pd.DataFrame({
			'Models': ["SVM", "KNN", "Random Forest"],
			'Accuracy Score': [max(accuracy_values_svm), max(accuracy_values_knn), max(accuracy_values_rf)]
		})

		st.header("Comparison of Accuracy Score for Each Model")
		centered_table = f'<div style="text-align: center;">{results_df.to_html(index=False)}</div>'
		st.markdown(centered_table, unsafe_allow_html=True)
			

		if "SVM" in accuracy_scores and "KNN" in accuracy_scores and "Random Forest" in accuracy_scores:
			maximum_accuracy = {
            "SVM": accuracy_scores["SVM"],
            "KNN": accuracy_scores["KNN"],
            "Random Forest": accuracy_scores["Random Forest"]
        }
			
		highest_mse_model  = max(accuracy_scores, key=accuracy_scores.get)  # Find the highest Accuracy Score
		st.write(f"Highest Accuracy Model: {highest_mse_model }, Accuracy: {maximum_accuracy[highest_mse_model ]}")

		st.subheader("Conclusion")
		st.write('''In conclusion, The table above shows the summary of the result for the Rice - Pest and Disease. 
		   The KNN model has the highest accuracy among the three, with an accuracy of 25.75%. Therefore, if accuracy is the primary criterion, 
		   the KNN model (with 25 neighbors) would be chosen as the representation for the prediction model. The reason why the accuracy score of 
		   the model is that there may have been overfitting by the lack of regularization and imbalance of data of the dataset.''')