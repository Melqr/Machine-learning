import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from PIL import Image

st.title("Wheat Variety")
st.header("What is Wheat Variety?")
st.write('''Wheat variety refers to the classification of wheat grains into distinct types, 
		 each characterized by specific attributes and features. In the context of the Wheat 
		 Variety Classification dataset, the varieties include Kama, Rosa, and Canadian. 
		 These classifications are crucial for various stakeholders in the agricultural 
		 sector, such as growers, processors, and consumers. The identification of wheat 
		 varieties plays a vital role in determining the market value of different types 
		 of wheat''')
st.write('''Basically, the understanding wheat variety involves recognizing and categorizing 
		 wheat grains based on their unique characteristics, ultimately influencing 
		 decision-making processes across the agricultural supply chain.''')

st.subheader("Models Chosen for This Machine")
st.write("The models that are chosen for this machine are Support Vector Machine (SVM), Decision Tree and Random Forest")

st.subheader("Wheat Variety Classification Using Three Different Models")
st.write('''The goal of wheat variety classification using machine learning is to develop a 
		 predictive model that can accurately and efficiently identify the variety of wheat 
		 grains based on input features or characteristics''')

selectView = st.sidebar.selectbox("Select View:", options = ["Select to View", "Variable Input & Target", "Dataset Training and Testing"])

data = pd.read_csv('wheat.csv')
X = data.drop(columns=['bil', 'category'])
y = data['category']


if selectView == "Variable Input & Target":
	
	st.write('''The dataset provides a solid foundation for classification and cluster 
		  analysis by painstakingly measuring 7 geometric parameters, including area, 
		  perimeter, compactness, length, breadth, asymmetry coefficient, and kernel groove 
		  length''')
	st.subheader("Dataset Variables Input Description")
	st.write('''Here are descriptions of the dataset features:

- **Area (A):** Area of the wheat kernel.
- **Perimeter (P):** Perimeter of the wheat kernel.
- **Compactness (C):** Compactness of the wheat kernel calculated as 4πA/P^2.
- **Length of Kernel:** Length of the wheat kernel.
- **Width of Kernel:** Width of the wheat kernel.
- **Asymmetry Coefficient:** Asymmetry coefficient of the wheat kernel.
- **Length of Kernel Groove:** Length of the groove in the wheat kernel.
''', unsafe_allow_html=True)

	data = pd.read_csv('wheat.csv')
	X = data.drop(columns=['bil', 'category'])
	X

	st.subheader("Dataset Variables Target Description")
	st.write('''<span style="font-family: 'Arial, sans-serif'; color: green;">Here are descriptions of the dataset features:
	
- **Category:** Variety category of wheat (Kama, Rosa, Canadian).
		The target variable in the Wheat Variety Classifier dataset is 'Category'. 
		This variable represents the variety category of the wheat kernel and is the 
		  label we want to predict.</span>''', unsafe_allow_html=True)

	y = data['category']
	y


# DECLARE DATA INPUT AND DATA TARGET FOR TRAINING AND TESTING DATA		
if selectView == "Dataset Training and Testing":
	st.write('''The total dataset in the have 211 rows and 8 columns, there is a new add 
		  column "bil" and random are added to the CSV, providing it as an additional feature. 
		  Furthermore, the categorical numbers for wheat types are converted into meaningful 
		  labels: 1 for Kama, 2 for Rosa, and 3 for Canadian. A 'bil' column labeled the 210 
		  rows to increase interpretability in wheat variety classification, while a 'random' 
		  column allowed for dataset shuffling using =RANDbetween(1,210). After sorting by 
		  'random', the column was eliminated. The dataset is now ready for classification 
		  tasks, providing greater clarity when depicting wheat types.''')
	st.write('''The dataset has been manually split into training and testing sets. 
		  80% of the data is used for training, and the 
		  remaining 20% is used for testing.''')
	
	st.header("Split Data Input (X) and Data Target (y) into Training and Testing Data")
	st.subheader("Data Training")
	train_data = pd.read_csv('wheat_train.csv')
	st.write("Data Input From Training Set")
	data_input_training = train_data.drop(columns=['bil', 'category'])
	data_input_training

	st.write("Data Target From Training Set")
	data_target_training = train_data['category']
	data_target_training


	st.subheader("Data Testing")
	test_data = pd.read_csv('wheat_test.csv')
	st.write("Data Input From Testing Set")
	data_input_testing = test_data.drop(columns=['bil', 'category'])
	data_input_testing

	st.write("Data Target From Testing Set")
	data_target_testing = test_data['category']
	data_target_testing
		
	
# SELECT MODEL FOR CLASSIFICATION
	selectModel = st.sidebar.selectbox("Select Model", options = ["Select Model", "Support Vector Machine (SVM)", "K-Nearest Neighbors", "Random Forest", "Compare All Models"])
	
	accuracy_scores = {}

#----------------------TRAIN SVM--------------------
	svKernelLinear = SVC(kernel='linear')	
	svKernelLinear.fit(data_input_training, data_target_training)
	LinearResult = svKernelLinear.predict(data_input_testing)
	accuracyLinear = accuracy_score(LinearResult, data_target_testing)
		
	svKernelPoly = SVC(kernel='poly')	
	svKernelPoly.fit(data_input_training, data_target_training)
	PolyResult = svKernelPoly.predict(data_input_testing)
	accuracyPoly = accuracy_score(PolyResult, data_target_testing)
		
	svKernelRBF = SVC(kernel='rbf')	
	svKernelRBF.fit(data_input_training, data_target_training)
	rbfResult = svKernelRBF.predict(data_input_testing)
	accuracyRBF = accuracy_score(rbfResult, data_target_testing)

	svKernelSigmoid = SVC(kernel='sigmoid')	
	svKernelSigmoid.fit(data_input_training, data_target_training)
	SigmoidResult = svKernelSigmoid.predict(data_input_testing)
	accuracySigmoid = accuracy_score(SigmoidResult, data_target_testing)

#----------------------TRAIN KNN--------------------
	knn_model_5 = KNeighborsClassifier(n_neighbors=5)
	knn_model_5.fit(data_input_training, data_target_training)	
	knn_result_5 = knn_model_5.predict(data_input_testing)
	knn_accuracy_5 = accuracy_score(knn_result_5, data_target_testing)
		
	knn_model_10 = KNeighborsClassifier(n_neighbors=10)
	knn_model_10.fit(data_input_training, data_target_training)	
	knn_result_10 = knn_model_10.predict(data_input_testing)
	knn_accuracy_10 = accuracy_score(knn_result_10, data_target_testing)
		
	knn_model_15 = KNeighborsClassifier(n_neighbors=15)
	knn_model_15.fit(data_input_training, data_target_training)	
	knn_result_15 = knn_model_15.predict(data_input_testing)
	knn_accuracy_15 = accuracy_score(knn_result_15, data_target_testing)
		
	knn_model_20 = KNeighborsClassifier(n_neighbors=20)
	knn_model_20.fit(data_input_training, data_target_training)	
	knn_result_20 = knn_model_20.predict(data_input_testing)
	knn_accuracy_20 = accuracy_score(knn_result_20, data_target_testing)

#----------------------TRAIN RANDOM FOREST--------------------
	random_forest_5 = RandomForestClassifier(n_estimators=5, random_state=42)
	random_forest_5.fit(data_input_training, data_target_training)	
	random_forest_result_5 = random_forest_5.predict(data_input_testing)
	rf_accuracy_5 = accuracy_score(random_forest_result_5, data_target_testing)
		
	random_forest_10 = RandomForestClassifier(n_estimators=10, random_state=42)
	random_forest_10.fit(data_input_training, data_target_training)	
	random_forest_result_10 = random_forest_10.predict(data_input_testing)
	rf_accuracy_10 = accuracy_score(random_forest_result_10, data_target_testing)
		
	random_forest_15 = RandomForestClassifier(n_estimators=15, random_state=42)
	random_forest_15.fit(data_input_training, data_target_training)	
	random_forest_result_15 = random_forest_15.predict(data_input_testing)
	rf_accuracy_15 = accuracy_score(random_forest_result_15, data_target_testing)
		
	random_forest_20 = RandomForestClassifier(n_estimators=20, random_state=42)
	random_forest_20.fit(data_input_training, data_target_training)	
	random_forest_result_20 = random_forest_20.predict(data_input_testing)
	rf_accuracy_20 = accuracy_score(random_forest_result_20, data_target_testing)

	# ---------------------------------TRAIN SVM------------------------------------------------------
	if selectModel == "Support Vector Machine (SVM)":
		st.subheader("Support Vector Machine Wheat Variety Classification Model")
		st.subheader("Predicted Results for Dataset Testing:")
		
        # ACCURACY SCORE IN TABLE
		results_df = pd.DataFrame({
			'Kernel': ["Linear", "Poly", "RBF", "Sigmoid"],
			'Accuracy Score': [accuracyLinear, accuracyPoly, accuracyRBF, accuracySigmoid]
		})

		st.write("Accuracy Score in Table")
		centered_table = f'<div style="text-align: center;">{results_df.to_html(index=False)}</div>'
		st.markdown(centered_table, unsafe_allow_html=True)

		st.subheader("Conclusion")
		st.write('''The accuracy of the SVM model changes depending on the kernel function used. 
		   The Linear kernel has the highest accuracy (93.18%), demonstrating effective linear separability. 
		   Both Polynomial (Poly) and Radial Basis Function (RBF) kernels perform consistently at 88.63%. 
		   However, the Sigmoid kernel lags significantly, with an accuracy of 22.73%, indicating that it 
		   may not be appropriate for the dataset's underlying structure. In conclusion, the choice of kernel, 
		   notably Linear, Poly, or RBF, is critical in achieving competitive accuracy in this SVM model, 
		   but the Sigmoid kernel looks to be less suitable for the dataset.''')
		
		#USER INPUT FOR SVM MODEL
		st.header("User Input for SVM Model")
		st.subheader("Enter the following details:")
		area = st.number_input("Area", min_value=0.0, max_value=20.0, value=15.0)
		perimeter = st.number_input("Perimeter", min_value=0.0, max_value=30.0, value=15.0)
		compactness = st.number_input("Compactness", min_value=0.0, max_value=1.0, value=0.8)
		length = st.number_input("Length of Kernel", min_value=-10.0, max_value=10.0, value=5.0)
		width = st.number_input("Width of Kernel", min_value=0.0, max_value=10.0, value=5.0)
		asymmetry_coefficient = st.number_input("Asymmetry Coefficient", min_value=0.0, max_value=10.0, value=5.0)
		groovelength = st.number_input("Length of Kernel Grove", min_value=0.0, max_value=10.0, value=5.0)

		# User data
		user_data = pd.DataFrame({
    		'area': [area],
    		'perimeter': [perimeter],
    		'compactness': [compactness],
    		'length': [length],
    		'width': [width],
    		'asymmetry_coefficient': [asymmetry_coefficient],
    		'groove_length': [groovelength]
		})

		 # Prediction and display results for each kernel
		st.subheader("SVM Results Based on User Input:")
		for kernel in ["linear", "poly", "rbf", "sigmoid"]:
			# Train KNN with selected kernel type
			rf_classifier = SVC(kernel=kernel)
			rf_classifier.fit(data_input_training, data_target_training)
			
			# Prediction
			rf_prediction = rf_classifier.predict(user_data)
			
			# Display results for each kernel
			st.write(f"SVM Prediction (Kernel: {kernel}): {rf_prediction[0]}")
		

	# ---------------------------------TRAIN KNN------------------------------------------------------
	elif selectModel == "K-Nearest Neighbors":
		st.subheader("K-Nearest Neighbors Wheat Variety Classification Model")
		st.subheader("Predicted Results for Dataset Testing:")

        # ACCURACY SCORE IN TABLE
		results_df = pd.DataFrame({
			'Number of Neighbors': [5, 10, 15, 20],
			'Accuracy Score': [knn_accuracy_5, knn_accuracy_10, knn_accuracy_15, knn_accuracy_20]
		})

		st.write("Accuracy Score in Table")
		centered_table = f'<div style="text-align: center;">{results_df.to_html(index=False)}</div>'
		st.markdown(centered_table, unsafe_allow_html=True)

		st.subheader("Conclusion")
		st.write('''The K-Nearest Neighbours (KNN) models, which use varying values of n_neighbors (10, 15, 20), 
		   consistently perform well on the dataset. Based on classification reports with an average accuracy of 84%, 
		   the precision, recall, and F1-scores for each class (Canadian, Kama, Rosa) are consistently high, showing 
		   accurate predictions and a balanced trade-off between precision and memory. The confusion matrices further 
		   show the models' ability to make accurate predictions with minimum misclassifications. In particular, this 
		   robust performance is maintained across n_neighbors values. In summary, the KNN models demonstrate reliable 
		   and effective classification skills for the given dataset, meeting or exceeding expectations across a variety 
		   of evaluation metric.''')
		
		#USER INPUT FOR KNN MODEL
		st.header("User Input for KNN Model")
		st.subheader("Enter the following details:")
		area = st.number_input("Area", min_value=0.0, max_value=20.0, value=15.0)
		perimeter = st.number_input("Perimeter", min_value=0.0, max_value=30.0, value=15.0)
		compactness = st.number_input("Compactness", min_value=0.0, max_value=1.0, value=0.8)
		length = st.number_input("Length of Kernel", min_value=-10.0, max_value=10.0, value=5.0)
		width = st.number_input("Width of Kernel", min_value=0.0, max_value=10.0, value=5.0)
		asymmetry_coefficient = st.number_input("Asymmetry Coefficient", min_value=0.0, max_value=10.0, value=5.0)
		groovelength = st.number_input("Length of Kernel Grove", min_value=0.0, max_value=10.0, value=5.0)

		# User data
		user_data = pd.DataFrame({
    		'area': [area],
    		'perimeter': [perimeter],
    		'compactness': [compactness],
    		'length': [length],
    		'width': [width],
    		'asymmetry_coefficient': [asymmetry_coefficient],
    		'groove_length': [groovelength]
		})

		 # Prediction and display results for each neighbors
		st.subheader("KNN Results Based on User Input:")
		for neighbors in [5, 10, 15, 20]:
			# Train KNN with selected kernel type
			rf_classifier = KNeighborsClassifier(n_neighbors=neighbors)
			rf_classifier.fit(data_input_training, data_target_training)
			
			# Prediction
			rf_prediction = rf_classifier.predict(user_data)
			
			# Display results for each kernel
			st.write(f"KNN Prediction (Number of Neighbors: {neighbors}): {rf_prediction[0]}")
	
	# ---------------------------------TRAIN RANDOM FOREST------------------------------------------------------
	elif selectModel == "Random Forest":
		st.subheader("Random Forest Wheat Variety Classification Model")
		st.subheader("Predicted Results for Dataset Testing:")
		
        # ACCURACY SCORE IN TABLE
		results_df = pd.DataFrame({
			'Number of Estimators': [5, 10, 15, 20],
			'Accuracy Score': [rf_accuracy_5, rf_accuracy_10, rf_accuracy_15, rf_accuracy_20]
		})

		st.write("Accuracy Score in Table")
		centered_table = f'<div style="text-align: center;">{results_df.to_html(index=False)}</div>'
		st.markdown(centered_table, unsafe_allow_html=True)

		st.subheader("Conclusion")
		st.write('''Based on the results, the highest accuracy recorded at 5 and 20 estimators (90.90%). 
		   Notably, the model reaches its peak accuracy at 5 estimators (90.90%), but then dips slightly 
		   at 10 estimators (86.36%) and slightly increases at estimator 15 with percentage (88.63%). 
		   Next, the accuracy rose up, till reaching the peak accuracy again which is 90.90%, suggesting 
		   that the best configuration for this dataset follows a clear pattern that shows that it decreases 
		   slightly and slowly rises to peak accuracy. It emphasizes the importance of conducting a thorough 
		   search of hyperparameter space to guarantee that the Random Forest model performs optimally on the 
		   specific properties of the data.''')
		
		#USER INPUT FOR Random Forest MODEL
		st.header("User Input for Random Forest Model")
		st.subheader("Enter the following details:")
		area = st.number_input("Area", min_value=0.0, max_value=20.0, value=15.0)
		perimeter = st.number_input("Perimeter", min_value=0.0, max_value=30.0, value=15.0)
		compactness = st.number_input("Compactness", min_value=0.0, max_value=1.0, value=0.8)
		length = st.number_input("Length of Kernel", min_value=-10.0, max_value=10.0, value=5.0)
		width = st.number_input("Width of Kernel", min_value=0.0, max_value=10.0, value=5.0)
		asymmetry_coefficient = st.number_input("Asymmetry Coefficient", min_value=0.0, max_value=10.0, value=5.0)
		groovelength = st.number_input("Length of Kernel Grove", min_value=0.0, max_value=10.0, value=5.0)

		# User data
		user_data = pd.DataFrame({
    		'area': [area],
    		'perimeter': [perimeter],
    		'compactness': [compactness],
    		'length': [length],
    		'width': [width],
    		'asymmetry_coefficient': [asymmetry_coefficient],
    		'groove_length': [groovelength]
		})

		 # Prediction and display results for each estimators
		st.subheader("Random Forest Results Based on User Input:")
		for estimators in [5, 10, 15, 20]:
			# Train KNN with selected kernel type
			rf_classifier = RandomForestClassifier(n_estimators=estimators)
			rf_classifier.fit(data_input_training, data_target_training)
			
			# Prediction
			rf_prediction = rf_classifier.predict(user_data)
			
			# Display results for each kernel
			st.write(f"Random Forest Prediction (Number of Estimators: {estimators}): {rf_prediction[0]}")
	
		
	elif selectModel == "Compare All Models":
		# Find the highest accuracy for SVM
		highest_accuracy_svm = max(accuracyLinear, accuracyPoly, accuracyRBF, accuracySigmoid)
		kernel_svm = 'linear' if highest_accuracy_svm == accuracyLinear else \
              'poly' if highest_accuracy_svm == accuracyPoly else \
              'rbf' if highest_accuracy_svm == accuracyRBF else 'sigmoid'		
		accuracy_scores["Highest_SVM"] = highest_accuracy_svm

		# Find the highest accuracy for KNN
		highest_accuracy_knn = max(knn_accuracy_5, knn_accuracy_10, knn_accuracy_15, knn_accuracy_20)
		bestKNN = 5 if highest_accuracy_knn == knn_accuracy_5 else \
              10 if highest_accuracy_knn == knn_accuracy_10 else \
              15 if highest_accuracy_knn == knn_accuracy_15 else 20		
		accuracy_scores["Highest_KNN"] = highest_accuracy_knn

		# Find the highest accuracy for Random Forest
		highest_accuracy_rf = max(rf_accuracy_5, rf_accuracy_10, rf_accuracy_15, rf_accuracy_20)
		bestRF = 5 if highest_accuracy_rf == rf_accuracy_5 else \
              10 if highest_accuracy_rf == rf_accuracy_10 else \
              15 if highest_accuracy_rf == rf_accuracy_15 else 20
		accuracy_scores["Highest_RF"] = highest_accuracy_rf

		# ACCURACY SCORE IN TABLE
		results_df = pd.DataFrame({
			'Models': ["SVM", "KNN", "Random Forest"],
			'Accuracy Score': [highest_accuracy_svm, highest_accuracy_knn, highest_accuracy_rf]
		})

		st.header("Comparison of Accuracy Score for Each Model")
		centered_table = f'<div style="text-align: center;">{results_df.to_html(index=False)}</div>'
		st.markdown(centered_table, unsafe_allow_html=True)


		if "Highest_SVM" in accuracy_scores and "Highest_KNN" in accuracy_scores and "Highest_RF" in accuracy_scores:
			highest_accuracies = {
				"SVM": accuracy_scores["Highest_SVM"],
				"KNN": accuracy_scores["Highest_KNN"],
				"Random Forest": accuracy_scores["Highest_RF"]
				}
		
			highest_accuracy_model = max(highest_accuracies, key=highest_accuracies.get)
			st.write(f"Highest Accuracy Model: {highest_accuracy_model}, Accuracy: {highest_accuracies[highest_accuracy_model]}")
		
		st.subheader("Conclusion")
		st.write('''In summary, the K-Nearest Neighbours (KNN) model continuously maintains an accuracy of 84.09% throughout 
		a range of k values, including 10, 15, 20, and the succeeding value. This stability emphasizes the model's robust performance. 
		However, KNN lags significantly behind SVM and Random Forest (RF) models, which have higher accuracy at 93.18%. 
		Trade-offs must be made while deciding between various models, taking into account criteria like consistency, 
		interpretability, computing efficiency, and data nature.''')

		