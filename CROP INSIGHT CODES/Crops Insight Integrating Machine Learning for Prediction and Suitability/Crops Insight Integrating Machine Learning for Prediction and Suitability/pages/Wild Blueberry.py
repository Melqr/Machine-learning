import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
 
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

from PIL import Image

st.title("Wild Blueberry")
st.header("What is Wild blueberry Yield Prediction?")
st.write('''he Wild Blueberry Yield Prediction dataset addresses the difficulty of predicting 
		 crop yields accurately, especially for wild blueberries found in Maine, USA. 
		 This crop, a native forest understory plant, is primarily reliant on bees for 
		 cross-pollination, yet its production is regulated by weather, soil fertility, 
		 and pests.''')
st.write('''Precision crop production estimates are critical in precision agriculture, 
		 but acquiring high-quality training data is difficult. To address this, the research 
		 team used computer simulation modeling to produce synthetic data based on verified
		  wild blueberry pollination models. These simulations supplied a large amount of 
		 training data, which is required by machine learning algorithms.''')

st.subheader("Models Chosen for This Machine")
st.write("The models that are chosen for this machine are Support Vector Machine (SVM), K-Nearest Neighbors (KNN) and Random Forest")

st.header("Wild blueberry Yield Prediction Using Three Different Models")
st.write('''Its goal is to help farmers optimize agricultural practices for wild blueberry 
		 growing based on climate variables, therefore predicting crop yields and improving 
		 agricultural decision-making precision.
         ''')

selectView = st.sidebar.selectbox("Select View:", options = ["Select to View", "Variable Input & Target", "Dataset Training and Testing"])


if selectView == "Variable Input & Target":
	st.write('''The dataset has 17 columns that address meteorological and soil variables that suitable 
		   for variable input. Below are all columns in the data:''')

	st.subheader("Dataset Variables Input Description")
	st.write('''Here are descriptions of the dataset features:

- **row:** The row identifier.
- **clonesize:** The average blueberry clone size in the field.
- **honeybee:** Honeybee density in the field.
- **bumbles:** Bumblebee density in the field.
- **andrena:** Andrena bee density in the field.
- **osmia:** Osmia bee density in the field.
- **MaxOfUpperTRange:** ℃ The highest record of the upper band daily air temperature during the bloom season.
- **MinOfUpperTRange:** ℃ The lowest record of the upper band daily air temperature.
- **AverageOfUpperTRange:** ℃ The average of the upper band daily air temperature.
- **MaxOfLowerTRange:** ℃ The highest record of the lower band daily air temperature.
- **MinOfLowerTRange:** ℃ The lowest record of the lower band daily air temperature.
- **AverageOfLowerTRange:** ℃ The average of the lower band daily air temperature.
- **RainingDays:** The total number of days during the bloom season, each of which has precipitation larger than zero.
- **AverageRainingDays:** The average of raining days of the entire bloom season.
- **fruitset:** Fruit set. 
- **fruitmass:** Fruit mass.
- **seeds:** Number of seeds.
''')

	df = pd.read_csv('WildBlueberryPollinationSimulationData.csv')
	X = df.drop("yield", axis = 1)
	X

	st.subheader("Dataset Variables Target Description")
	st.write('''<span style="font-family: 'Arial, sans-serif'; color: green;">Here are descriptions of the dataset features:
	
- **yield:** Wild blueberry yield.
		Wild blueberry yield is measured in ton unit. 
  This is the variable that the machine learning models aim to predict 
		  based on the given features.</span>''', unsafe_allow_html=True)

	Y = df["yield"]
	Y


# DECLARE DATA INPUT AND DATA TARGET FOR TRAINING AND TESTING DATA		
if selectView == "Dataset Training and Testing":
	st.subheader("Split Data Input (X) and Data Target (y) into Training and Testing Data")
	st.write('''In this dataset, the total number of rows is 777, which is relatively low. '
		  To address this, a common practice is to employ a train-test split technique, 
		  where 20% of the data is allocated for testing purposes, while the remaining 80% is 
		  designated for training the model. This approach helps ensure that the model is 
		  exposed to a sufficient amount of data during training, allowing it to generalize 
		  well. The use of an 80-20 split is aimed at minimizing the mean squared error (MSE), 
		  a metric used to evaluate the performance of the model. By dedicating a substantial 
		  portion of the dataset to training, the model has a better chance of learning 
		  patterns and relationships within the data, which should contribute to achieving a 
		  lower MSE.''')
	st.write('''The dataset has been split into training and testing sets for machine learning model evaluation. 
This was done using the `train_test_split` function from the scikit-learn library.

- **Training Data:** The training set contains a portion of the dataset used to train the machine learning models.
  It consists of a random subset of examples that the model learns from.

- **Testing Data:** The testing set is a separate portion of the dataset reserved for evaluating the model's performance.
  It helps assess how well the model generalizes to new, unseen data.

This split is crucial for ensuring that the model can make accurate predictions on new data it hasn't seen during training.
''')
	
	df = pd.read_csv('WildBlueberryPollinationSimulationData.csv')
	df.drop("row", axis=1, inplace=True)
	
    # Quantile Transformation
	qt = QuantileTransformer(output_distribution='normal')
	for col in df.columns:
		df[col] = qt.fit_transform(pd.DataFrame(df[col]))
		
    
    #Outlier Handling
	for col in df:
		q1 = df[col].quantile(0.25)
		q3 = df[col].quantile(0.75)
		iqr = q3 - q1
		whisker_width = 1.5
		lower_whisker = q1 - (whisker_width * iqr)
		upper_whisker = q3 + whisker_width * iqr
		df[col] = np.where(df[col] > upper_whisker, upper_whisker, np.where(df[col] < lower_whisker, lower_whisker, df[col]))
		
    # Extract features and target
	X = df.drop("yield", axis = 1)
	
	Y = df["yield"]

# Split the data into training and testing sets
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
	
	st.subheader("Training Data:")
	X_train
	
	st.subheader("Testing Data:")
	X_test
	
		
# SELECT MODEL FOR CLASSIFICATION
	selectModel = st.sidebar.selectbox("Select Model", options = ["Select Model", "Support Vector Machine (SVM)", "K-Nearest Neighbors", "Random Forest", "Compare All Models"])

	mse_accuracies  = {}

	# ---------------------------------TRAIN SVM------------------------------------------------------
	kernel_values_svm = []
	mse_values_svm = []
		
	kernels = ['linear', 'poly', 'rbf', 'sigmoid']
	for kernel in kernels:
		regSVR = SVR(kernel=kernel)
		regSVR.fit(X_train, Y_train)
		yPredRegSVR = regSVR.predict(X_test)
		MSEsvr = mean_squared_error(yPredRegSVR, Y_test)
		kernel_values_svm.append(kernel)
		mse_values_svm.append(MSEsvr)
				
	# ---------------------------------TRAIN KNN------------------------------------------------------
	knn_values = []
	mse_values_knn = []

	n_neighbors_list = [5, 10, 15, 20]
	for n_neighbors in n_neighbors_list:
		knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
		knn_regressor.fit(X_train, Y_train)
		yPredKNN = knn_regressor.predict(X_test)
		MSEKnn = mean_squared_error(yPredKNN, Y_test)
		knn_values.append(n_neighbors)
		mse_values_knn.append(MSEKnn)

	# ---------------------------------TRAIN RANDOM FOREST------------------------------------------------------
	rf_values = []
	mse_values_rf = []
		
	n_estimators_list = [5, 10, 15, 20]
	for n_estimators in n_estimators_list:
		regRFR = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
		regRFR.fit(X_train, Y_train)
		yPredRegRFR = regRFR.predict(X_test)
		MSErf = mean_squared_error(yPredRegRFR, Y_test)
		rf_values.append(n_estimators)
		mse_values_rf.append(MSErf)


	# ---------------------------------TRAIN SVM------------------------------------------------------
	if selectModel == "Support Vector Machine (SVM)":
		st.subheader("Support Vector Machine Wild Blueberry Prediction Model")
		st.subheader("Predicted Results for Dataset Testing:")
	
        # ACCURACY SCORE IN TABLE
		st.write("Mean Square Value in Table")
		results_df_svr = pd.DataFrame({
			'Kernel': kernel_values_svm,
			'Mean Squared Error': mse_values_svm
        })
		centered_table_svr = f'<div style="text-align: center;">{results_df_svr.to_html(index=False)}</div>'
		st.markdown(centered_table_svr, unsafe_allow_html=True)
		
		st.subheader("Conclusion")
		st.write('''Based on the results, the Linear and RBF kernels showed low MSE values (0.0116 and 0.0283), 
		   indicating effective performance. However, the Polynomial kernel exhibited a higher MSE (0.1132), 
		   while the Sigmoid kernel had a significantly elevated MSE (99.29), indicating poor performance. 
		   In conclusion, the Linear and RBF kernels outperformed the Poly and Sigmoid kernels, making them 
		   more suitable for wild blueberry yield prediction in this SVM model.''')
		
		#USER INPUT FOR SVM MODEL
		st.header("User Input for SVM Model")
		st.subheader("Enter the following details:")
		clone_size = st.number_input("Clone Size", min_value=0.0, max_value=100.0, value=12.0)
		honey_bee = st.number_input("Honey Bee Density", min_value=0.0, max_value=1.0, value=0.5)
		Bumbles = st.number_input("Bumbles Density", min_value=0.0, max_value=1.0, value=0.4)
		Andrena = st.number_input("Andrena Density", min_value=-10.0, max_value=1.0, value=0.5)
		Osmia = st.number_input("Osmia Density", min_value=0.0, max_value=1.0, value=0.5)
		MaxOf_UpperTRange = st.number_input("Highest Upper Temperature", min_value=0.0, max_value=100.0, value=50.0)
		MinOf_UpperTRange = st.number_input("Lowest Upper Temperature", min_value=0.0, max_value=100.0, value=30.0)
		AverageOf_UpperTRange = st.number_input("Average Upper Temperature", min_value=0.0, max_value=100.0, value=50.0)
		MaxOf_LowerTRange = st.number_input("Highest Lower Temperature", min_value=-10.0, max_value=100.0, value=50.0)
		MinOf_LowerTRange = st.number_input("Lowest Lower Temperature", min_value=0.0, max_value=100.0, value=30.0)
		AverageOf_LowerTRange = st.number_input("Average Lower Temperature", min_value=0.0, max_value=100.0, value=50.0)
		Raining_Days = st.number_input("Number of Raining Days", min_value=0.0, max_value=100.0, value=25.0)
		Average_RainingDays = st.number_input("Average of Raining Days", min_value=0.0, max_value=1.0, value=0.5)
		Fruit_set = st.number_input("Measure of Fruit Set", min_value=0.0, max_value=1.0, value=0.5)
		Fruit_mass = st.number_input("Measure of Fruit Mass", min_value=0.0, max_value=1.0, value=0.5)
		Seed = st.number_input("Numbe of Seeds", min_value=0.0, max_value=100.0, value=40.0)

		# User data
		user_data = pd.DataFrame({
    		'clonesize': [clone_size],
    		'honeybee': [honey_bee],
    		'bumbles': [Bumbles],
    		'andrena': [Andrena],
    		'osmia': [Osmia],
    		'MaxOfUpperTRange': [MaxOf_UpperTRange],
    		'MinOfUpperTRange': [MinOf_UpperTRange],
			'AverageOfUpperTRange': [AverageOf_UpperTRange],
			'MaxOfLowerTRange': [MaxOf_LowerTRange],
			'MinOfLowerTRange': [MinOf_LowerTRange],
			'AverageOfLowerTRange': [AverageOf_LowerTRange],
			'RainingDays': [Raining_Days],
			'AverageRainingDays': [Average_RainingDays],
			'fruitset': [Fruit_set],
			'fruitmass': [Fruit_mass],
			'seeds': [Seed]
		})

		 # Prediction and display results for each kernel
		st.subheader("SVM Results Based on User Input:")
		for kernel in ["linear", "poly", "rbf", "sigmoid"]:
			# Train KNN with selected kernel type
			regSVR = SVR(kernel=kernel)
			regSVR.fit(X_train, Y_train)
			
			# Prediction
			svm_prediction = regSVR.predict(user_data)
			
			# Display results for each kernel
			st.write(f"SVM Prediction (Kernel: {kernel}): {svm_prediction[0]}")

	# ---------------------------------TRAIN KNN------------------------------------------------------
	elif selectModel == "K-Nearest Neighbors":
		st.subheader("K-Nearest Neighbors Wild Blueberry Prediction Model")
		st.subheader("Predicted Results for Dataset Testing:")
	
			
        # ACCURACY SCORE IN TABLE
		st.write("Mean Square Value in Table")
		results_df_svr = pd.DataFrame({
			'Number of Neighbors': knn_values,
			'Mean Squared Error': mse_values_knn
        })
		centered_table_svr = f'<div style="text-align: center;">{results_df_svr.to_html(index=False)}</div>'
		st.markdown(centered_table_svr, unsafe_allow_html=True)

		st.subheader("Conclusion")
		st.write('''Based on the results, a lower number of neighbors, specifically n=5, yields 
		   the best predictive accuracy with the lowest MSE of 0.0721. As the number of neighbors 
		   increases (n=10, n=15, and n=20), the MSE values increase, highlighting the importance 
		   of careful parameter selection for optimal model performance. In practical terms, a KNN 
		   configuration with fewer neighbors is recommended for effective wild blueberry yield prediction.''')
		
		#USER INPUT FOR KNN MODEL
		st.header("User Input for SVM Model")
		st.subheader("Enter the following details:")
		clone_size = st.number_input("Clone Size", min_value=0.0, max_value=100.0, value=12.0)
		honey_bee = st.number_input("Honey Bee Density", min_value=0.0, max_value=1.0, value=0.5)
		Bumbles = st.number_input("Bumbles Density", min_value=0.0, max_value=1.0, value=0.4)
		Andrena = st.number_input("Andrena Density", min_value=-10.0, max_value=1.0, value=0.5)
		Osmia = st.number_input("Osmia Density", min_value=0.0, max_value=1.0, value=0.5)
		MaxOf_UpperTRange = st.number_input("Highest Upper Temperature", min_value=0.0, max_value=100.0, value=50.0)
		MinOf_UpperTRange = st.number_input("Lowest Upper Temperature", min_value=0.0, max_value=100.0, value=30.0)
		AverageOf_UpperTRange = st.number_input("Average Upper Temperature", min_value=0.0, max_value=100.0, value=50.0)
		MaxOf_LowerTRange = st.number_input("Highest Lower Temperature", min_value=-10.0, max_value=100.0, value=50.0)
		MinOf_LowerTRange = st.number_input("Lowest Lower Temperature", min_value=0.0, max_value=100.0, value=30.0)
		AverageOf_LowerTRange = st.number_input("Average Lower Temperature", min_value=0.0, max_value=100.0, value=50.0)
		Raining_Days = st.number_input("Number of Raining Days", min_value=0.0, max_value=100.0, value=25.0)
		Average_RainingDays = st.number_input("Average of Raining Days", min_value=0.0, max_value=1.0, value=0.5)
		Fruit_set = st.number_input("Measure of Fruit Set", min_value=0.0, max_value=1.0, value=0.5)
		Fruit_mass = st.number_input("Measure of Fruit Mass", min_value=0.0, max_value=1.0, value=0.5)
		Seed = st.number_input("Numbe of Seeds", min_value=0.0, max_value=100.0, value=40.0)

		# User data
		user_data = pd.DataFrame({
    		'clonesize': [clone_size],
    		'honeybee': [honey_bee],
    		'bumbles': [Bumbles],
    		'andrena': [Andrena],
    		'osmia': [Osmia],
    		'MaxOfUpperTRange': [MaxOf_UpperTRange],
    		'MinOfUpperTRange': [MinOf_UpperTRange],
			'AverageOfUpperTRange': [AverageOf_UpperTRange],
			'MaxOfLowerTRange': [MaxOf_LowerTRange],
			'MinOfLowerTRange': [MinOf_LowerTRange],
			'AverageOfLowerTRange': [AverageOf_LowerTRange],
			'RainingDays': [Raining_Days],
			'AverageRainingDays': [Average_RainingDays],
			'fruitset': [Fruit_set],
			'fruitmass': [Fruit_mass],
			'seeds': [Seed]
		})

		 # Prediction and display results for each kernel
		st.subheader("KNN Results Based on User Input:")
		for neighbors in [5, 10, 15, 20]:
			# Train KNN with selected neighbors
			regKNN = KNeighborsRegressor(n_neighbors=neighbors)
			regKNN.fit(X_train, Y_train)
			
			# Prediction
			knn_prediction = regKNN.predict(user_data)
			
			# Display results for each kernel
			st.write(f"KNN Prediction (Number of Neighbors: {neighbors}): {knn_prediction[0]}")
	
	# ---------------------------------TRAIN RANDOM FOREST------------------------------------------------------
	elif selectModel == "Random Forest":
		st.subheader("Random Forest Wild Blueberry Prediction Model")
		st.subheader("Predicted Results for Dataset Testing:")

        # ACCURACY SCORE IN TABLE  
		st.write("Mean Square Value in Table")
		results_df_svr = pd.DataFrame({
			'Number of Estimators': rf_values,
			'Mean Squared Error': mse_values_rf
        })
		centered_table_svr = f'<div style="text-align: center;">{results_df_svr.to_html(index=False)}</div>'
		st.markdown(centered_table_svr, unsafe_allow_html=True)
		
		st.subheader("Conclusion")
		st.write('''Based on the results, Random Forest model with with 10 estimators exhibited the lowest 
		   MSE (0.015952), indicating superior predictive accuracy compared to other configurations. 
		   This suggests that, within this context, a moderate number of estimators (10) strikes a balance 
		   between accuracy and generalization. Fine-tuning model parameters is crucial for optimal performance 
		   in wild blueberry yield prediction using the Random Forest algorithm.''')
		
		#USER INPUT FOR RANDOM FOREST MODEL
		st.header("User Input for SVM Model")
		st.subheader("Enter the following details:")
		clone_size = st.number_input("Clone Size", min_value=0.0, max_value=100.0, value=12.0)
		honey_bee = st.number_input("Honey Bee Density", min_value=0.0, max_value=1.0, value=0.5)
		Bumbles = st.number_input("Bumbles Density", min_value=0.0, max_value=1.0, value=0.4)
		Andrena = st.number_input("Andrena Density", min_value=-10.0, max_value=1.0, value=0.5)
		Osmia = st.number_input("Osmia Density", min_value=0.0, max_value=1.0, value=0.5)
		MaxOf_UpperTRange = st.number_input("Highest Upper Temperature", min_value=0.0, max_value=100.0, value=50.0)
		MinOf_UpperTRange = st.number_input("Lowest Upper Temperature", min_value=0.0, max_value=100.0, value=30.0)
		AverageOf_UpperTRange = st.number_input("Average Upper Temperature", min_value=0.0, max_value=100.0, value=50.0)
		MaxOf_LowerTRange = st.number_input("Highest Lower Temperature", min_value=-10.0, max_value=100.0, value=50.0)
		MinOf_LowerTRange = st.number_input("Lowest Lower Temperature", min_value=0.0, max_value=100.0, value=30.0)
		AverageOf_LowerTRange = st.number_input("Average Lower Temperature", min_value=0.0, max_value=100.0, value=50.0)
		Raining_Days = st.number_input("Number of Raining Days", min_value=0.0, max_value=100.0, value=25.0)
		Average_RainingDays = st.number_input("Average of Raining Days", min_value=0.0, max_value=1.0, value=0.5)
		Fruit_set = st.number_input("Measure of Fruit Set", min_value=0.0, max_value=1.0, value=0.5)
		Fruit_mass = st.number_input("Measure of Fruit Mass", min_value=0.0, max_value=1.0, value=0.5)
		Seed = st.number_input("Numbe of Seeds", min_value=0.0, max_value=100.0, value=40.0)

		# User data
		user_data = pd.DataFrame({
    		'clonesize': [clone_size],
    		'honeybee': [honey_bee],
    		'bumbles': [Bumbles],
    		'andrena': [Andrena],
    		'osmia': [Osmia],
    		'MaxOfUpperTRange': [MaxOf_UpperTRange],
    		'MinOfUpperTRange': [MinOf_UpperTRange],
			'AverageOfUpperTRange': [AverageOf_UpperTRange],
			'MaxOfLowerTRange': [MaxOf_LowerTRange],
			'MinOfLowerTRange': [MinOf_LowerTRange],
			'AverageOfLowerTRange': [AverageOf_LowerTRange],
			'RainingDays': [Raining_Days],
			'AverageRainingDays': [Average_RainingDays],
			'fruitset': [Fruit_set],
			'fruitmass': [Fruit_mass],
			'seeds': [Seed]
		})

		 # Prediction and display results for each estimators
		st.subheader("Random Forest Results Based on User Input:")
		for estimators in [5, 10, 15, 20]:
			# Train KNN with selected estimators
			regRF = RandomForestRegressor(n_estimators=estimators)
			regRF.fit(X_train, Y_train)
			
			# Prediction
			rf_prediction = regRF.predict(user_data)
			
			# Display results for each kernel
			st.write(f"Random Forest Prediction (Number of Estimators: {estimators}): {rf_prediction[0]}")
		
	elif selectModel == "Compare All Models":
			
		mse_accuracies["SVM"] = min(mse_values_svm)
			
		mse_accuracies["KNN"] = min(mse_values_knn)
			
		mse_accuracies["Random Forest"] = min(mse_values_rf)
		
		results_df = pd.DataFrame({
			'Models': ["SVM", "KNN", "Random Forest"],
			'MSE Value': [min(mse_values_svm), min(mse_values_knn), min(mse_values_rf)]
		})

		st.header("Comparison of MSE Value for Each Model")
		centered_table = f'<div style="text-align: center;">{results_df.to_html(index=False)}</div>'
		st.markdown(centered_table, unsafe_allow_html=True)
			

		if "SVM" in mse_accuracies and "KNN" in mse_accuracies and "Random Forest" in mse_accuracies:
			minimum_mse = {
				"SVM": mse_accuracies["SVM"],
				"KNN": mse_accuracies["KNN"],
				"Random Forest": mse_accuracies["Random Forest"]
				}
			lowest_mse_model  = min(mse_accuracies, key=mse_accuracies.get)  # Find the highest MSE (use min since it's an error)
			st.write(f"Lowest MSE Model: {lowest_mse_model }, MSE: {minimum_mse[lowest_mse_model ]}")

		st.subheader("Conclusion")
		st.write('''In conclusion, the performance of various models for predicting wild blueberry yield was evaluated based 
		   on Mean Squared Error (MSE) values. The SVM model with a linear kernel achieved the lowest MSE of 0.0116, indicating 
		   its superior accuracy in capturing the underlying patterns in the dataset. Following closely, the Random Forest (RF) 
		   model with 15 estimators demonstrated a competitive MSE value of 0.015970, emphasizing its effectiveness in 
		   predicting wild blueberry yield. On the other hand, the K-Nearest Neighbors (KNN) model with 5 neighbors exhibited 
		   a higher MSE of 0.0721, suggesting a relatively less accurate prediction compared to SVM and RF. In summary, 
		   the linear SVM model stands out as the most effective in this context, providing a reliable approach for wild 
		   blueberry yield prediction.''')