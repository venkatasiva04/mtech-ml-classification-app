a. Problem Statement
Build and compare multiple classification models on a single public dataset, evaluate them using Accuracy, AUC, Precision, Recall, F1, and MCC, and deploy an interactive Streamlit web application that lets a user upload test data, choose a model, and view its evaluation metrics.
b. Dataset Description
Dataset: Wisconsin Diagnostic Breast Cancer dataset (UCI, via scikit-learn).
Instances: 569    Features: 30 numeric features.
Target: binary classification (0 = malignant, 1 = benign).
Meets requirements: more than 500 instances and more than 12 features.
c. GitHub Repository Link
venkatasiva04/mtech-ml-classification-app
d. Models Used and Comparison Table
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9825	0.9954	0.9861	0.9861	0.9861	0.9623
Decision Tree	0.9123	0.9157	0.9559	0.9028	0.9286	0.8174
kNN	0.9561	0.9788	0.9589	0.9722	0.9655	0.9054
Naive Bayes	0.9298	0.9868	0.9444	0.9444	0.9444	0.8492
Random Forest (Ensemble)	0.9561	0.9937	0.9589	0.9722	0.9655	0.9054
Gradient Boosting (Ensemble)	0.9561	0.9907	0.9467	0.9861	0.9660	0.9058

Observations
ML Model Name	Observation About Model Performance
Logistic Regression	Achieved superior overall performance across all metrics (98.25% accuracy, 0.9623 MCC) due to clear linear decision boundaries after feature standard scaling.
Decision Tree	Exhibited structural overfitting on training features, resulting in lower generalization accuracy (91.23%) and lower MCC (0.8174) on unscaled test samples.
kNN	Delivered strong classification performance (95.61% accuracy, 0.9788 AUC) when paired with feature standardization, effectively capturing local feature clusters.
Naive Bayes	Produced high AUC (0.9868), indicating strong probabilistic class separation despite conditional feature independence assumptions.
Random Forest (Ensemble)	Demonstrated high stability and robust feature sampling, attaining 95.61% accuracy and an impressive AUC score of 0.9937.
Gradient Boosting (Ensemble)	Achieved the highest recall (0.9861) along with Logistic Regression, minimizing false negatives critical for medical diagnosis.
Overall Winner	Logistic Regression is the overall winning model for this dataset, leading in Accuracy (0.9825), F1-Score (0.9861), and MCC (0.9623).

