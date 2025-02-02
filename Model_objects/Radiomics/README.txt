Radiomics models are stored here within .pkl files. 
Radiomics features (PyRadiomics) are reached via variance filtering, clustering and MRMR selection (Peng et al. 2003) on the cluster representatives.
.pkl objects are organised as dictionaries with the following key-value schema:
"Variance_Scaler": 
"feats_after_variance":
"Standard_Scaler":
"Model_object": statsmodels smf.glm model with Binomial family with binary classification (statsmodels == 0.14.2)
"Out_dataframe": dataframe of features used for model building
"Out_features": column names of the dataframe, chosen features
"Probabilities": predicted probabilities of the training cohort
"True_labels_train": target values the model is trained on.
"Levels": levels used to code the classes, dict
