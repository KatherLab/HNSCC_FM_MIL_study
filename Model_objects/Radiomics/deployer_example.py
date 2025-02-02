import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pickle as pkl

with open("xxx/OS_Radiomics.pkl", "wb") as file:
    pkl.dump(a, file)
def radiomics_deployer(deploy_feats_pth:Path,
                 deploy_clinitable_pth:Path,
                 object: dict,
                 target: str,
                 id_col: str):
    '''
    Utilises the model object from logistic_binary_radiomics to preprocess the data according to training specifications 
    as well as predict on the new cohort.
    Inputs:
    deploy_feats_pth: Path, path to the radiomcis features of the cohort
    deploy_clinitable_pth: Path, path to the deployment clinitable with ID and target identifers
    object: dict, output from the logistic_binary_radiomics models 
    target: str, name of the target to use the model to predict
    id_col: str, patient identifier column name
    Outputs:
    Dictionary with model information such as paths, features, probabilities, target levels and model object from training.
    '''
    rad_features = pd.read_csv(deploy_feats_pth)
    clini = pd.read_csv(deploy_clinitable_pth)
    clini= clini[[id_col,target]].merge(rad_features, how="left",on=id_col)
    #Now we get the true clinitable, with the 
    #Let us separate the x and y:
    clini_x = object["Standard_Scaler"].transform(clini[object["Variance_Scaler"].get_feature_names_out()])
    clini_x= pd.DataFrame(clini_x, columns=object["Variance_Scaler"].get_feature_names_out() )
    clini_x = clini_x.loc[:,object["Out_features"]]
    trgt= clini[target].copy()
    clini_y= clini[target].replace(object["Levels"])
    probs= object["Model_object"].predict(clini_x)
    return({"Path_rad_feats":deploy_feats_pth,
            "Path_clinitable":deploy_clinitable_pth,
            "IDs":clini[id_col],
            "Features":clini_x,
            "True_labels_deploy":trgt,
            "Probabilities":probs,
            "Model_object": object["Model_object"],
            "Levels":object["Levels"]})
