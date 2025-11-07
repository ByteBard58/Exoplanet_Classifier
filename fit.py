from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import time
import pandas as pd
import numpy as np
import joblib

# Loading and processing 1st Dataset (NASA Kepler Objects of Interest)
df_raw = pd.read_csv("data/kepler_data.csv", comment="#")

feature_list = ["koi_disposition","koi_period","koi_time0bk",
 "koi_depth","koi_prad","koi_sma","koi_incl","koi_teq","koi_insol",
 "koi_impact","koi_ror","koi_srho","koi_dor","koi_num_transits"
]

df_selected = df_raw[feature_list]

df_1 = df_selected.copy()

# Loading and processing 2nd dataset (NASA K2 Objects of Interest)
df_2 = pd.read_csv("data/k2_data.csv",comment="#")

## Feature Engineering missing column in K2 (koi_num_transits) 
# Campaign dates dictionary (BJD)
campaign_dates = {
    0: (2456725.0, 2456805.0),
    1: (2456808.0, 2456891.0),
    2: (2456893.0, 2456975.0),
    3: (2456976.0, 2457064.0),
    4: (2457065.0, 2457159.0),
    5: (2457159.0, 2457246.0),
    6: (2457250.0, 2457338.0),
    7: (2457339.0, 2457420.0),
    8: (2457421.0, 2457530.0),
    9: (2457504.0, 2457579.0),
    10: (2457577.0, 2457653.0),
    11: (2457657.0, 2457732.0),
    12: (2457731.0, 2457819.0),
    13: (2457820.0, 2457900.0),
    14: (2457898.0, 2457942.0),
    15: (2457941.0, 2458022.0),
    16: (2458020.0, 2458074.0),
    17: (2458074.0, 2458176.0),
    18: (2458151.0, 2458201.0),
    19: (2458232.0, 2458348.0)
}

def get_window(camps):
    if pd.isna(camps) or not camps:
        return np.nan, np.nan
    
    camps = str(camps).split(',') if isinstance(camps, str) else camps
    
    # Filter valid campaign numbers and get start/end times
    starts = []
    ends = []
    for c in camps:
        try:
            camp_num = int(c.strip())
            if camp_num in campaign_dates:
                start, end = campaign_dates[camp_num]
                starts.append(start)
                ends.append(end)
        except (ValueError, KeyError):
            continue  
    
    return (min(starts) if starts else np.nan, max(ends) if ends else np.nan)

df_2['campaigns'] = df_2['k2_campaigns']
df_2[['obs_start_bjd', 'obs_end_bjd']] = df_2['campaigns'].apply(lambda x: pd.Series(get_window(x)))

# For transit counting (as before)
df_2['n_min'] = np.ceil((df_2['obs_start_bjd'] - df_2['pl_tranmid']) / df_2['pl_orbper'])
df_2['n_max'] = np.floor((df_2['obs_end_bjd'] - df_2['pl_tranmid']) / df_2['pl_orbper'])
df_2['num_transits'] = (df_2['n_max'] - df_2['n_min'] + 1).clip(lower=0)
df_2 = df_2[["disposition","pl_orbper","pl_tranmid","pl_trandep","pl_rade","pl_orbsmax","pl_orbincl","pl_eqt","pl_insol","pl_imppar","pl_ratror","pl_dens","pl_ratdor","num_transits"]]

# Concatenating df_1 and df_2
mapping = {"disposition":"koi_disposition","pl_orbper":"koi_period","pl_tranmid":"koi_time0bk",
           "pl_trandep":"koi_depth","pl_rade":"koi_prad","pl_orbsmax":"koi_sma",
           "pl_orbincl":"koi_incl","pl_eqt":"koi_teq","pl_insol":"koi_insol","pl_imppar":"koi_impact",
           "pl_ratror":"koi_ror","pl_dens":"koi_srho","pl_ratdor":"koi_dor","num_transits":"koi_num_transits"
           }
df_2 = df_2.rename(columns=mapping)

df = pd.concat([df_1,df_2])

# Input-output separation
X = df.iloc[:,1:]
column_name = X.columns
X = X.to_numpy()
y = df["koi_disposition"].map({"FALSE POSITIVE":0,"CANDIDATE":1,"CONFIRMED":2,"REFUTED":0}).to_numpy()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, shuffle=True, random_state=91, stratify=y
)

# Define Models
rf = RandomForestClassifier(n_estimators=1000,max_depth=None,random_state=542,class_weight="balanced")
xgb = XGBClassifier(n_estimators=1000,max_depth=None,learning_rate=0.5,random_state=9)

# Define estimators list for stacking
estimators = [
    ("rf",rf),
    ("xgb",xgb)
]

# Define final estimator for stacking
final_estimator = LogisticRegression(
    random_state=891,class_weight="balanced",C=0.1,penalty="l2",solver="saga",max_iter=5000
)

# Stacking
mv = StackingClassifier(
    estimators=estimators,final_estimator=final_estimator,
    cv=5,passthrough=True,n_jobs=-1
)

# Pipeline
pipe_mv = Pipeline([
  ("impute",SimpleImputer(strategy="mean")),
  ("scale",StandardScaler()),
  ("smote",SMOTE()),
  ("model",mv)
])

print("Starting model training. It will take some time, sit tight......")
t1 = time.time()
pipe_mv.fit(x_train,y_train)
t2 = time.time()
print("Model trained successfully")
minutes,seconds = np.divmod(t2-t1,60)
print(f"Time Elapsed: {minutes} M {seconds:.2f} S")

# Dumping the model in pickle files
joblib.dump(pipe_mv,"models/pipe.pkl")
joblib.dump(column_name,"models/column_names.pkl")