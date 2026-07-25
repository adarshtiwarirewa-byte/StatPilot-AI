import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression,LogisticRegression
from sklearn.ensemble  import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics  import r2_score,root_mean_squared_error,accuracy_score




def detect_possibly_categorical_numeric(df:pd.DataFrame, feature_columns:list,threshold:int = 10) -> list :
    """
    Flags numeric feature columns that have few unique values,
    since they might actually represent categories, not continuous quantities.
    """

    flagged = []
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique()<= threshold:
            flagged.append(col)
    
    return flagged






def detect_problem_type(df:pd.DataFrame , target_column:str) -> str :
    """
    Decides whether target represents classification or regression problem.
    """
    target = df[target_column]

    if target.dtype == 'object' or target.dtype.name == 'category':
        return "classification"
    
    unique_values = target.nunique()
    if unique_values <= 10 :
        return "classification"
    
    return "regression"



def preprocess_data(df:pd.DataFrame,target_column:str,feature_columns:list,fill_strategy: dict = None,cardinality_threshold:int = 10,categorical_override:list=None):
    """
    Handles missing values (user-chosen strategy) and encodes categorical columns:
    - Low-cardinality columns (<= cardinality_threshold unique values) -> One-Hot Encoding
    - High-cardinality columns (> cardinality_threshold unique values) -> Target Encoding (with smoothing)
    - Target column itself -> Label Encoding (if categorical)

    categorical_override: list of numeric columns the user has confirmed should be treated as categorical.
    """

    data = df[feature_columns + [target_column]].copy()
    data = df.dropna(subset=[target_column])

    # --------Fill missing values ---------

    if fill_strategy is None:
        fill_strategy = {}
    if categorical_override is None:
        categorical_override = []

    for col in data.columns:
        if data[col].isna().sum() == 0:
            continue

        strategy = fill_strategy.get(col,{"method":"Auto"})
        method = strategy.get("method","Auto")

        if method == "Median":
            data[col] = data[col].fillna(data[col].median())
        elif method == "Mode":
            data[col] = data[col].fillna(data[col].mode()[0])
        elif method == "Zero":
            data[col] = data[col].fillna(0)
        elif method == "Costum Value":
            data[col] = data[col].fillna(0)
        else :
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode()[0])
        




#========== Convert user-confirmed categorical numeric columns to string  ======
    for col in categorical_override:
        data[col] = data[col].astype(str)



# ------- Prepare target for target-encoding calculations -------
    if not pd.api.types.is_numeric_dtype(data[target_column]):
        target_le = LabelEncoder()
        y_for_encoding = target_le.fit_transform(data[target_column].astype(str))
    else:
        y_for_encoding = data[target_column].values
    
    global_mean = y_for_encoding.mean()
    smoothing = 10  # higher = trusts global mean more for rare categories


#----------Encode categorical feature columns ---------
    low_cardinality_cols = []

    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            continue


        nunique = data[col].nunique()
        if nunique <= cardinality_threshold:
            low_cardinality_cols.append(col)
        else: 
            temp = pd.DataFrame({col: data[col], "target": y_for_encoding})
            agg = temp.groupby(col)["target"].agg(["mean", "count"])
            agg["smoothed"] = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
            data[col] = data[col].map(agg["smoothed"])
    
    if low_cardinality_cols:                         # one hot encoding  to be done for these
        data = pd.get_dummies(data,columns=low_cardinality_cols)
    

#=========Encode target column------------
    if not pd.api.types.is_numeric_dtype(data[target_column]):
        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column].astype(str))

    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y






def train_and_evaluate(X,y,problem_type:str):
    """
    Trains a couple of baseline models and returns their evaluation results.
    """

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    result = []
    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42,n_estimators=100,n_jobs=-1),
        }

        for name,model in models.items():
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test,pred)
            result.append({"Model":name,"Accuracy": round(acc,4)})

    else :
        models = {"Linear Regression": LinearRegression(),
                  "Random Forest": RandomForestRegressor(random_state=42,n_estimators=100,n_jobs=-1)}
        
        for name,model in models.items():
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            rmse = root_mean_squared_error(y_test, pred)
            result.append({"Model": name, "R2 Score": round(r2, 4), "RMSE": round(rmse, 4)})

    
    return pd.DataFrame(result)