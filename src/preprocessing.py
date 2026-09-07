import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, FunctionTransformer

def create_preprocessor():
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])

    log_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p)),
    ])

    pt_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('power_tr', PowerTransformer(method='yeo-johnson')),
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first'))
    ])

    preprocessor_tree = ColumnTransformer([
        ('num', num_pipeline, ['Age', 'Parch', 'Pclass']),
        ('log', log_pipeline, ['Fare']),
        ('pt', pt_pipeline, ['SibSp']),
        ('cat', cat_pipeline, ['Sex', 'Embarked'])
    ])

    return preprocessor_tree