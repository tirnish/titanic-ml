from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.preprocessing import create_preprocessor

def create_final_model():
    preprocessor = create_preprocessor()

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(
            kernel='linear',
            gamma='auto',
            C=0.1623776739188721
        ))
    ])
    
    return model