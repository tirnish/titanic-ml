from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from src.preprocessing import create_preprocessor

def create_final_model():
    preprocessor = create_preprocessor()

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            subsample=1,
            n_estimators=200,
            min_samples_split=10,
            min_samples_leaf=1,
            max_depth=5,
            learning_rate=0.01,
            random_state=42
        ))
    ])
    
    return model