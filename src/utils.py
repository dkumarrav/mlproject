import os
import sys
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging  # ✅ Added logging

def save_object(file_path, obj):
    """
    Saves a Python object using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")  # ✅ Logging success
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Trains multiple models using hyperparameter tuning and evaluates them.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")  # ✅ Logging which model is training

            para = param.get(model_name, {})  # ✅ Safely get parameters

            # Only use GridSearch if there are hyperparameters to tune
            if para:
                logging.info(f"Performing GridSearch for {model_name}...")
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1)  # ✅ Optimized GridSearchCV
                gs.fit(X_train, y_train)
                best_params = gs.best_params_
                logging.info(f"Best params for {model_name}: {best_params}")
            else:
                best_params = {}  # No tuning needed

            # Train model with the best parameters (or default if no tuning)
            model.set_params(**best_params)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # R² scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score  # Store test R² score

            logging.info(f"{model_name}: Train R² = {train_model_score:.4f}, Test R² = {test_model_score:.4f}")

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
