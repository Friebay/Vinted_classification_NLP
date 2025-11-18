import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
import scipy.stats
from sklearn.model_selection import ParameterSampler, ParameterGrid
import joblib


def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('final_df_1.csv').drop(columns=['Unnamed: 0'])
    print(f"Data loaded. Shape: {df.shape}")
    
    # Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1], 
        df.iloc[:, -1], 
        test_size=0.2, 
        random_state=1
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # XGBoost hyperparameter search
    print("\n" + "="*60)
    print("Starting XGBoost hyperparameter search...")
    print("="*60)
    
    param_dist = {
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [8, 12, 16],
        'n_estimators': [200, 400, 600]
    }
    
    best_params = None
    best_fscore = 0.0
    
    for params in ParameterGrid(param_dist):
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=6,
            **params
        )
        
        print('\nStarted:', params)
        
        xgb_model.fit(X_train, y_train)
        
        y_pred = xgb_model.predict(X_test)
        f_score_iter = f1_score(y_test, y_pred, average="macro")
        
        if f_score_iter > best_fscore:
            best_fscore = f_score_iter
            best_params = params
        
        print('Parametrai:', params, '\nF matas:', f_score_iter)
    
    # Train final model with best parameters
    print("\n" + "="*60)
    print("Training final model with best parameters...")
    print("="*60)
    print(f"Best parameters: {best_params}")
    print(f"Best F-score: {best_fscore}")
    
    final_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=6,
        **best_params
    )
    
    final_model.fit(X_train, y_train)
    y_pred_final = final_model.predict(X_test)
    
    # Evaluate final model
    print("\n" + "="*60)
    print("Final Model Evaluation:")
    print("="*60)
    print('Accuracy:', accuracy_score(y_test, y_pred_final))
    print('Recall (per class):', recall_score(y_test, y_pred_final, average=None))
    print('Precision (per class):', precision_score(y_test, y_pred_final, average=None))
    print('F1-score (macro):', f1_score(y_test, y_pred_final, average='macro'))
    print('F1-score (micro):', f1_score(y_test, y_pred_final, average='micro'))
    
    # Confusion matrix
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred_final))

    # Save the model
    joblib.dump(final_model, 'best_xgb_model2.joblib')
    joblib.dump(best_params, 'best_params2.joblib')

    print("\nModel saved to 'best_xgb_model2.joblib'")
    
    return final_model, best_params


if __name__ == '__main__':
    model, params = main()