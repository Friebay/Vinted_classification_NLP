import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import ParameterGrid
import joblib


def create_model(input_size, hidden_sizes, num_classes, dropout_rate, learning_rate):
    model = models.Sequential()
    
    model.add(layers.Input(shape=(input_size,)))

    for i, hidden_size in enumerate(hidden_sizes):
        model.add(layers.Dense(hidden_size, activation='relu', name=f'hidden_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
    
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    # Set random seeds for reproducibility
    np.random.seed(1234)
    tf.random.set_seed(1234)
    
    df = pd.read_csv('final_df_1.csv').drop(columns=['Unnamed: 0'])
    print(f"Data loaded. Shape: {df.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1], 
        df.iloc[:, -1], 
        test_size=0.2, 
        random_state=1
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    input_size = X_train_scaled.shape[1]
    num_classes = len(np.unique(y_train))

    print("Starting neural network hyperparameter search")
    
    param_dist = {
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        'hidden_sizes': [[256, 128, 64]],
        'batch_size': [64, 128, 256, 512],
        'dropout_rate': [0.01, 0.05, 0.1, 0.2]
    }
    
    best_params = None
    best_fscore = 0.0
    best_model = None
    num_epochs = 500
    
    for params in ParameterGrid(param_dist):
        print('\nStarted:', params)
        
        model = create_model(
            input_size=input_size,
            hidden_sizes=params['hidden_sizes'],
            num_classes=num_classes,
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate']
        )
                
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            restore_best_weights=True
        )

        history = model.fit(
            X_train_scaled, 
            y_train,
            batch_size=params['batch_size'],
            epochs=num_epochs,
            verbose=0,
            validation_split=0.1,
            callbacks=[early_stop]
        )

        
        y_pred_proba = model.predict(X_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        f_score_iter = f1_score(y_test, y_pred, average="macro")
        
        if f_score_iter > best_fscore:
            best_fscore = f_score_iter
            best_params = params
            best_model = model
        
        print('Parameters:', params, '\nF score:', f_score_iter)
    
    print("\n")
    print("Training final model with best parameters")
    print(f"Best parameters: {best_params}")
    print(f"Best F-score: {best_fscore}")
    
    # Create and train final model with more epochs
    final_model = create_model(
        input_size=input_size,
        hidden_sizes=best_params['hidden_sizes'],
        num_classes=num_classes,
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate']
    )
    
    print("\nTraining final model")
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        restore_best_weights=True
    )

    history = final_model.fit(
        X_train_scaled,
        y_train,
        batch_size=best_params['batch_size'],
        epochs=500,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stop]
    )

    
    # Evaluate final model
    y_pred_proba_final = final_model.predict(X_test_scaled, verbose=0)
    y_pred_final = np.argmax(y_pred_proba_final, axis=1)
    
    print("\n")
    print("Final Model Evaluation:")
    print('Accuracy:', accuracy_score(y_test, y_pred_final))
    print('Recall (per class):', recall_score(y_test, y_pred_final, average=None))
    print('Precision (per class):', precision_score(y_test, y_pred_final, average=None))
    print('F1-score (macro):', f1_score(y_test, y_pred_final, average='macro'))
    print('F1-score (micro):', f1_score(y_test, y_pred_final, average='micro'))
    
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred_final))

    final_model.save('best_nn_model.keras')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(best_params, 'best_params.joblib')

    print("\nModel saved to 'best_nn_model.keras'")
    
    return final_model, best_params, scaler


if __name__ == '__main__':
    model, params, scaler = main()