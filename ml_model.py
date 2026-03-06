import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

class BayesClassifier:
    def evaluate_model(self, df, target_col, target_value, feature_cols):
        # 1. Limpiar datos vacíos
        data = df.dropna(subset=feature_cols + [target_col]).copy()
        
        # 2. Binarizar la variable objetivo (1 si es el evento que buscamos, 0 si no)
        y = (data[target_col] == target_value).astype(int)
        X = data[feature_cols]
        
        # 3. Convertir variables categóricas (texto) a números (One-Hot Encoding)
        X = pd.get_dummies(X, drop_first=True)
        
        # Validar que tengamos datos suficientes
        if len(X) < 10:
            raise ValueError("No hay suficientes datos válidos para entrenar.")
            
        # 4. Dividir en Entrenamiento (70%) y Prueba (30%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 5. Entrenar Naive Bayes
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        # 6. Predecir valores reales vs modelo
        y_pred = model.predict(X_test)
        
        # 7. Métricas (Fijamos los labels a [0, 1] para asegurar una matriz 2x2 siempre)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        acc = accuracy_score(y_test, y_pred)
        
        # Extraer TN, FP, FN, TP de la matriz
        tn, fp, fn, tp = cm.ravel()
        
        sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
            
        return cm, acc, sensibilidad, especificidad