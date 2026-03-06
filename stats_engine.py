import pandas as pd

class StatsEngine:
    def calculate_bayes(self, df, target_col, target_value, feature_col, threshold):
        # Evento A: Donde la columna objetivo es igual al valor seleccionado
        target_mask = df[target_col] == target_value
        p_a = target_mask.mean() # True=1, False=0, por lo que el promedio da la probabilidad
        
        # Evento B: La evidencia
        if pd.api.types.is_numeric_dtype(df[feature_col]):
            try:
                event_b_mask = df[feature_col] > float(threshold)
            except ValueError:
                event_b_mask = df[feature_col] == threshold
        else:
            event_b_mask = df[feature_col] == threshold
            
        p_b = event_b_mask.mean()
        
        # P(B|A): Probabilidad de la evidencia dado el objetivo
        if target_mask.sum() > 0:
            p_b_given_a = (event_b_mask & target_mask).sum() / target_mask.sum()
        else:
            p_b_given_a = 0
            
        # P(A|B): Teorema de Bayes
        if p_b > 0:
            p_a_given_b = (p_b_given_a * p_a) / p_b
        else:
            p_a_given_b = 0
            
        return p_a, p_b_given_a, p_a_given_b