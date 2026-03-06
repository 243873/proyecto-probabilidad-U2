import pandas as pd

class DataProcessor:
    def load_data(self, file):
        return pd.read_csv(file)

    def detect_columns(self, df):
        # Detección automática de tipos de columnas
        cols = {
            "numeric": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime": df.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist(),
            "binary": [col for col in df.columns if df[col].nunique() == 2]
        }
        
        # Intento básico de detectar fechas en columnas de texto
        for col in cols["categorical"][:]: 
            try:
                pd.to_datetime(df[col], format="mixed")
                cols["datetime"].append(col)
                cols["categorical"].remove(col)
            except (ValueError, TypeError):
                pass
                
        return cols