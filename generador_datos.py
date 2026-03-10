import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Semilla para que los resultados sean reproducibles
np.random.seed(42)
random.seed(42)

# Generar 200 registros
n = 200

# 1. Columna de Fechas (Datetime)
fecha_base = datetime(2026, 1, 1)
fechas = [fecha_base + timedelta(days=i//3, hours=random.randint(0, 23), minutes=random.randint(0, 59)) for i in range(n)]

# 2. Columnas Numéricas
temperatura = np.random.normal(65, 12, n)  # Media de 65 grados
humedad = np.random.normal(50, 15, n)      # Media de 50%

# 3. Columnas Categóricas
turno = np.random.choice(["Mañana", "Tarde", "Noche"], n)
mantenimiento = np.random.choice(["Sí", "No"], n, p=[0.7, 0.3])

# 4. Columna Binaria Objetivo (Fallo 0/1) con lógica probabilística real
fallos = []
for i in range(n):
    prob_fallo = 0.05  # 5% de probabilidad base de que algo falle
    
    # Si la temperatura es alta, aumenta el riesgo drásticamente
    if temperatura[i] > 75:
        prob_fallo += 0.45 
        
    # Si no tuvo mantenimiento, también aumenta el riesgo
    if mantenimiento[i] == "No":
        prob_fallo += 0.30
        
    # Asignar 1 (Fallo) o 0 (Normal) basado en esa probabilidad
    fallos.append(1 if random.random() < prob_fallo else 0)

# Crear el DataFrame
df = pd.DataFrame({
    "Fecha_Registro": fechas,
    "Turno_Operativo": turno,
    "Temperatura_C": np.round(temperatura, 1),
    "Humedad_Pct": np.round(humedad, 1),
    "Mantenimiento_Previo": mantenimiento,
    "Fallo": fallos
})

# Guardar a CSV
nombre_archivo = "datos_prueba_servidores.csv"
df.to_csv(nombre_archivo, index=False)
print(f"¡Éxito! Archivo '{nombre_archivo}' generado correctamente con {n} registros.")