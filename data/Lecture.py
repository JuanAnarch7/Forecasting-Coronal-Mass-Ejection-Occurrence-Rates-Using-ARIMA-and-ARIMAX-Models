import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File path definition
file_path = "Parent_folder/datos_30_11_2025.txt" #(This line must be modified for your scpecific parent folrder)

# Column's name definition
nombre_columnas = ["Fecha", "Hora", "Central", "Ancho", "Rapidez", "Col4", "Col5", "Col6", "Aceleracion", "Masa", "Energia", "MPA", "Comentarios"]

# Create the DataFrame from the .txt
df = pd.read_csv(
    file_path,
    delimiter=r"\s+",  
    engine='python',  
    names=nombre_columnas,  
    index_col=False  
)
# Fecha to datetime
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y/%m/%d', errors='coerce')
# Central, Ancho, Rapidez, Aceleracion, Masa y Energía to numeric format
df[["Central", "Ancho", "Rapidez", "Aceleracion", "Masa", "Energia"]] = df[["Central", "Ancho", "Rapidez", "Aceleracion", "Masa", "Energia"]].apply(pd.to_numeric, errors="coerce")
# Preview
print("Preview of data")
print(df.head())
print(df["Fecha"].size)

# Saved Csv
df.to_csv("datos_procesados_2025_11_30.csv", index=False)
print("Saved CSV file: 'datos_procesados_2025_11_30.csv'")

