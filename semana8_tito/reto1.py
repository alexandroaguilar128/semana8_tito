import joblib
import os
import pandas as pd

MODEL_PATH = 'modelo_tito_semana8.joblib'

if not os.path.exists(MODEL_PATH):
    print("No se encontró el modelo guardado. Ejecutando entrenamiento rápido...")
    import clase2_evaluacion

modelo = joblib.load(MODEL_PATH)

print("\n=== MODO INTERACTIVO - TITO CONSEJERO ===")
print("Escribe tu pregunta para Tito (o escribe 'salir' para terminar)\n")

while True:
    pregunta = input("Pregunta para Tito: ").strip()
    if pregunta.lower() == 'salir':
        print("Cerrando sesión de Tito.")
        break
    
    if not pregunta:
        continue

    categoria = modelo.predict([pregunta])[0]
    print(f"-> Categoría detectada: [{categoria}]\n")