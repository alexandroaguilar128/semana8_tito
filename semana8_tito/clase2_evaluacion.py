import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('dataset_tito_v1.csv', encoding='utf-8')
df.columns = df.columns.str.strip()

col_objetivo = 'categoria' if 'categoria' in df.columns else 'columna'

X = df['texto']
y = df[col_objetivo]

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y
)

modelo = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clasificador', MultinomialNB())
])
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("=" * 60)
print(f"ACCURACY DEL MODELO: {round(accuracy, 3)}")
print("=" * 60)

print("\n--- REPORTE DE CLASIFICACIÓN ---")
print(classification_report(y_test, y_pred, zero_division=0))

categorias = sorted(y.unique())
matriz = confusion_matrix(y_test, y_pred, labels=categorias)

print("--- MATRIZ DE CONFUSIÓN ---")
print("Categorías en orden:", categorias)
print("\nMatriz:")
print(matriz)

resultados = pd.DataFrame({
    'pregunta': X_test,
    'real': y_test,
    'predicha': y_pred
})

errores = resultados[resultados['real'] != resultados['predicha']]

print("\n--- ANÁLISIS DE ERRORES REALES DETECTADOS ---")
if errores.empty:
    print("¡No se registraron errores en el conjunto de prueba!")
else:
    print(f"Total de errores: {len(errores)} de {len(y_test)} muestras de prueba\n")
    print(errores.to_string(index=False))

joblib.dump(modelo, 'modelo_tito_semana8.joblib')
print("\nModelo guardado exitosamente como 'modelo_tito_semana8.joblib'")