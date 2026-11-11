import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('dataset_tito_v1.csv', encoding='utf-8')
df.columns = df.columns.str.strip()

print(df.head())
print(df['columna'].value_counts())
X = df['texto']
y = df['columna']

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
preguntas = [
    'que necesito para una beca',
    'a que hora abre la biblioteca',
    'como me inscribo al siguiente semestre'
]
for pregunta in preguntas:
    categoria = modelo.predict([pregunta])[0]
    print(pregunta, '->', categoria)