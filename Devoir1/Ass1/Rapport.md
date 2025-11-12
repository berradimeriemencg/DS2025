from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Récupérer le dataset mushroom
mushroom = fetch_ucirepo(id=73)
X = mushroom.data.features
y = mushroom.data.targets

# Concaténer X et y pour analyse complète
df = pd.concat([X, y], axis=1)

# Afficher quelques métadonnées
print("Metadata:\n", mushroom.metadata)
print("\nVariables:\n", mushroom.variables)

# Analyse descriptive statistique
print("\nStatistiques descriptives:\n", df.describe(include='all'))

# Visualisation des distributions pour les variables catégorielles
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x=col, hue=y.name)
    plt.title(f"Distribution de {col} par la cible")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Visualisation des relations entre variables (exemple avec un pairplot sur un sous-échantillon si trop de variables)
subset = df[categorical_cols].iloc[:, :5]  # prendre les premières 5 colonnes catégorielles
subset[y.name] = y
sns.pairplot(subset, hue=y.name)
plt.show()

