
nom : berradi meriem 
appogée:22007719

![image](https://github.com/user-attachments/assets/5616a879-06b7-4f83-ba71-84475fc8ac15)


# 📘 **RAPPORT DESCRIPTIF – PROJET DATA SCIENCE**

## *Dataset : Wine Recognition (Classification Multiclasses)*

---

## **1. Contexte & Objectif du Projet**

Le dataset **Wine Recognition** est un classique de la Data Science.
Il permet de classifier trois types de vins (classes 0, 1 et 2) à partir de 13 caractéristiques chimiques telles que l’alcool, le magnésium, l’acidité, les polyphénols ou encore la couleur.

### 🎯 **Objectif principal**

Construire une chaîne complète de Machine Learning permettant de :

1. Charger et inspecter le dataset
2. Simuler un cas réel en introduisant des données manquantes
3. Nettoyer les données
4. Réaliser une analyse exploratoire
5. Construire un modèle de classification
6. Évaluer sa performance

---

## **2. Les Données (X et y)**

### 🔹 Variables explicatives (X)

13 caractéristiques chimiques des vins, parmi lesquelles :

* Alcohol
* Malic acid
* Ash
* Hue
* Proline (la caractéristique la plus discriminante)

### 🔹 Variable cible (y)

Trois classes :

* **0 : Classe 0**
* **1 : Classe 1**
* **2 : Classe 2**

➡️ Le dataset original contient **178 lignes** et **14 colonnes (13 features + 1 target)**.

---

## **3. Simulation de Données Manquantes (Réalisme du Projet)**

Pour simuler un cas réel – où les mesures chimiques peuvent manquer – tu as introduit :

* **7% de valeurs manquantes dans toutes les colonnes (sauf target)**

Total NaN générés :
👉 **~162 valeurs manquantes**

Ce processus reproduit un scénario industriel où les données sont imparfaites.

---

## **4. Nettoyage & Imputation**

### 🔧 Stratégie utilisée :

`SimpleImputer(strategy='mean')`

Pour chaque feature :

* `fit()` calcule la moyenne
* `transform()` remplace les NaN par cette moyenne

🔹 Résultat :
**0 valeur manquante restante** après imputation.

📌 *Note méthodologique*
Dans un projet strictement professionnel, on ferait l’imputation **après** le train/test split pour éviter le **data leakage**.

---

## **5. Analyse Exploratoire (EDA)**

### 📊 Statistiques descriptives

Les premières colonnes montrent des amplitudes différentes :

* Variables comme **alcohol** → bien réparties
* Variables comme **color_intensity** → asymétriques
* **Proline** → très dispersée (écart-type élevé), bonne variable discriminante

### 📉 Distribution de l’alcool par classe

Le graphique montre une séparation claire :

* Classe 0 → vins plus alcoolisés
* Classe 2 → valeurs plus faibles
* Classe 1 → intermédiaire

Cela indique que **l’alcool est un bon prédicteur**.

### 🔥 Matrice de corrélation

On observe des relations fortes :

* **Proline**, **alcalinity_of_ash**, **OD280** ont des corrélations marquées avec la classe.
* Certaines variables sont redondantes (corrélation > 0.7)

---

## **6. Méthodologie de Split (Train/Test)**

`train_test_split(test_size=0.2, random_state=42)`

* Train : **142 lignes**
* Test : **36 lignes**

Pourquoi 20% ?
→ Suffisant pour valider un modèle tout en conservant un volume d’entraînement solide.

Pourquoi random_state=42 ?
→ Reproductibilité des expériences.

---

## **7. Modélisation : Random Forest**

### Pourquoi un Random Forest ?

* Robuste aux données bruitées
* Supporte bien les corrélations entre variables
* Peu sensible aux valeurs extrêmes
* Capable de capturer des interactions complexes

### Paramètres principaux :

* `n_estimators=150`
* `random_state=42`

Le modèle apprend à distinguer les 3 classes de vin grâce aux patterns extraits.

---

## **8. Évaluation du Modèle**

### 🎯 **Accuracy obtenue : environ 97–100 %**

Très fort score grâce :

* À la simplicité du dataset
* À un fort pouvoir discriminant des features chimiques
* À la stabilité du Random Forest

### 📌 Rapport de classification :

Les métriques (precision, recall, f1-score) sont **excellentes pour les 3 classes**.

### 🧩 Matrice de confusion

Peu ou pas d’erreurs de classification :

* Les classes sont clairement séparables
* Le modèle capture très bien leurs signatures chimiques

---

## **9. Conclusion Générale**

Ce projet illustre parfaitement toutes les étapes d’un pipeline complet de Data Science :

1. Chargement des données
2. Simulation de données manquantes
3. Nettoyage (imputation)
4. EDA visuelle et statistique
5. Split méthodologique
6. Modélisation Random Forest
7. Évaluation pertinente

🎓 **Compétences validées :**

* Data wrangling
* Analyse exploratoire
* Nettoyage des valeurs manquantes
* Construction d’un modèle supervisé
* Lecture de métriques ML
* Interprétation de graphes

Le dataset WINE est un excellent terrain d’apprentissage pour comprendre la classification multiclasses et l’impact des variables chimiques sur la typologie des vins.

---

