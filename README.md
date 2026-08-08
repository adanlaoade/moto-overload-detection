# Détection de surcharge de passagers sur motos par vision artificielle

Système de vision par ordinateur pour détecter automatiquement la surcharge de passagers sur des motos au Bénin — zemidjan comme motos personnelles — avec module optionnel de lecture de plaque d'immatriculation.

## Démo en ligne

**Application déployée :** https://moto-surcharge.streamlit.app

## Contexte

Au Bénin, la surcharge de passagers sur moto (zemidjan ou usage personnel) est fréquente et difficile à contrôler manuellement. Ce projet explore la faisabilité d'un contrôle automatisé par vision artificielle, applicable à l'ensemble des motocyclistes, pas seulement au transport rémunéré.

## Fonctionnement

1. **Détection** : YOLOv8 (pré-entraîné sur COCO) détecte motos et personnes dans l'image.
2. **Association passager-moto** : une logique géométrique (chevauchement horizontal, puis affinée avec normalisation par échelle) relie chaque personne détectée à la moto correspondante.
3. **Classification** : si le nombre de personnes associées à une moto dépasse un seuil réglementaire (réglable), la moto est classée en surcharge.
4. **Module ANPR (optionnel)** : tentative de lecture de plaque d'immatriculation en cas de surcharge détectée (preuve de concept, cf. limites).

## Stack technique

- Python, YOLOv8 (Ultralytics)
- OpenCV
- Streamlit (interface web)
- Déployé sur Streamlit Community Cloud

## Installation locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure du repo

```
├── app.py                  # Application Streamlit principale
├── requirements.txt        # Dépendances Python
├── packages.txt             # Dépendances système (libGL, etc.)
├── docs/                    # Documentation méthodologique
└── README.md
```
## Notebook d'expérimentation

Le notebook complet (essais, calibrage, évaluation) est disponible ici :
https://colab.research.google.com/drive/1dP48-ucCO43oaix10gtRv26BZ5y7Qv7s?usp=sharing

## Méthodologie et limites

Ce projet repose sur l'assemblage de modèles pré-entraînés (YOLOv8, EasyOCR) plutôt qu'un réentraînement complet, choix assumé compte tenu des contraintes de temps du projet. Plusieurs approches d'association passager-moto ont été testées et comparées (voir article scientifique et documentation dans `docs/`). Les limites principales identifiées :

- L'association passager-moto par méthode géométrique plafonne en scène encombrée (piétons proches, occlusion), cohérent avec la littérature du domaine.
- Le module de lecture de plaque reste une preuve de concept, non fiabilisé pour un usage de verbalisation automatique.
- L'évaluation s'appuie sur un corpus d'images limité, pas sur un dataset annoté à grande échelle spécifique au contexte béninois.

## Auteur

Projet réalisé dans un cadre académique.

## Licence

MIT
