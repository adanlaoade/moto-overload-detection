# Détection automatique de la surcharge de passagers sur motos par vision artificielle

**LUBUNGU KISAMBULA Pacifique, ADANLAO Adéyinka Laurinda**
*AMA — Académie des Mathématiques Appliquées, Groupe 1*

Système de vision par ordinateur pour détecter automatiquement la surcharge de passagers sur des motos au Bénin — zemidjan comme motos personnelles — avec module optionnel de lecture de plaque d'immatriculation. Ce travail s'inscrit dans la dynamique de vidéoprotection engagée par le gouvernement béninois (Conseil des ministres du 4 mars 2026).

## Liens

- **Application déployée :** https://moto-surcharge.streamlit.app
- **Article scientifique (PDF) :** [`article/article.pdf`](./article/article.pdf)
- **Présentation (démo day) :** [`presentation/presentation_demoday.pptx`](./presentation/presentation_demoday.pptx)
- **Notebook d'expérimentation (Colab) :** [lien à insérer ici]

## Résultats clés

| Métrique | Valeur |
|---|---|
| Exactitude de statut (conforme/surcharge) | 66,4 % |
| Précision (classe surcharge) | 92,7 % |
| Rappel (classe surcharge) | 57,3 % |
| Corpus d'évaluation | 250 images annotées (Roboflow *Tripple_Riding*) |
| Vérification contextuelle | 134 images de motocyclistes béninois |

Détails complets, méthodologie et limites : voir l'article scientifique.

## Fonctionnement

1. **Détection** : YOLOv8n (pré-entraîné COCO) détecte motos et personnes ; YOLOv8n-pose extrait les points-clés corporels (posture).
2. **Association passager-moto** : normalisation géométrique par l'échelle de la moto + résolution par algorithme hongrois (assignment optimal), avec déduplication des détections dupliquées par IoU. Seuil de détection moto calibré empiriquement à 0,20.
3. **Classification** : si le nombre de personnes associées à une moto dépasse un seuil réglementaire (réglable, par défaut 3), la moto est classée en surcharge.
4. **Module ANPR (optionnel)** : tentative de lecture de plaque d'immatriculation en cas de surcharge détectée — preuve de concept, non fiabilisée (voir limites).

## Stack technique

- Python, YOLOv8 / YOLOv8-pose (Ultralytics)
- OpenCV, SciPy (algorithme hongrois)
- Streamlit (interface web), déployé sur Streamlit Community Cloud
- EasyOCR (module ANPR expérimental)

## Installation locale

```bash
git clone https://github.com/adanlaoade/moto-overload-detection.git
cd moto-overload-detection
pip install -r requirements.txt
streamlit run app.py
```

## Structure du repo

```
├── app.py                  # Application Streamlit (pipeline final)
├── requirements.txt        # Dépendances Python
├── packages.txt             # Dépendances système (libgl1, etc.)
├── src/                     # Scripts de pipeline / nettoyage de données
├── results/                 # Graphiques d'analyse (fig1 à fig4)
├── article/                 # Article scientifique (.tex + .pdf)
├── presentation/             # Support de présentation (.pptx)
├── LICENSE
└── README.md
```

## Méthodologie et limites

Ce projet repose sur l'assemblage de modèles pré-entraînés (YOLOv8, YOLOv8-pose, EasyOCR) plutôt qu'un réentraînement complet, choix assumé et documenté comme contribution (approche *training-free* pour un contexte sans corpus annoté local). Cinq approches d'association passager-moto ont été testées et comparées itérativement (voir article scientifique, section Méthodologie).

Limites principales :

- Le rappel (57,3 %) reste modéré : environ 43 % des surcharges réelles ne sont pas détectées, principalement en cas d'occlusion extrême de la moto par les passagers.
- Les résultats quantitatifs portent sur un corpus externe de référence (majoritairement indien) ; la vérification sur le contexte béninois reste qualitative, faute de vérité terrain indépendante disponible à ce stade.
- Le module de lecture de plaque reste une preuve de concept, non fiabilisé pour un usage de verbalisation automatique.
- Considérations éthiques (proportionnalité, biais de représentation, impact socio-économique) détaillées dans l'article scientifique, section Discussion.

## Auteurs

LUBUNGU KISAMBULA Pacifique, ADANLAO Adéyinka Laurinda — AMA, Groupe 1

## Licence

MIT
