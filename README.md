1. Problème 

La croissance démographique du Canada représente un enjeu majeur pour la planification des 
services publics, notamment dans le domaine de la santé. Sans outils de prévision fiables, il devient 
difficile pour les décideurs d'anticiper les besoins futurs en infrastructures. 
Ce projet vise à répondre à la question suivante : comment prédire l'évolution de la population 
canadienne à l'horizon 2045 et en déduire les besoins en centres de santé nécessaires ? 

2. Solution proposée 

La solution développée repose sur un pipeline complet de science des données : collecte de données 
officielles, analyse exploratoire, entraînement de plusieurs modèles de machine learning, sélection 
du meilleur modèle, et déploiement via une API accessible. 
L'objectif est de fournir un outil de prédiction interactif permettant d'estimer la population 
canadienne pour n'importe quelle année future à partir de 2025, ainsi que d'en déduire le nombre 
de centres de santé requis. 

3. Données 

3.1 Source et format 

Les données utilisées proviennent de Statistique Canada (données officielles) et couvrent la période 
1980 à 2025. Elles sont structurées en format tabulaire (CSV) avec 44 lignes et 15 colonnes. 

3.2 Variables principals

Population totale 

Naissances et décès 

Immigration et émigration 

Taux de natalité et de mortalité 

Accroissement naturel 

4. Analyse exploratoire des données (EDA) 

L'analyse exploratoire a permis de dégager les tendances suivantes : 

Croissance continue de la population sur toute la période étudiée 

Accélération notable de la croissance après 2015 

Immigration en forte hausse, devenant le principal moteur de croissance 

Aucune donnée manquante, les données sont complètes et fiables 

5. Modèles testés et méthodologie 

5.1 Modèles comparés 
Plusieurs algorithmes de machine learning ont été évalués afin d'identifier le modèle le plus précis 
pour la prédiction de séries temporelles démographiques. 

Meilleur 

GrowthModel 

Autres modèles 

Régression linéaire 

modèle polynomiale

Ridge

modèle exponentielle

5.2 Résultats de la comparaison 

Le modèle GrowthModel s'est avéré le plus performant avec :

- Le meilleur score R² explique la plus grande proportion de la variance

- La meilleure précision globale sur les données de test 

- Une capacité de généralisation adaptée aux projections à long terme 

6. Prédictions de la population 

Sur la base du modèle GrowthModel, les projections démographiques jusqu'en 2045 indiquent : 

Population estimée en 2045 

≈ 62 millions d'habitants 

Tendance générale : Croissance continue 

7. Impact sur les infrastructures de santé 
En appliquant le ratio actuel de un centre de santé pour 66 000 habitants, les projections 
permettent d'estimer les besoins futurs : 

Ratio actuel 
1 centre pour 66 000 habitants 
Nombre de centres aujourd'hui 
≈ 620 centres 

Centres supplémentaires requis d'ici 2045 

+315 centres 

Total estimé en 2045: ≈ 935 centres 

Ces résultats soulignent l'urgence d'une planification proactive en matière d'infrastructures de 
santé pour faire face à la croissance démographique prévue. 

8. Architecture de la solution 

L'architecture de la solution suit un flux de données clair et modulaire : 

Composants

Interface Web 

Point d'entrée pour l'utilisateur: API REST (FastAPI) 

Traitement des requêtes de prédiction 

Modèle (.pkl) 

Exécution du modèle de machine learning 

Résultat (JSON) 

Retour des prédictions à l'utilisateur 

Docker 

Conteneurisation pour le déploiement 

9. Déploiement 

L'API REST a été développée avec FastAPI et offre les fonctionnalités suivantes :

- Endpoint principal : /predict?year=XXXX 

-Réponse en format JSON (population + besoins en santé) 

-Déploiement avec Docker pour faciliter la portabilité 

- Accès local via : http://localhost:8000 

9.1 Mode d'utilisation 

Pour utiliser l'outil de prédiction : 

- Choisir une année à partir de 2025 

- Cliquer sur « Calculer » 

- Le résultat s'affiche automatiquement avec l'estimation de population et les besoins en 
santé 

10. Limites 

Plusieurs limites doivent être prises en compte dans l'interprétation des résultats : 

- Certaines données récentes (2023-2025) sont partiellement estimées 

- Les prévisions peuvent varier significativement selon les politiques d'immigration ou 
les conditions économiques 

11. Perspectives 

Des améliorations futures pourraient renforcer la robustesse et l'utilité du système :  

- Prédire la population par province pour une granularité accrue 

- Améliorer la précision du modèle en intégrant davantage de variables explicatives 

- Développer une interface utilisateur plus interactive et visuelle 

- Intégrer le modèle dans des applications réelles de planification publique 

Conclusion 

Ce projet démontre la faisabilité et la pertinence d'une approche de machine learning pour la 
prédiction démographique au Canada. Le modèle GrowthModel offre des projections fiables 
indiquant une population d'environ 62 millions d'habitants d'ici 2045, soit une augmentation de 50 
%. 
Ces résultats ont des implications directes pour la planification des infrastructures de santé : 
environ 315 nouveaux centres seraient nécessaires d'ici cette échéance. La solution déployée via 
FastAPI et Docker constitue un point de départ solide pour des outils décisionnels plus avancés.
