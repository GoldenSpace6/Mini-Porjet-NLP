# Conseiller de jeux Steam

Agent conversationnel adapté par LoRA

Mini-projet NLP, CY Tech, ING2 GSI
Esteban Vela, Vincent Delb, Nicolao Hyacinthe, Rayane Saighi
23 octobre 2026

---

## 1. Ce que fait l'agent

Un conseiller de jeux vidéo qui interprète un goût exprimé vaguement, pose une question quand la demande ne permet pas de trancher, puis recommande des jeux qui existent réellement en nommant le lien avec les goûts exprimés.

Principe : les *faits* viennent du catalogue, le *comportement* vient de l'adaptation LoRA. Le modèle ne connaît aucun jeu de mémoire. Il ne peut citer que les candidats qu'un module Python lui fournit à chaque tour.

| | |
|---|---|
| Traite | recommander, explorer un goût flou, répondre sur un jeu du catalogue, comparer deux jeux, affiner après un rejet, signaler qu'une licence citée est absente |
| Refuse | support technique, remboursement, compte Steam, consoles, et tout fait absent du catalogue |

---

## 2. Fichiers livrés

| Fichier | Contenu |
|---|---|
| `agent.ipynb` | le notebook complet, neuf parties, exécutable de bout en bout |
| `rapport.pdf` | le rapport |
| `README.md` | ce fichier |

Le notebook écrit lui-même ses huit modules Python via des cellules `%%writefile`. Il n'y a donc aucun fichier `.py` à fournir séparément : un Restart & Run All les recrée intégralement.

---

## 3. Environnement

| | |
|---|---|
| Plateforme | Kaggle Notebook, accélérateur GPU T4 |
| GPU | un seul rendu visible (`CUDA_VISIBLE_DEVICES=0`) : avec deux, le `Trainer` bascule en DataParallel alors que le modèle est sur le GPU 0 |
| Python | 3.13 |
| Paquets | `transformers`, `peft`, `datasets`, `accelerate`, `gradio`, installés par la première cellule |
| Modèle | `Qwen/Qwen2.5-1.5B-Instruct` en float16, 1,54 milliard de paramètres, 3,09 Go en mémoire GPU |
| Graine | `GRAINE = 2026`, posée sur `random` et `torch` |

---

## 4. Données à attacher avant exécution

Panneau de droite du notebook, *Add Input*.

### a) Le dataset Steam

`artermiloff/steam-games-dataset`, fichier `games_march2025_cleaned.csv`, 89 618 lignes et 47 colonnes.

Le chemin est détecté par `glob` : il n'y a aucune variable à modifier. Si le dataset n'est pas attaché, la cellule de configuration s'arrête sur un `assert` qui indique quoi faire.

### b) La sauvegarde de l'adaptateur `e4`

Notre propre sauvegarde, produite lors du premier entraînement (4 epochs sur la première version des données). Elle sert de troisième point de comparaison en partie 8.

Sans cette sauvegarde, le notebook reste exécutable. Il faut alors :

1. commenter la cellule de reprise, celle qui définit `CHEMIN_E4` ;
2. remplacer `PeftModel.from_pretrained(...)` par `get_peft_model(mod, config_lora)` en partie 7 ;
3. retirer `("e4", "default", True)` de la liste `CONDITIONS` en partie 8, ainsi que l'entrée correspondante du menu Gradio.

L'évaluation compare alors deux conditions au lieu de trois.

### Jeton Hugging Face

Aucun n'est nécessaire, le modèle est public. Si un secret `HF_TOKEN` existe il est utilisé, sinon le notebook affiche un avertissement et continue.

---

## 5. Comment exécuter

Run All, de haut en bas. Durée indicative sur T4 : environ 50 minutes, dont l'entraînement LoRA (1 epoch, 503 exemples) et l'évaluation des trois conditions sur les vingt requêtes de test.

Quatre points d'attention.

Après les huit cellules `%%writefile` se trouve une cellule de rechargement (`importlib.reload`). Toute modification d'un module exige de relancer cette cellule : `%%writefile` écrit sur le disque, mais Python garde l'ancienne version du module en mémoire.

L'adaptateur entraîné est sauvegardé dans `/kaggle/working/adaptateurs/e1`. Il disparaît avec la session, donc il faut le télécharger avant de fermer.

L'interface Gradio de la dernière cellule est lancée avec `prevent_thread_lock=True`, sans quoi un Save & Run All ne se termine jamais.

L'étape d'analyse s'exécute dans un bloc `mod.disable_adapter()`. Ce n'est pas un détail : l'adaptateur, entraîné uniquement sur du dialogue, détruit la capacité du modèle à produire du JSON. La section 8 du rapport détaille le diagnostic.

---

## 6. Organisation du notebook

| Partie | Contenu |
|---|---|
| 1 | Périmètre : ce que l'agent traite, ce qu'il refuse, cinq conversations types |
| 2 | Les huit modules (cellules `%%writefile`) |
| 3 | Construction du catalogue à partir du CSV Steam |
| 4 | Observation du modèle : tokenisation, logits, température, top-k, top-p, entropie |
| 5 | Baseline par prompting et comparaison de deux messages système |
| 6 | Génération du jeu d'entraînement et quatre contrôles chiffrés |
| 7 | Adaptation LoRA et courbe de perte |
| 8 | Évaluation comparative : récupération seule, base, `e4`, `e1` |
| 9 | Interface Gradio avec bascule entre les trois modèles |

---

## 7. Les huit modules

| Module | Rôle |
|---|---|
| `schemas.py` | contrat partagé : intentions, axes, plateformes, états vides |
| `axes.py` | table tags Steam vers axes qualitatifs, et conversion |
| `catalogue.py` | lecture du CSV, chaîne de filtres, fiches de jeu |
| `modele.py` | point d'entrée unique pour la génération |
| `prompts.py` | messages système : analyse, et deux variantes de conseil |
| `agent.py` | `analyser` (modèle vers JSON) et `recuperer` (Python vers candidats) |
| `pipeline.py` | enchaînement des trois étapes et état du dialogue |
| `donnees.py` | génération des exemples d'entraînement |

---

## 8. Fichiers produits par l'exécution

| Fichier | Contenu |
|---|---|
| `catalogue.parquet` | 14 657 jeux retenus, avec leurs axes |
| `exemples_v2_train.jsonl` | 503 exemples d'entraînement |
| `exemples_v2_val.jsonl` | 60 exemples de validation |
| `exemples_v2_test.jsonl` | 60 exemples de test interne |
| `adaptateurs/e1/` | l'adaptateur LoRA entraîné |
| `resultats_base.json` | sorties des vingt requêtes, modèle de base |
| `resultats_e4.json` | idem, adaptateur 4 epochs |
| `resultats_e1.json` | idem, adaptateur 1 epoch |

---

## 9. Résultats principaux

Catalogue : 14 657 jeux après filtrage au seuil de 200 avis, dont 98 % portent au moins deux axes qualitatifs.

Données : 640 exemples issus de 480 dialogues, 623 après contrôles, 454 débuts de réponse distincts, formule la plus fréquente à 4 %.

LoRA : 18 464 768 paramètres entraînés sur 1 580 643 840, soit 1,17 % du modèle. Perte de validation de 1,4434 à 0,8476 sur une epoch.

Sur les vingt requêtes de test :

| | base | e4 | e1 |
|---|---|---|---|
| Réponses citant un titre non fourni | 5/20 | 5/20 | 3/20 |
| Hors sujet : aucun jeu cité | 1/4 | 1/4 | 3/4 |
| Intention bien routée (commune) | 17/20 | 17/20 | 17/20 |
| JSON valide (commun) | 19/20 | 19/20 | 19/20 |
| Longueur moyenne, en mots | 81 | 40 | 36 |
| Accord avec le top-2 du score Python | 53 % | 67 % | 44 % |

Lecture : l'adaptation réduit l'invention et raccourcit les réponses. L'accord à 44 % signifie que le modèle arbitre parmi les candidats au lieu de recopier l'ordre du score Python.

---

## 10. Limites connues

- Les exemples d'entraînement sont produits par gabarits : leur diversité de formulation reste inférieure à celle de vrais dialogues.
- Les deux adaptateurs comparés diffèrent par deux variables à la fois, le nombre d'epochs et la version des données. L'écart observé combine les deux effets et ne permet pas de les séparer.
- Trois réponses sur vingt citent encore un titre absent de la liste de candidats. Le garde-fou Python détecte le titre inventé après coup, il ne l'empêche pas.
- Il n'existe pas d'axe de perspective : une demande comme « pas en 2D » ne peut pas être filtrée.
- Le relâchement des contraintes dures les retire toutes d'un coup, pas une par une.
- Le catalogue est figé à mars 2025 : les prix et la disponibilité ont pu changer depuis. L'agent le signale quand il cite un prix.
- L'évaluation porte sur vingt requêtes, ce qui suffit à observer des tendances mais non à conclure statistiquement.
