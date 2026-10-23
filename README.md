CONSEILLER DE JEUX STEAM
Agent conversationnel adapte par LoRA
 
Mini-projet NLP, CY Tech, ING2 GSI
Esteban Vela, Vincent Delb, Nicolao Hyacinthe, Rayane Saighi
23 octobre 2026
===============================================================================
 
 
1. CE QUE FAIT L'AGENT
-------------------------------------------------------------------------------
Un conseiller de jeux video qui interprete un gout exprime vaguement, pose une
question quand la demande ne permet pas de trancher, puis recommande des jeux
qui existent reellement en nommant le lien avec les gouts exprimes.
 
Principe : les FAITS viennent du catalogue, le COMPORTEMENT vient de
l'adaptation LoRA. Le modele ne connait aucun jeu de memoire. Il ne peut citer
que les candidats qu'un module Python lui fournit a chaque tour.
 
Traite   : recommander, explorer un gout flou, repondre sur un jeu du catalogue,
           comparer deux jeux, affiner apres un rejet, signaler qu'une licence
           citee est absente du catalogue.
Refuse   : support technique, remboursement, compte Steam, consoles, et tout
           fait absent du catalogue.
 
 
2. FICHIERS LIVRES
-------------------------------------------------------------------------------
agent.ipynb      le notebook complet, dix parties, executable de bout en bout
rapport.pdf      le rapport
README.txt       ce fichier
 
Le notebook ecrit lui-meme ses huit modules Python via des cellules %%writefile.
Il n'y a donc aucun fichier .py a fournir separement : un Restart & Run All les
recree integralement.
 
 
3. ENVIRONNEMENT
-------------------------------------------------------------------------------
Plateforme   Kaggle Notebook, accelerateur GPU T4
             Un seul GPU est rendu visible (CUDA_VISIBLE_DEVICES=0) : avec deux,
             le Trainer bascule en DataParallel alors que le modele est sur le
             GPU 0.
Python       3.13
Paquets      transformers, peft, datasets, accelerate, gradio
             installes par la premiere cellule du notebook
Modele       Qwen/Qwen2.5-1.5B-Instruct, charge en float16 depuis Hugging Face
             1,54 milliard de parametres, 3,09 Go en memoire GPU
Graine       GRAINE = 2026, posee sur random et torch
 
 
4. DONNEES A ATTACHER AVANT EXECUTION
-------------------------------------------------------------------------------
Panneau de droite du notebook, Add Input :
 
  a) Dataset   artermiloff/steam-games-dataset
     Fichier utilise : games_march2025_cleaned.csv
     89 618 lignes, 47 colonnes. Le chemin est detecte par glob, il n'y a aucune
     variable a modifier. Si le dataset n'est pas attache, la cellule de
     configuration s'arrete sur un assert qui indique quoi faire.
 
  b) Dataset   la sauvegarde contenant l'adaptateur e4
     Il s'agit de notre propre sauvegarde, produite lors du premier entrainement
     (4 epochs sur la premiere version des donnees). Elle sert de troisieme
     point de comparaison dans la partie 8.
 
     SANS CETTE SAUVEGARDE le notebook reste executable. Il faut alors :
       - commenter la cellule de reprise (celle qui definit CHEMIN_E4) ;
       - remplacer PeftModel.from_pretrained(...) par get_peft_model(mod,
         config_lora) dans la partie 7 ;
       - retirer la ligne ("e4", "default", True) de la liste CONDITIONS en
         partie 8 et l'entree correspondante du menu Gradio.
     L'evaluation compare alors deux conditions au lieu de trois.
 
Aucun jeton Hugging Face n'est necessaire : le modele est public. Si un secret
HF_TOKEN existe, il est utilise ; sinon le notebook affiche un avertissement et
continue normalement.
 
 
5. COMMENT EXECUTER
-------------------------------------------------------------------------------
Run All, de haut en bas. Duree indicative sur T4 : environ 50 minutes, dont
l'entrainement LoRA (1 epoch, 503 exemples) et l'evaluation des trois conditions
sur les vingt requetes de test.
 
Points d'attention :
 
  - Apres les huit cellules %%writefile se trouve une cellule de rechargement
    (importlib.reload). Toute modification d'un module exige de relancer cette
    cellule : %%writefile ecrit sur le disque, mais Python garde l'ancienne
    version du module en memoire.
 
  - L'adaptateur entraine est sauvegarde dans /kaggle/working/adaptateurs/e1.
    Il disparait avec la session : le telecharger avant de fermer.
 
  - L'interface Gradio de la derniere cellule est lancee avec
    prevent_thread_lock=True, sans quoi un Save & Run All ne se termine jamais.
 
  - L'etape d'analyse s'execute dans un bloc mod.disable_adapter(). Ce n'est pas
    un detail : l'adaptateur, entraine uniquement sur du dialogue, detruit la
    capacite du modele a produire du JSON. Voir la section 8 du rapport.
 
 
6. ORGANISATION DU NOTEBOOK
-------------------------------------------------------------------------------
Partie 1    Perimetre : ce que l'agent traite, ce qu'il refuse, cinq
            conversations types
Partie 2    Les huit modules (cellules %%writefile)
Partie 3    Construction du catalogue a partir du CSV Steam
Partie 4    Observation du modele : tokenisation, logits, temperature, top-k,
            top-p, entropie
Partie 5    Baseline par prompting et comparaison de deux messages systeme
Partie 6    Generation du jeu d'entrainement et quatre controles chiffres
Partie 7    Adaptation LoRA et courbe de perte
Partie 8    Evaluation comparative : recuperation seule, base, e4, e1
Partie 9    Interface Gradio avec bascule entre les trois modeles
Partie 10   Livrables
 
 
7. LES HUIT MODULES
-------------------------------------------------------------------------------
schemas.py     contrat partage : intentions, axes, plateformes, etats vides
axes.py        table tags Steam vers axes qualitatifs, et conversion
catalogue.py   lecture du CSV, chaine de filtres, fiches de jeu
modele.py      point d'entree unique pour la generation
prompts.py     messages systeme : analyse, et deux variantes de conseil
agent.py       analyser (modele vers JSON) et recuperer (Python vers candidats)
pipeline.py    enchainement des trois etapes et etat du dialogue
donnees.py     generation des exemples d'entrainement
 
 
8. FICHIERS PRODUITS PAR L'EXECUTION
-------------------------------------------------------------------------------
catalogue.parquet          14 657 jeux retenus, avec leurs axes
exemples_v2_train.jsonl    503 exemples d'entrainement
exemples_v2_val.jsonl      60 exemples de validation
exemples_v2_test.jsonl     60 exemples de test interne
adaptateurs/e1/            l'adaptateur LoRA entraine
resultats_base.json        sorties des vingt requetes, modele de base
resultats_e4.json          idem, adaptateur 4 epochs
resultats_e1.json          idem, adaptateur 1 epoch
 
 
9. RESULTATS PRINCIPAUX
-------------------------------------------------------------------------------
Catalogue           14 657 jeux apres filtrage (seuil de 200 avis)
                    98 % portent au moins deux axes qualitatifs
Donnees             640 exemples issus de 480 dialogues, 623 apres controles
                    454 debuts de reponse distincts, formule la plus frequente
                    a 4 %
LoRA                18 464 768 parametres entraines sur 1 580 643 840, soit
                    1,17 % du modele
                    perte de validation de 1,4434 a 0,8476 sur une epoch
 
Sur les vingt requetes de test                    base      e4      e1
  reponses citant un titre non fourni             5/20     5/20    3/20
  hors sujet : aucun jeu cite                      1/4      1/4     3/4
  intention bien routee (commune aux trois)      17/20    17/20   17/20
  JSON valide (commun aux trois)                 19/20    19/20   19/20
  longueur moyenne, en mots                         81       40      36
  accord avec le top-2 du score Python            53 %     67 %    44 %
 
Lecture : l'adaptation reduit l'invention et raccourcit les reponses. L'accord a
44 % signifie que le modele arbitre parmi les candidats au lieu de recopier
l'ordre du score Python.
 
 
10. LIMITES CONNUES
-------------------------------------------------------------------------------
  - Les exemples d'entrainement sont produits par gabarits : leur diversite de
    formulation reste inferieure a celle de vrais dialogues.
  - Les deux adaptateurs comparees different par deux variables a la fois, le
    nombre d'epochs et la version des donnees. L'ecart observe combine les deux
    effets et ne permet pas de les separer.
  - Trois reponses sur vingt citent encore un titre absent de la liste de
    candidats fournie au modele. Le garde-fou Python detecte le titre invente
    apres coup, il ne l'empeche pas.
  - Il n'existe pas d'axe de perspective : une demande comme "pas en 2D" ne peut
    pas etre filtree.
  - Le relachement des contraintes dures les retire toutes d'un coup, pas une
    par une.
  - Le catalogue est fige a mars 2025 : les prix et la disponibilite ont pu
    changer depuis. L'agent le signale quand il cite un prix.
  - L'evaluation porte sur vingt requetes, ce qui suffit a observer des
    tendances mais non a conclure statistiquement.
 
