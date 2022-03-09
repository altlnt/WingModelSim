# identification_modele_avion
identification_modele_avion collab A. LETALENET M. ALLICHE
Dans cette collaboration, nous avons travaillé sur la modélisation, la simulation et l'identification d'un drone de type avion. Le projet se sépare en 2 parties, la simulation et l'identification du modèle. 


Simulation d'un drone de type avion : 
=
# Pour lancer une simulation en contrôle manuel: 
1. Lancer le script "Modelization/equation_generator.ipynb", et attendre que les fichiers de fonction soiten bien écrit. 
2. Brancher une manette et lancer le fichier "Simulation/Simulator_Class.py", la simulation va se lancer et va se dérouler en 3 temps : 
* Temps d'initialisation, les efforts sont nuls (par défault 1s dans la classe MoteurPhysique.py) 
* Le décollage, jusqu'à ce que le drone décolle les forces allant vers le sol sont bloqué
* Au moment du décollage la grille change de couleur et les forces s'appliquent normalement quelque soit la position du drone dans l'espace. 
* Vol en pilotage manuel avec la manette.
3. Les données du vols sont logger dans le dossier Logs/log_sim/AAAA_MM_DD_Hours avec la date du début de la simulation. Ce dossier contient 2 fichier, un "params.json" qui contient toutes les données physiques utilisées lors de la simulation. Et un "log.txt" qui contient les données de vol (acc,vitesse,position..)
4. Il est possible de tracer les données après la simulation avec le script "Logs/log_sim/plot.py", cela va tracer toutes les données écrites lors de la simualtion dans le fichier "log.txt"

Pour le contrôle en pilotage manuel : 
- joystick gauche horizontale = roll
- joystick gauche vertical = pitch
- joystick droite horizontale = yaw
- joystick drotie vericale = vitesse des moteurs

Pour ajuster le comportement, on peut modifier les gains des ailerons de controle (k0, k1, k2) par exemple.

# Lancer une optimisation des paramètres physiques: 
Il est possible de lancer des optimisations des paramètres avec des données simulées ou des données de vols réelles, il y seulement un preproccessing à effectuer sur les données réelles. 

1. Dans le cas de données de vols réelles utilisant PX4 :
* Extraire les données des fichiers .ulg avec le script : "Logs/Log_real/log_class.py" mettre dans la liste tous les fichier ulg à traiter. Pour chaque fichier ulg, un dossier sera créer avec les données extraites, et placera le fichier ulg à l'intérieur. 
* Utiliser le script /Modelization/concatenate_data.ipynb" pour concaténéer les données ensembles en un grand fichier "Logs/log_real/log_real.csv"
2. Pour l'optimisation il y a 2 possibilités : on peut soit utiliser "Optimization/Optimizer.py", ou "Optimization/Optimizer_parallele.py", les deux utilisent les mêmes algorithmes. La version parallele permet juste de lancer plusieurs optimisations avec différents réglages en même temps. 
3. Régler les valeurs initiales des données à optimiser, dans le cas de données simulées on prend les vraies valeurs qui seront randomizer. Pour le cas de données réelles, on peut les choisr dans l'optimizer.
4. Régler les différents gains du PID (par défault seul le gain proportionnel est mis), ainsi que le batch size, et le nombre d'epoch, et lancer l'optimisation. 2 fenêtres vont s'ouvrir, la premiere permet de voir les erreurs en % en fonction des epochs, et la secondes permet de voir l'évolution de la qualité de la simulation au fur et mesure que l'otpimisation avance. Pour cela elle relance une simulation avec les mêmes entrées, mais à chaque fois avec les paramètres courant, et cela permet de retracer la qualité de la simu. Ceci ne fonctionne qu'avec des données de vols simulées, avec des données réelles aucunes fenetre ne s'ouvre car la taille des fichiers est trop importante.
5. Les données utilisées lors de l'optimisation sont écrites dans un fichier portant le nom générique "OptiResults/Opti_#sim#real/identification_AAAA_MM_DD_Hours" avec la date et l'heure du début de l'optimisation.
6. Il est possible d'observer les résultats d'une opti à postériori, ou en durant son exécution avec le fichier "monitor_script.py", 


Liste des fichiers
=

equation_generator.ipynb = Génère les équations lambdifier qui seront utilisées dans le simulateurs et dans l'optimizer. Les fichiers écrits sont sauvegarder dans le dossier "Simulation", voici la liste des fonctions générées par ce script :

1. VelinLDPlane_function
2. dragDirection_function 
3. liftDirection_function
4. compute_alpha en fonction des VelinLDPlane, dragDirection, et liftDirection
5. Effort_Aero_complete_function = [Force, Couple] : renvoi un liste des efforts en fonction d'une liste de alpha ainsi que de la vitesse et orientation du drone, dans le repère Body
6. Grad_Effort_Aero_complete_function = renvoi le gradient des forces, dans le repère body
7. RMS_forces_function = Renvoi les erreurs au carrés des forces pour un jeu de données d'entrée, et des données de sorties normalisé
8. RMS_torque_function = Renvoi les erreurs au carrés des couoles pour un jeu de données d'entrée, et des données de sorties normalisé 
9. Cout_function = Calcul la fonction de cout, somme des RMS errors des couples et des forces normalisé. 
10. Grad_Cout_function = Calcul le gradient de la fonction de cout.
11. theta = List de clefs renvoyant toutes les données utilisé pour l'identification. Tout les calculs sont fait à partir de cette liste, il est important que les noms utilisés soit les mêmes que ceux du Moteur Physique.

Ces fonctions sont écrites suivant 4 fichiers :
* "/Simulation/function_moteur_physique" = [1:5] ces fonctions sont celles utilisées pour la simualations. 
* "/Simulation/function_moteur_physique+type_grad" (le terme "type_grad" prend différentes valeurs pour démarquer les différentes optimizations) qui prend toutes les fonctions précédentes mais adaptés suivant le type d'optimisation (avec estimation du vent, ou non par exemple)

MoteurPhysique_class.py = Dans le cas d'une simulation, exploite le fichier "/Simulation/function_moteur_physique" pour produire la simulation. Il log aussi les données de vols dans un fichier "log.txt", et il sauvegarde aussi dans un autre fichier ("params.txt") les grandeurs utilisées. Pour le cas d'une optimisation, la classe exploite les fonctions d'un des fichiers de fonctions pour calculer le gradient et le cout. 

Gui_class.py = Cette classe génère les fenêtres graphiques qui vont permettre d'observer le drone lors de la simulation en temps réel. De plus il gère aussi les entrées envoyées depuis le joystick vers le moteurs physique. 

Simulator_Class.py = Cette classe fait le lien entre le moteur physique et la classe GUI. Dans un premier temps, la classe récupère les entrées du joystick avec la classe GUI, qui sont ensuité utilisées dans le moteur physique pour générer la trajectoire, et ainsi alimenter le moteur graphique. 

Optimizer.py : Ce script permet de lancer les différentes optimisations, une fonction permet de préparer les données, en découpant et en mélangeant les datasets. Une fonction permet de calculer le gradient suivant différentes méthodes (calcul symbolique(+rapide) se trouvant dans le moteur physique, calcul numérique(-rapide)). Les résultats sont écrits dans un dossier en fonctions de la nature des données utilisées (réelles ou simulées) tel que "OptiResults/Opti_real/identification_2021_09_01_15h29m44s" par exemple.  5 fichier sont écrits : 
* "best_results.CSV" qui contient le meilleurs jeu de paramètres obtenu si le dernier jeu de données n'est pas le meilleur. 
* "Optimizer_script_used.py" il s'agit la version de l'optimizer utilisée. 
* "Params_intiaux.csv" donne les paramètres initiaux (vérité terrain dans le cas de données simulé, sinon paramètres servant à l'initialisation de l'optimisation). 
* Les fichiers "results.csv" et "results_opti.csv" donnent respectivement l'évolution des paramètres à optimizer en focntion des batch et des épochs. 


Optimizer_parallele.py : Permet de faire tourner plusieurs optimisations en même temps. le nom des dossier changement légèrement, la date est commune, et pour les différencier, le suffixe "numero_1" est ajouté par exemple.

monitor_script.py = Permet de tracer les résultats d'une opti en cours. Ile st possible de tracer plusieurs optimization, et/ou de choisir lesquelles on souhaite tracer. Cela va tracer toutes les données identifiées en fonction du nombre d'itération effectuées. Si souhaité, le script peut reconstruire les forces avec les jeux de paramètres initaux et finaux pour les comparer avec les données réelles.

		
