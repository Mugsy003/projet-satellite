\documentclass{article}
\usepackage{graphicx} % Required for inserting images

\title{Rapport de stage}
\author{Youssef Zidan}
\date{August 2026}

\begin{document}

\maketitle

 \section{Introduction}

L'eau est à l'épicentre des bouleversements climatiques et environnementaux de notre siècle. Sous l'effet du réchauffement climatique global, mis en évidence par les rapports successifs du GIEC, le cycle hydrologique naturel subit des altérations profondes. Celles-ci se traduisent par une intensification des événements météorologiques extrêmes, une irrégularité croissante des précipitations et, inévitablement, une multiplication des épisodes de sécheresse sévère. Dans ce contexte de stress hydrique, la gestion de la ressource en eau douce est passée du statut d'enjeu purement économique à celui d'urgence vitale et stratégique.

Au cœur de cette problématique se trouve le secteur agricole. Aujourd'hui, l'agriculture concentre à elle seule près de 70 pourcent des prélèvements mondiaux en eau douce. Face à une pression démographique mondiale qui imposera de nourrir près de 10 milliards d'individus à l'horizon 2050, le secteur agricole se retrouve face à un paradoxe complexe : il doit produire davantage, tout en réduisant drastiquement son empreinte hydrique. L'irrigation traditionnelle, souvent basée sur des modèles empiriques ou sur une distribution systématique et peu ciblée de l'eau, n'est plus viable. Une véritable transition vers une "agriculture de précision" s'impose. Cette démarche vise à optimiser chaque goutte d'eau apportée aux cultures, en ne fournissant à la plante que la quantité strictement nécessaire à son développement, au moment exact où elle en a besoin.

Pour parvenir à un tel niveau de précision, il est indispensable de pouvoir quantifier avec exactitude la quantité d'eau que le système "sol-plante" consomme et renvoie vers l'atmosphère. Cette dynamique d'échange est capturée par une variable thermodynamique et climatique fondamentale : l'évapotranspiration (ET). L'évapotranspiration englobe deux phénomènes physiques distincts mais indissociables : d'une part, l'évaporation directe de l'eau contenue dans les couches superficielles du sol, et d'autre part, la transpiration des végétaux. À l'image de la transpiration humaine, la plante libère de la vapeur d'eau par ses stomates pour réguler sa température interne lorsqu'elle est soumise à un fort rayonnement solaire. Si la plante vient à manquer d'eau dans le sol, elle ferme ses stomates pour se protéger de la déshydratation. Sa température augmente alors rapidement : c'est le début du stress hydrique. L'évapotranspiration est donc le principal indicateur de la santé et du besoin en eau d'une culture.

Historiquement, la mesure de cette variable s'effectue in situ, directement sur le terrain. Les scientifiques s'appuient sur des infrastructures de pointe, telles que les tours à flux micrométéorologiques. Ces tours utilisent la méthode des covariances de turbulences (Eddy Covariance) pour mesurer les échanges de gaz et d'énergie entre la surface et l'atmosphère avec une très grande précision. Ces stations, déployées à travers des réseaux internationaux de recherche comme ICOS (Integrated Carbon Observation System), fournissent une vérité terrain inestimable. Cependant, elles se heurtent à une limite physique et logistique majeure : leur représentativité spatiale. L'empreinte de mesure (footprint) d'une tour à flux dépasse rarement quelques centaines de mètres carrés autour du capteur. Or, il est financièrement et matériellement impossible d'équiper chaque parcelle agricole, de chaque exploitation, avec une telle infrastructure.

Pour pallier ce manque critique d'informations à grande échelle, l'observation de la Terre par satellite s'est imposée, depuis plusieurs décennies, comme la seule solution viable pour spatialiser ces mesures. Les programmes spatiaux contemporains (tels que le programme européen Copernicus ou les missions de la NASA) scrutent la surface terrestre en continu, fournissant des pétaoctets de données multispectrales. En mesurant le rayonnement réfléchi par la végétation (dans le domaine optique et proche infrarouge) et l'énergie émise par la surface (dans le domaine de l'infrarouge thermique), les satellites offrent la promesse de pouvoir cartographier l'évapotranspiration à l'échelle du globe.

Cependant, l'estimation et la prédiction de l'évapotranspiration à l'échelle de la parcelle agricole se heurtent à un double verrou technologique que j'ai eu pour mission d'étudier et de lever au cours de mon stage.

Le premier verrou est spatial. Pour calculer l'évapotranspiration depuis l'espace, la mesure de la température de surface est indispensable. Or, les capteurs thermiques satellitaires présentent un compromis contraignant entre résolution spatiale et fréquence de revisite. Les satellites capables d'observer une même zone tous les jours offrent des pixels de l'ordre du kilomètre, fusionnant ainsi des dizaines de champs différents en une seule valeur moyenne. À l'inverse, les satellites offrant un niveau de détail suffisant (de l'ordre de la dizaine de mètres) ne survolent une même zone que tous les dix à quinze jours. Cette discontinuité rend les données brutes difficilement exploitables pour le suivi dynamique d'une parcelle agricole individuelle. Mon premier objectif a donc été d'étudier des méthodes de super-résolution spatiale pour contourner cette limite physique et recréer virtuellement des pixels à haute résolution.

Le second verrou est temporel. Si le diagnostic de l'évapotranspiration passée est essentiel pour le suivi agronomique et l'analyse climatique, la prise de décision proactive des agriculteurs nécessite d'anticiper l'avenir. Savoir qu'une parcelle a manqué d'eau hier est utile ; savoir qu'elle en manquera dans quatre jours permet d'agir. Cette nécessité de prévision (Forecasting) requiert l'intégration de prévisions météorologiques futures. Mon second objectif a consisté à concevoir une intelligence artificielle robuste, capable d'apprendre la thermodynamique complexe du système sol-plante-atmosphère pour prédire l'évapotranspiration sur un horizon opérationnel de plusieurs jours, et ce, malgré l'incertitude inhérente aux prévisions météorologiques.

Ce rapport de stage détaille l'ensemble de ce cheminement d'ingénierie, depuis le traitement de la donnée brute satellitaire jusqu'à l'optimisation d'un modèle de prédiction opérationnelle, fruit d'une collaboration entre la recherche académique et l'ingénierie industrielle en IA.

Dans un premier temps, le Chapitre 2 posera le cadre de mon environnement de travail en présentant les structures d'accueil du stage.
Le Chapitre 3 (État de l'art) établira les fondations théoriques du document, en détaillant les principes physiques de l'évapotranspiration et les concepts clés de l'apprentissage automatique appliqués aux séries temporelles.
Le Chapitre 4 abordera la dimension spatiale de mon travail, en détaillant les algorithmes de désagrégation thermique implémentés pour résoudre le problème d'échelle des satellites.
Le Chapitre 5 sera consacré à la dimension temporelle, avec la création et la validation d'une architecture de Deep Learning (LSTM) sur des données météorologiques parfaites, accompagnée d'une preuve d'explicabilité physique.
Enfin, le Chapitre 6 illustrera le passage à l'échelle opérationnelle : il détaillera les défis d'ingénierie (surapprentissage, effet de lissage, généralisation multi-sites) rencontrés lors de l'intégration de véritables prévisions météorologiques incertaines, et les stratégies algorithmiques mises en œuvre pour les surmonter.

\section{Présentation de l'entreprise d'accueil}
\label{sec:presentation_entreprise}

\subsection{Le Groupe Atos et ses enjeux stratégiques}

J'ai eu l'opportunité de réaliser ce stage au sein du groupe Atos, un acteur historique et incontournable de la transformation numérique à l'échelle européenne et internationale. Si l'entreprise est historiquement connue pour ses services informatiques classiques, elle s'est fortement repositionnée ces dernières années sur des secteurs à très haute valeur ajoutée technologique. Aujourd'hui, Atos se distingue par son expertise pointue dans le Cloud, la cybersécurité, et tout particulièrement dans le Calcul Haute Performance (HPC), domaine dans lequel le groupe figure parmi les leaders mondiaux grâce à la conception de supercalculateurs. 

Plus récemment, l'Intelligence Artificielle est venue naturellement s'intégrer à ce socle technique. À travers sa branche spécialisée dans le digital, le Big Data et la sécurité, Atos ne se limite plus à fournir de l'infrastructure logicielle : le groupe accompagne les industries, les agences spatiales et les institutions publiques dans la résolution de problématiques complexes, qu'elles soient d'ordre scientifique, logistique ou environnemental.

\begin{figure}[h]
    \centering
    % \includegraphics[width=0.5\textwidth]{images/logo_atos.png}
    \vspace{2cm} % Espace temporaire pour l'image
    \caption{Logo du groupe Atos.}
    \label{fig:logo_atos}
\end{figure}

Ce qui a particulièrement motivé mon choix d'intégrer ce groupe, c'est l'engagement qu'il affiche en matière de développement durable. À travers des initiatives ambitieuses visant la neutralité carbone (\textit{Net Zero}), Atos cherche non seulement à réduire son propre impact environnemental, mais investit également massivement en R\&D pour mettre ses capacités technologiques au service de la transition écologique (une démarche souvent résumée par le terme \textit{Tech for Good}). 

C'est très exactement à l'intersection de ces deux mondes — la puissance de calcul d'une part et le suivi du changement climatique de l'autre — que prend racine le contexte de mon stage. Aujourd'hui, l'observation de la Terre par satellite génère des volumes de données littéralement colossaux. Les programmes spatiaux, tels que le programme européen Copernicus, produisent quotidiennement des dizaines de téraoctets d'images. 

Exploiter ces archives spatiales sur le long terme pour en extraire des informations environnementales pertinentes (comme l'état hydrique des sols) nécessite des infrastructures de stockage robustes et une ingénierie logicielle que peu d'acteurs maîtrisent de bout en bout. C'est dans ce contexte de traitement massif de la donnée que le couplage entre l'imagerie satellitaire et l'apprentissage automatique devient un axe stratégique pour l'entreprise, ouvrant la voie à de nouveaux services pour l'agriculture de précision et la gestion des ressources naturelles.

\subsection{L'équipe d'accueil : L'Inno'lab de Toulouse et la collaboration ANITI}

J'ai été intégré(e) au sein du site d'Atos à Toulouse, au cœur de l'Inno'lab. Cette structure d'innovation a pour vocation d'explorer les technologies émergentes et de développer des solutions d'Intelligence Artificielle sur mesure pour répondre aux cas d'usage diversifiés des clients du groupe, tous secteurs d'activité confondus.

Si l'Inno'lab traite un large spectre de problématiques IA (traitement du langage naturel, vision par ordinateur, optimisation), mon stage s'inscrit dans un cadre de recherche très spécifique : une collaboration scientifique avec ANITI (\textit{Artificial and Natural Intelligence Toulouse Institute}). ANITI est l'un des instituts interdisciplinaires d'intelligence artificielle de pointe en France, favorisant les synergies entre la recherche académique et le monde industriel.

\begin{figure}[h]
    \centering
    % \includegraphics[width=0.8\textwidth]{images/organigramme.png}
    \vspace{4cm} % Espace temporaire pour l'image
    \caption{Positionnement du stage à la croisée de l'Inno'lab d'Atos et de la recherche ANITI.}
    \label{fig:organigramme}
\end{figure}

Au sein de ce dispositif, j'ai mené mes travaux de recherche avec une forte autonomie. J'ai eu la responsabilité de porter et de développer ce sujet pointu, centré sur la télédétection spatiale et l'évapotranspiration, au sein d'une équipe dont l'expertise première est l'ingénierie IA généraliste. 

Ce degré d'autonomie m'a permis de gérer l'intégralité du cycle de vie du projet de recherche : depuis la recherche bibliographique de l'état de l'art (algorithmes de désagrégation TsHARP/DMS) jusqu'à l'implémentation et l'optimisation des architectures de \textit{Deep Learning} (LSTM). L'enjeu de ma mission a donc été de démontrer comment les méthodologies avancées d'IA développées au sein de l'Inno'lab et soutenues par ANITI pouvaient être adaptées pour lever les verrous technologiques majeurs de l'observation de la Terre.

\section{Contexte et objectifs du stage}
\label{sec:contexte_objectifs}

\subsection{Contexte agro-climatique et enjeu de la quantification de l'évapotranspiration}

\subsubsection{La crise hydrique et la nécessité d'une gestion raisonnée}
Le secteur agricole mondial se trouve aujourd'hui au carrefour d'une double contrainte sans précédent~: répondre aux besoins alimentaires d'une population mondiale en croissance constante tout en composant avec une ressource en eau douce de plus en plus précaire. Sous l'effet des changements climatiques, la variabilité spatio-temporelle des précipitations s'accroît, allongeant la durée et l'intensité des périodes de sécheresse estivale. Face à ces tensions, l'irrigation traditionnelle, souvent gérée de manière empirique par des calendriers fixes ou des estimations volumétriques globales, atteint ses limites écologiques et économiques.

Pour maintenir les rendements agricoles tout en préservant les nappes phréatiques et les cours d'eau, l'agriculture doit impérativement s'orienter vers une gestion de précision. L'objectif fondamental de cette démarche est d'ajuster l'apport hydrique au plus près des besoins réels de la plante, au moment opportun et à la dose exacte. Atteindre cet objectif requiert une connaissance continue et localisée de la dynamique de l'eau au sein du continuum sol-plante-atmosphère.

\subsubsection{L'évapotranspiration : le pivot thermodynamique des flux d'eau}
L'évapotranspiration constitue la variable maîtresse de ce bilan hydrique. Elle représente la somme de deux processus physiques distincts~:
\begin{enumerate}
    \item \textbf{L'évaporation directe ($E$)~:} la vaporisation de l'eau présente dans les premiers centimètres du sol nu ou interceptée par le feuillage.
    \item \textbf{La transpiration végétale ($T$)~:} l'extraction de l'eau du sol par le système racinaire, son transport au sein de la plante, et sa libération sous forme de vapeur dans l'atmosphère à travers les stomates situés à la surface des feuilles.
\end{enumerate}

Au-delà d'un simple transfert de masse, la transpiration est un mécanisme d'autorégulation thermique vital. Lorsqu'elle dispose de suffisamment d'eau, la plante maintient ses stomates ouverts pour absorber le dioxyde de carbone nécessaire à la photosynthèse, évacuant l'excédent d'énergie radiative par chaleur latente de vaporisation. Dès lors que la réserve utile en eau du sol s'épuise, la plante réagit pour éviter le flétrissement en fermant progressivement ses stomates. Cette fermeture stomatique limite les pertes d'eau mais bloque la photosynthèse et supprime le refroidissement par transpiration~: la température foliaire s'élève alors rapidement. L'évapotranspiration réelle ($ET$) constitue ainsi l'indicateur le plus précoce et le plus direct du stress hydrique d'une culture, bien avant que les symptômes visuels de dépérissement ne deviennent perceptibles à l'œil nu.

\subsubsection{Les limites des méthodes de mesure in situ}
Sur le plan expérimental, la quantification directe de l'évapotranspiration s'appuie principalement sur la micrométéorologie et la méthode des covariances de turbulences (\textit{Eddy Covariance}). En mesurant à haute fréquence (typiquement 10 à 20~Hz) les fluctuations conjointes de la vitesse verticale du vent et de la concentration en vapeur d'eau de l'air, ces instruments permettent de déduire les flux de chaleur latente ($LE$) avec une très grande rigueur physique. Des réseaux internationaux de surveillance, à l'image d'ICOS (\textit{Integrated Carbon Observation System}) en Europe, ont déployé des dizaines de tours à flux instrumentées sur divers écosystèmes forestiers, prairiaux et agricoles.

Cependant, ces dispositifs \textit{in situ} présentent des contraintes structurelles majeures~:
\begin{itemize}
    \item \textbf{Une représentativité spatiale restreinte~:} L'empreinte spatiale de mesure (\textit{footprint}) d'une tour à flux n'excède généralement pas quelques centaines de mètres autour du mât, dépendant directement de la hauteur de mesure, de la rugosité de surface et de la direction du vent.
    \item \textbf{Un coût prohibitif~:} L'installation, l'étalonnage régulier et la maintenance de tels capteurs requièrent des budgets et des compétences techniques considérables, interdisant leur déploiement systématique à l'échelle de chaque exploitation agricole.
    \item \textbf{Une hétérogénéité non capturée~:} Une mesure ponctuelle ne peut refléter la variabilité spatiale intra-parcellaire induite par les différences de types de sol, de topographie, de variétés culturales ou de pratiques d'irrigation.
\end{itemize}

Face à cette limitation physique, le recours à la télédétection spatiale s'est imposé comme la seule alternative capable d'offrir une vision synoptique et spatialisée des états de surface. Toutefois, l'exploitation de l'imagerie satellitaire pour l'estimation de l'évapotranspiration soulève un double verrou technologique (spatial et temporel) qui a constitué le fil conducteur de l'ensemble de mes travaux de stage.

\subsection{Premier verrou : La dimension spatiale et la super-résolution thermique}

\subsubsection{Le compromis physique des capteurs satellitaires}
Pour déduire l'évapotranspiration par satellite via les modèles de bilan d'énergie de surface (tels que TSEB, SEBAL ou METRIC), l'information indispensable est la Température de Surface Terrestre (LST, \textit{Land Surface Temperature}), obtenue via les bandes spectrales de l'infrarouge thermique (TIR, 8--14~$\mu$m). C'est cette température radiométrique qui témoigne du refroidissement évaporatif de la végétation.

Or, les lois fondamentales de l'optique et du rayonnement (notamment la loi de Planck) imposent une contrainte instrumentale sévère~: l'énergie émise dans le domaine thermique est bien plus faible que l'énergie solaire réfléchie dans le spectre optique (visible et proche infrarouge). Pour collecter un rapport signal sur bruit acceptable sans élargir de façon démesurée l'optique des capteurs embarqués, les agences spatiales sont contraintes de concevoir des instruments dont le champ de vue instantané est large. 

Il en résulte un compromis technologique strict entre résolution spatiale et fréquence de revisite temporelle~:
\begin{itemize}
    \item \textbf{Les capteurs à haute fréquence temporelle (MODIS, Sentinel-3 SLSTR)~:} Ils offrent une revisite quotidienne indispensable pour suivre la dynamique rapide des flux hydriques, mais leur résolution spatiale thermique est grossière (de 500~m à 1~km). À cette échelle, un seul pixel englobe un mélange hétérogène de cultures différentes, de routes, de bosquets et de bâtis, rendant l'information inutilisable pour la gestion d'une parcelle agricole individuelle dont la taille moyenne en Europe est de quelques hectares.
    \item \textbf{Les capteurs à moyenne/haute résolution spatiale (Landsat TIRS)~:} Ils fournissent une résolution spatiale de 100~m (ré-échantillonnée à 30~m), bien plus compatible avec l'échelle parcellaire, mais leur période de revisite est de 8 à 16~jours. Cette périodicité est incompatible avec le pilotage de l'irrigation, d'autant plus que le passage d'un nuage peut masquer la surface et priver l'agriculteur de données pendant près d'un mois.
\end{itemize}

\subsubsection{Objectifs de la désagrégation thermique (Super-résolution)}
Mon premier objectif de recherche a donc consisté à développer, implémenter et évaluer des algorithmes de super-résolution spatiale --- couramment appelés méthodes de désagrégation ou de \textit{thermal sharpening}. Le principe de ces méthodes repose sur l'exploitation d'une relation d'échelle entre la température de surface (disponible à basse résolution spatiale) et des indices biophysiques issus du domaine optique et proche infrarouge (disponibles à haute résolution spatiale, typiquement 10 à 20~m avec Sentinel-2 ou 30~m avec Landsat), tels que le NDVI (\textit{Normalized Difference Vegetation Index}) ou la réflectance des bandes spectrales courtes.

J'ai structuré cette étude autour de deux familles d'approches algorithmiques distinctes~:
\begin{enumerate}
    \item \textbf{L'approche déterministe et semi-empirique (TsHARP)~:} Basée sur la formulation d'une relation statistique (généralement linéaire ou polynomiale) entre la LST et la fraction de couverture végétale (dérivée du NDVI), ajustée à l'échelle grossière puis appliquée aux pixels optiques fins pour redistribuer la température tout en conservant le bilan d'énergie global.
    \item \textbf{L'approche par apprentissage automatique (\textit{Data Mining Sharpening} - DMS)~:} Fondée sur des modèles non-linéaires multivariés (comme les forêts d'arbres décisionnels ou \textit{Random Forests}). Le DMS apprend à prédire la température radiométrique à partir d'un ensemble étendu de bandes spectrales optiques à basse résolution, puis applique cette fonction de régression complexe directement sur les bandes à haute résolution spatiale.
\end{enumerate}

\subsubsection{Une démarche méthodologique progressive}
Afin d'éprouver la robustesse de ces algorithmes face à des configurations satellitaires hétérogènes, j'ai conçu un protocole expérimental progressif en trois étapes d'ingénierie spatiale~:
\begin{itemize}
    \item \textbf{Étape 1 : Désagrégation intra-satellite (Landsat)~:} En guise de cas d'école et de point de référence, j'ai appliqué TsHARP et DMS sur les données de la constellation Landsat 8/9. Dans cette configuration, les bandes optiques (30~m) et la bande thermique (100~m) sont acquises simultanément par la même plateforme, sous des angles de visée et des conditions atmosphériques strictement identiques, ce qui élimine les biais géométriques et temporels.
    \item \textbf{Étape 2 : Fusion multi-satellites opérationnelle (Sentinel-3 + Sentinel-2)~:} J'ai ensuite abordé la complexité de la fusion de deux systèmes orbitaux distincts du programme Copernicus. L'objectif était de désagréger les flux thermiques quotidiens de Sentinel-3 SLSTR (pixel de 1~km) à l'échelle spatiale des bandes optiques de Sentinel-2 MSI (pixels de 10 à 20~m). Cette étape a nécessité la mise en place d'un pipeline complet de recalage géométrique, de rééchantillonnage et de synchronisation temporelle fine pour corriger le décalage horaire d'acquisition entre les deux satellites.
    \item \textbf{Étape 3 : Synergie avancée à haute résolution (ECOSTRESS + Sentinel-2)~:} Enfin, j'ai appliqué ces méthodes sur les données de la mission ECOSTRESS de la NASA, instrument thermique embarqué sur la Station Spatiale Internationale (ISS). ECOSTRESS présente une résolution thermique native plus fine ($\sim$70~m) et une orbite précessionnelle permettant des acquisitions à différentes heures de la journée. Le croisement de ces données thermiques riches avec la précision spectrale de Sentinel-2 a permis d'explorer les limites ultimes de la désagrégation spatiale sur des cycles diurnes spécifiques.
\end{itemize}

\subsection{Second verrou : La dimension temporelle et la prévision opérationnelle par Deep Learning}

\subsubsection{La nécessité de passer du diagnostic rétrospectif à l'anticipation}
Si la résolution du verrou spatial permet d'obtenir une cartographie rétrospective précise de l'évapotranspiration à l'échelle de la parcelle, elle ne résout que la moitié de la problématique agronomique. Un diagnostic de l'état hydrique passé (ce qui s'est produit hier ou les jours précédents) fournit un constat précieux, mais il s'avère insuffisant pour un pilotage proactif de l'exploitation.

En effet, les décisions culturales majeures --- telles que le déclenchement d'un cycle d'irrigation, la gestion de l'apport en fertilisants ou la planification des traitements phytosanitaires --- exigent un délai d'anticipation opérationnel. L'agriculteur doit pouvoir répondre à des questions concrètes~: \textit{Quelle sera la demande évaporative de ma parcelle dans les 3, 5 ou 7 prochains jours~? Un épisode de stress hydrique sévère va-t-il se déclencher compte tenu des conditions météorologiques annoncées~?}

Ce passage de l'estimation historique (\textit{hindcast}) à la prévision prospective (\textit{forecasting}) constitue le second verrou technologique majeur de mon stage.

\subsubsection{L'architecture Deep Learning retenue : LSTM Encodeur-Décodeur (Seq2Seq)}
Pour modéliser l'évolution temporelle non-linéaire de l'évapotranspiration, j'ai fait le choix de concevoir une architecture basée sur l'apprentissage profond (\textit{Deep Learning}), et plus particulièrement sur les réseaux de neurones récurrents de type LSTM (\textit{Long Short-Term Memory}). Les cellules LSTM possèdent des portes internes (\textit{gates}) qui leur permettent de réguler le flux d'information, de retenir des dépendances temporelles sur de longues séquences et d'éviter le problème de disparition du gradient (\textit{vanishing gradient}).

Pour formaliser le problème de la prévision multi-pas (de $J+1$ à $J+7$), j'ai implémenté une architecture de type Encodeur-Décodeur (\textit{Sequence-to-Sequence})~:
\begin{itemize}
    \item \textbf{L'Encodeur~:} Il ingère une séquence temporelle passée de forçages (données météorologiques historiques et indices de végétation dérivés de la télédétection, comme le NDVI ou le SAVI sur les 14 jours précédents). Son rôle est de comprimer l'historique et l'inertie hydrique du sol dans un vecteur d'état latent condensé.
    \item \textbf{Le Décodeur~:} Initialisé par l'état caché de l'encodeur, il reçoit en entrée les forçages météorologiques futurs prévus pour les 7 jours à venir (température de l'air, humidité relative, rayonnement net) et génère séquentiellement les prédictions d'évapotranspiration pour chaque pas de temps futur.
\end{itemize}

\subsubsection{Les deux phases du développement algorithmique}
Le développement et l'évaluation de cette chaîne de prévision par Deep Learning ont été articulés en deux phases scientifiques rigoureuses~:

\paragraph{Phase A : Preuve physique et validation en données parfaites (\textit{Hindcast})}
Dans un premier temps, afin de m'assurer de la validité conceptuelle de mon architecture, j'ai entraîné et évalué le modèle en lui fournissant des forçages météorologiques passés parfaits, issus des réanalyses atmosphériques globales ERA5 du CEPMMT (ECMWF). 

L'objectif de cette étape n'était pas seulement d'obtenir une faible erreur quadratique, mais de vérifier que le réseau de neurones n'agissait pas comme une simple ``boîte noire'' effectuant des corrélations trompeuses. Pour ce faire, j'ai couplé le modèle à des méthodes d'intelligence artificielle explicable (XAI) via la librairie \textbf{Captum} (notamment par la méthode des gradients intégrés). Cette analyse a permis de quantifier l'attribution de chaque variable d'entrée sur les sorties du modèle et de prouver mathématiquement que le réseau s'alignait sur les lois thermodynamiques de l'évapotranspiration~:
\begin{itemize}
    \item Une prédominance marquée accordée au rayonnement net ($R_n$), moteur premier du bilan d'énergie.
    \item Une sensibilité avérée à la dynamique de la végétation via les indices spectraux ($SAVI$, $NDVI$).
    \item L'absence de dépendance paresseuse à la seule persistance temporelle de l'évapotranspiration passée.
\end{itemize}

\paragraph{Phase B : Confrontation à la réalité opérationnelle (\textit{Forecasting}) et défis MLOps}
Une fois la cohérence physique démontrée en environnement contrôlé, j'ai confronté le modèle à son cas d'usage réel en remplaçant les réanalyses ERA5 par de véritables prévisions météorologiques numériques historiques (via l'API \textbf{Open-Meteo}, interrogeant les modèles de prévision opérationnels tels que GFS et ICON).

Cette transition du laboratoire vers l'environnement opérationnel a mis en évidence des problématiques concrètes de \textit{Machine Learning} appliqué~:
\begin{itemize}
    \item \textbf{L'incertitude croissante des forçages~:} Contrairement à ERA5, les prévisions météorologiques comportent un bruit et une marge d'erreur qui augmentent à mesure que l'horizon de prévision s'éloigne (de $J+1$ vers $J+7$).
    \item \textbf{Le phénomène de lissage statistique (\textit{Smoothing Effect})~:} Confronté à ces incertitudes d'entrée et guidé par une fonction de perte classique (l'erreur quadratique moyenne --- MSE), le modèle a initialement développé un comportement averse au risque, tendant à prédire une trajectoire moyenne quasi-plate et échouant à reproduire les variations brusques et les pics d'évapotranspiration.
    \item \textbf{La gestion du surapprentissage et la famine de données~:} Le modèle mémorisait le bruit des prévisions d'entraînement lorsqu'il était entraîné sur un échantillon temporel trop restreint (une seule année).
\end{itemize}

Pour surmonter ces obstacles, mon travail s'est orienté vers une phase intensive d'ingénierie et d'optimisation~:
\begin{enumerate}
    \item Mise en place de régularisations structurelles (ajustement du \textit{Dropout}, calibration fine du taux d'apprentissage et ajout de pénalités $L_2$ de type \textit{Weight Decay}).
    \item Instauration d'une stratégie stricte d'arrêt précoce (\textit{Early Stopping}) pour sauvegarder les poids optimaux du réseau avant la divergence des pertes de validation.
    \item Élargissement de la base d'apprentissage par la constitution d'un jeu de données multi-sites et multi-années (combinant plusieurs stations de mesure à travers l'Europe sur les années 2022 et 2023), tout en conservant l'année 2024 comme jeu de test strictement aveugle afin d'évaluer la robustesse de la généralisation spatio-temporelle.
\end{enumerate}

\subsection{Synthèse des objectifs poursuivis}

En résumé, mon stage s'est articulé autour d'un continuum méthodologique structuré visant à transformer la donnée brute de télédétection et de météorologie en un outil d'aide à la décision agronomique prédictif et robuste. 

Les objectifs poursuivis peuvent être synthétisés ainsi~:
\begin{enumerate}
    \item \textbf{Évaluer et calibrer les méthodes de désagrégation spatiale thermique (TsHARP et DMS)} sur des données satellitaires de complexité croissante (Landsat, puis les synergies Sentinel-3/Sentinel-2 et ECOSTRESS/Sentinel-2) pour générer des estimations d'évapotranspiration à haute résolution spatiale.
    \item \textbf{Développer une architecture Deep Learning Seq2Seq LSTM} capable d'ingérer ces séries temporelles hétérogènes et de modéliser les flux d'évapotranspiration sur un horizon multi-pas ($J+1$ à $J+7$).
    \item \textbf{Auditer la cohérence physique du réseau de neurones} au moyen d'outils d'attribution et d'explicabilité (Captum), garantissant le respect des principes thermodynamiques du transfert d'énergie.
    \item \textbf{Adapter le modèle aux contraintes du déploiement opérationnel} en intégrant de vraies prévisions météorologiques bruitées, en neutralisant les effets de lissage statistique et en validant sa capacité de généralisation spatio-temporelle sur des jeux de données multi-sites indépendants.
\end{enumerate}

\section{État de l'art et fondements théoriques}
\label{sec:etat_art}

Ce chapitre présente le cadre théorique sur lequel reposent l'ensemble de mes travaux. Il s'articule autour de trois piliers scientifiques~: la thermodynamique des transferts d'eau (modélisation de l'évapotranspiration), la physique de la télédétection spatiale thermique, et enfin les fondements mathématiques de l'apprentissage profond appliqué aux séries temporelles.

\subsection{Modélisation physique de l'évapotranspiration}

\subsubsection{Le bilan d'énergie de surface}
L'évapotranspiration n'est pas qu'un simple transfert de masse~; c'est avant tout un transfert d'énergie. À la surface terrestre, l'énergie provenant du Soleil doit être conservée. Ce principe fondamental se traduit par l'équation du bilan d'énergie de surface~:
\begin{equation}
    R_n - G = H + LE
    \label{eq:bilan_energie}
\end{equation}
où~:
\begin{itemize}
    \item $R_n$ est le rayonnement net (W$\cdot$m$^{-2}$), représentant l'énergie totale absorbée par la surface (différence entre les rayonnements incidents et réfléchis/émis).
    \item $G$ est le flux de chaleur dans le sol (W$\cdot$m$^{-2}$), c'est-à-dire l'énergie absorbée par conduction dans le sol.
    \item $H$ est le flux de chaleur sensible (W$\cdot$m$^{-2}$), qui réchauffe l'air environnant par convection.
    \item $LE$ est le flux de chaleur latente (W$\cdot$m$^{-2}$), qui correspond à l'énergie consommée pour vaporiser l'eau. C'est l'équivalent énergétique de l'évapotranspiration.
\end{itemize}

L'enjeu de l'agronomie spatiale est de parvenir à estimer la fraction d'énergie allouée au terme $LE$. Plus une plante transpire, plus $LE$ augmente, et mécaniquement, moins il reste d'énergie pour $H$~: la surface se refroidit.

\begin{figure}[htbp]
    \centering
    \vspace{4cm} % Espace pour insérer ton schéma du bilan d'énergie
    % \includegraphics[width=0.7\textwidth]{images/bilan_energie.png}
    \caption{Représentation schématique du bilan d'énergie à la surface du continuum sol-plante.}
    \label{fig:bilan_energie}
\end{figure}

\subsubsection{L'équation de Penman-Monteith}
Pour relier ces flux d'énergie aux variables météorologiques et physiologiques, la communauté scientifique s'appuie sur le modèle de Penman-Monteith (1965). Cette équation physique complexe modélise l'évapotranspiration en considérant la végétation comme une "grosse feuille" possédant une résistance aérodynamique et une résistance de surface (les stomates)~:
\begin{equation}
    LE = \frac{\Delta (R_n - G) + \rho_a c_p \frac{(e_s - e_a)}{r_a}}{\Delta + \gamma \left(1 + \frac{r_s}{r_a}\right)}
    \label{eq:penman_monteith}
\end{equation}
où $\Delta$ est la pente de la courbe de pression de vapeur saturante, $\rho_a$ la densité de l'air, $c_p$ la chaleur massique de l'air, $(e_s - e_a)$ le déficit de pression de vapeur (VPD), $r_a$ la résistance aérodynamique, $\gamma$ la constante psychrométrique, et $r_s$ la résistance stomatique.

\subsubsection{L'approche de Priestley-Taylor (Cible PT-SINRH)}
Si le modèle de Penman-Monteith est exhaustif, il requiert des variables souvent inaccessibles par satellite (comme la vitesse du vent ou la résistance stomatique exacte). C'est pourquoi, dans mes modélisations (et notamment pour la construction de ma variable cible \textit{PT-SINRH}), je me base sur des formulations dérivées et simplifiées, comme l'équation de Priestley-Taylor (1972). Celle-ci s'affranchit du terme aérodynamique en introduisant un coefficient empirique $\alpha$~:
\begin{equation}
    LE = \alpha \frac{\Delta}{\Delta + \gamma} (R_n - G)
\end{equation}
Cette approche physique guidée par le rayonnement ($R_n$) explique pourquoi cette variable a été identifiée comme prépondérante lors de mon analyse d'explicabilité par intelligence artificielle.

\subsection{Télédétection thermique et Super-résolution}

\subsubsection{Principes physiques de l'infrarouge thermique}
L'observation de la Terre repose sur la détection des rayonnements électromagnétiques. La Température de Surface Terrestre (LST) est dérivée des mesures dans l'infrarouge thermique (8-14 $\mu$m). Selon la loi de Stefan-Boltzmann, l'exitance énergétique $M$ d'une surface est proportionnelle à la puissance quatrième de sa température absolue $T$~:
\begin{equation}
    M = \varepsilon \sigma T^4
\end{equation}
où $\varepsilon$ est l'émissivité de surface et $\sigma$ la constante de Stefan-Boltzmann. En inversant cette relation (et la loi de Planck) à partir de la radiance mesurée au sommet de l'atmosphère par un satellite, on déduit la température de surface.

\subsubsection{Les missions satellitaires exploitées}
Au cours de ce stage, j'ai manipulé des données issues de plusieurs missions, chacune présentant des compromis spécifiques~:
\begin{itemize}
    \item \textbf{Landsat 8/9 (NASA/USGS)~:} Équipés de l'instrument TIRS, ils fournissent une LST à 100~m de résolution, ré-échantillonnée à 30~m, mais avec une revisite de 8 jours (en combinant les deux satellites).
    \item \textbf{Sentinel-3 (ESA/Copernicus)~:} L'instrument SLSTR offre une excellente revisite (quasi-quotidienne) mais une résolution thermique spatiale très grossière de 1~km.
    \item \textbf{ECOSTRESS (NASA)~:} Embarqué sur l'ISS, ce radiomètre multispectral innovant offre une résolution native d'environ 70~m et permet d'observer la Terre à différentes heures de la journée, capturant ainsi le cycle diurne de la chaleur.
\end{itemize}

\subsubsection{Algorithmes de désagrégation spatiale (TsHARP et DMS)}
Pour pallier la faible résolution thermique des satellites à haute fréquence, des techniques de désagrégation sont employées. Le principe général est d'établir une fonction $f$ liant la LST à basse résolution ($LST_{BR}$) à des indices spectraux $I$ disponibles à haute résolution ($I_{HR}$), comme le NDVI.

La méthode \textbf{TsHARP} postule une relation linéaire classique. Une fois cette relation ajustée à basse résolution, elle est appliquée aux pixels haute résolution, avec l'ajout d'un terme résiduel $\Delta T$ pour assurer la conservation de l'énergie~:
\begin{equation}
    \widehat{LST}_{HR} = f_{TsHARP}(NDVI_{HR}) + \Delta T
\end{equation}

La méthode \textbf{DMS} (\textit{Data Mining Sharpening}) remplace cette régression linéaire simple par des algorithmes de Machine Learning (tels que des \textit{Random Forests}). Le DMS permet d'ingérer plusieurs dizaines de bandes spectrales (réflectances optiques) pour cartographier de façon non-linéaire la complexité du paysage thermique, offrant des résultats nettement supérieurs dans les zones agricoles hétérogènes.

\begin{figure}[htbp]
    \centering
    \vspace{5cm} % Espace pour un schéma illustrant la désagrégation d'un gros pixel en sous-pixels
    % \includegraphics[width=0.8\textwidth]{images/schema_tsharp.png}
    \caption{Processus de désagrégation thermique : passage d'un pixel kilométrique à une résolution parcellaire via la fusion optique/thermique.}
    \label{fig:schema_desagregation}
\end{figure}

\subsection{Apprentissage Profond pour les séries temporelles}

\subsubsection{Des RNN aux LSTM (Long Short-Term Memory)}
La prévision de l'évapotranspiration nécessite d'analyser des séquences météorologiques. Les Réseaux de Neurones Récurrents (RNN) classiques sont conçus pour cela, mais souffrent du problème de disparition du gradient (\textit{Vanishing Gradient}), les empêchant de retenir des informations sur de longues périodes (comme l'historique d'une sécheresse).

Pour surmonter ce problème, j'ai utilisé des cellules LSTM (Hochreiter \& Schmidhuber, 1997). L'architecture interne d'une cellule LSTM s'articule autour de trois portes (\textit{gates}) qui contrôlent le flux d'information de l'état de la cellule $C_t$ et de l'état caché $h_t$ au temps $t$~:

\begin{align}
    f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad &\text{(Porte d'oubli)} \label{eq:lstm_f}\\
    i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad &\text{(Porte d'entrée)} \label{eq:lstm_i}\\
    \tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad &\text{(Candidat mémoire)} \label{eq:lstm_c_tilde}\\
    C_t &= f_t \ast C_{t-1} + i_t \ast \tilde{C}_t \quad &\text{(Mise à jour de la cellule)} \label{eq:lstm_c}\\
    o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad &\text{(Porte de sortie)} \label{eq:lstm_o}\\
    h_t &= o_t \ast \tanh(C_t) \quad &\text{(État caché final)} \label{eq:lstm_h}
\end{align}
où $\sigma$ est la fonction d'activation sigmoïde, $W$ les matrices de poids, $b$ les biais, et $\ast$ le produit matriciel élément par élément (Hadamard). 

\begin{figure}[htbp]
    \centering
    \vspace{4.5cm} % Espace pour le schéma d'une cellule LSTM
    % \includegraphics[width=0.6\textwidth]{images/lstm_cell.png}
    \caption{Architecture interne d'une cellule LSTM illustrant le mécanisme des portes.}
    \label{fig:lstm_cell}
\end{figure}

\subsubsection{Architecture Encodeur-Décodeur (Seq2Seq)}
Pour réaliser des prévisions opérationnelles (de $J+1$ à $J+7$), j'ai implémenté une structure Sequence-to-Sequence (Seq2Seq).
L'\textbf{Encodeur} traite les données météorologiques et satellitaires passées. À la fin de la séquence historique, son état caché final $h_N$ contient une représentation compacte (le contexte) de la dynamique hydrique du sol.
Le \textbf{Décodeur} utilise ce contexte $h_N$ comme point de départ. Il prend ensuite en entrée les prévisions météorologiques futures pour générer séquentiellement les prédictions d'évapotranspiration pas à pas.

\subsection{Intelligence Artificielle Explicable (XAI)}

L'adoption de modèles de \textit{Deep Learning} dans l'industrie et la recherche environnementale se heurte souvent à l'opacité de ces algorithmes (effet "boîte noire"). Il est essentiel de s'assurer que le modèle apprend la physique sous-jacente et non des artefacts statistiques biaisés.

Dans le cadre de mon stage, j'ai eu recours au domaine de l'IA explicable (XAI), et plus spécifiquement à la méthode des \textbf{Gradients Intégrés} implémentée via la librairie PyTorch Captum. 
Contrairement à une simple analyse d'importance par permutation, les gradients intégrés calculent l'intégrale des gradients de la prédiction du modèle par rapport à ses entrées, le long d'un chemin allant d'une image de référence (un tenseur vide ou neutre) jusqu'à l'entrée réelle. Mathématiquement, l'attribution $IG_i$ pour la variable d'entrée $x_i$ se définit par~:
\begin{equation}
    IG_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha \times (x - x'))}{\partial x_i} d\alpha
\end{equation}
Cette méthode satisfait les axiomes mathématiques de sensibilité et de préservation de l'implémentation, garantissant une attribution rigoureuse du rôle thermodynamique de chaque forçage (comme $R_n$ ou le NDVI) dans mes prédictions.
% N'oublie pas d'ajouter \usepackage{listings} et \usepackage{xcolor} dans ton préambule pour les blocs de code.

\section{Contribution I : Architecture logicielle et Extraction Multi-sources}
\label{sec:contrib_archi}

Dans le cadre de l'étude et du traitement des données géospatiales pour l'estimation de l'évapotranspiration et des températures de surface, la première étape cruciale fut de concevoir une architecture logicielle robuste[cite: 1]. Le besoin de manipuler des volumes massifs de données (plus de 50 Go à terme), provenant de sources hétérogènes (satellites, tours à flux, forçages météorologiques), imposait de dépasser le stade de scripts isolés pour construire un pipeline unifié, automatisé et paramétrable[cite: 1]. Ce chapitre détaille la conception de cette architecture et les solutions apportées pour l'acquisition et le couplage de ces données complexes[cite: 1].

\subsection{Conception de l'architecture du projet : Orchestration et modularité}

Au début de ce stage, la manipulation des données satellitaires impliquait des actions manuelles fastidieuses : téléchargement depuis différentes plateformes web, traitements locaux morcelés, et scripts d'analyse indépendants[cite: 1]. L'une de mes premières contributions majeures a consisté à unifier ce processus au sein d'un pipeline Python modulaire et centralisé[cite: 1].

\subsubsection{Le patron de conception "Pipeline" et l'orchestrateur \texttt{main.py}}

Pour structurer le projet, j'ai opté pour une architecture en pipeline, où la sortie d'une étape devient l'entrée de la suivante[cite: 1]. Ce pipeline se divise en cinq grandes phases : l'extraction (téléchargement des données), la transformation (calcul des indices et des rasters), le Machine Learning (Downscaling/Sharpening), la modélisation (TTME/PT-SINRH) et enfin, l'analyse/visualisation[cite: 1].

Le point névralgique de ce système est le script \texttt{main.py}[cite: 1]. Il agit comme un chef d'orchestre qui exécute séquentiellement les différents modules[cite: 1]. Plutôt que de lancer manuellement une dizaine de scripts, l'utilisateur n'a plus qu'à invoquer la commande standard. L'orchestrateur s'occupe de gérer le flux d'exécution, d'intercepter les erreurs potentielles (via des blocs \texttt{try/except} globaux pour éviter que l'échec d'une image ne fasse crasher tout le processus), et de chronométrer chaque étape pour des besoins d'optimisation de performance[cite: 1].

\begin{figure}[htbp]
    \centering
    \vspace{6cm} % Espace pour un diagramme UML ou un schéma d'architecture
    \caption{Architecture générale du pipeline de données : de l'ingestion brute à la modélisation de l'évapotranspiration.}
    \label{fig:pipeline_archi}
\end{figure}

\subsubsection{Une configuration flexible et centralisée (\texttt{config.py})}

La flexibilité était un prérequis essentiel, car selon les jours, le chercheur ou l'utilisateur peut vouloir ne traiter que les données d'un satellite spécifique, sur un site précis, ou tester un nouvel algorithme sans relancer l'extraction complète (très coûteuse en temps et en bande passante)[cite: 1].

J'ai ainsi mis en place un fichier \texttt{config.py} qui agit comme la tour de contrôle du projet[cite: 1]. Ce fichier contient notamment :
\begin{itemize}
    \item \textbf{Le dictionnaire \texttt{PIPELINE\_STEPS}} : Une structure de données booléenne permettant d'activer ou de désactiver chaque brique du projet à la volée (ex: \texttt{"extraction\_landsat": True}, \texttt{"sharpening\_dms\_sentinel3": False})[cite: 1].
    \item \textbf{Les dictionnaires de sites (\texttt{SITES\_PILOTES})} : Un référentiel des coordonnées géographiques (longitude, latitude) des sites expérimentaux (comme Gebesee, Selhausen, ou Lamasquere) couplé à leurs identifiants de tours à flux ICOS (\texttt{PIDS\_ICOS})[cite: 1].
    \item \textbf{Les fenêtres spatio-temporelles} : Des variables comme \texttt{TIME\_OF\_INTEREST} ou \texttt{radius\_km} qui définissent dynamiquement le champ de recherche pour les APIs, évitant ainsi d'avoir des paramètres codés en dur (\textit{hardcodés}) dispersés dans les sous-modules[cite: 1].
\end{itemize}

Cette séparation stricte entre la logique métier (\texttt{main.py} et sous-dossiers) et la configuration (\texttt{config.py}) garantit que le code source n'a pas besoin d'être modifié lors de chaque nouvelle expérimentation[cite: 1].

\subsection{Automatisation de l'acquisition des données satellitaires hétérogènes}

Le deuxième grand défi technique consistait à interagir informatiquement avec les gigantesques bases de données spatiales mondiales[cite: 1]. Le projet repose sur des données de résolutions, d'orbites et d'opérateurs très différents[cite: 1] :
\begin{itemize}
    \item \textbf{Landsat 8/9 (USGS/NASA)} : Haute résolution (30~m / 100~m) mais faible revisite (16 jours)[cite: 1].
    \item \textbf{Sentinel-2 (ESA/Copernicus)} : Très haute résolution optique (10~m) avec revisite de 5 jours[cite: 1].
    \item \textbf{Sentinel-3 (ESA/Copernicus)} : Basse résolution (1~km thermique) mais revisite quotidienne[cite: 1].
    \item \textbf{ECOSTRESS (NASA ISS)} : Résolution thermique très fine (70~m) mais orbite non héliosynchrone[cite: 1].
\end{itemize}

\subsubsection{L'utilisation du standard STAC (\textit{SpatioTemporal Asset Catalog})}

Pour automatiser la recherche et le téléchargement, je me suis massivement appuyé sur le protocole \textbf{STAC}[cite: 1]. STAC est un standard émergent qui permet d'interroger des catalogues d'images satellites via des requêtes HTTP (API REST)[cite: 1].

Plutôt que de naviguer sur un portail web et de tracer des polygones manuellement, le module d'extraction que j'ai développé construit dynamiquement des requêtes JSON contenant[cite: 1] :
\begin{enumerate}
    \item Le polygone géographique (\texttt{GeoJSON} ou \textit{Bounding Box}) centré sur les sites définis dans \texttt{config.py}[cite: 1].
    \item La fenêtre temporelle d'intérêt[cite: 1].
    \item Le seuil de couverture nuageuse maximum toléré[cite: 1].
\end{enumerate}

Ce système permet au code de filtrer automatiquement des centaines d'archives pour ne conserver que les "scènes" (tuiles) pertinentes[cite: 1]. Une fois la liste des scènes obtenue, le pipeline interroge les \textit{Assets} (les liens de téléchargement direct vers les bandes spectrales spécifiques) pour ne télécharger que ce qui est strictement nécessaire, économisant ainsi drastiquement la bande passante et le stockage local[cite: 1].

\subsubsection{L'intégration des données de référence (Réseaux ICOS et Météo)}

Parallèlement aux images satellites, la modélisation de l'évapotranspiration et la validation de nos algorithmes nécessitaient des données de "vérité terrain" et de forçage météorologique[cite: 1]. J'ai donc implémenté des scripts d'extraction spécifiques pour[cite: 1] :
\begin{itemize}
    \item \textbf{Le réseau ICOS (\textit{Integrated Carbon Observation System})} : Développement d'un script capable de s'authentifier et de télécharger les séries temporelles de chaleur latente ($LE$) et de chaleur sensible ($H$) directement depuis les PIDs des stations européennes[cite: 1].
    \item \textbf{Les données ERA5 (\textit{Copernicus Climate Change Service})} : Utilisation de l'API \textit{Copernicus Climate Data Store (CDS)} pour télécharger les réanalyses horaires (température de l'air, vitesse du vent, humidité) nécessaires au fonctionnement des modèles TTME et PT-SINRH[cite: 1].
    \item \textbf{Open-Meteo} : Intégration d'une API secondaire pour des données météorologiques \textit{in situ} rapides[cite: 1].
\end{itemize}

Cette phase d'extraction aboutit à la création d'un "Data Lake" local, organisé rigoureusement dans une arborescence de dossiers par satellite et par site, prêt à être ingéré par l'étape de transformation[cite: 1].

\begin{figure}[htbp]
    \centering
    \vspace{4cm} % Espace pour illustrer l'arborescence du Data Lake local
    \caption{Structure de l'arborescence du Data Lake local généré par le module d'extraction.}
    \label{fig:datalake_tree}
\end{figure}

\subsection{Le défi du couplage de données : Synergies spatio-temporelles}

Si télécharger des données est complexe, les croiser l'est encore davantage[cite: 1]. Le projet repose sur le "Sharpening" (amélioration de résolution), qui nécessite de fusionner la température (LST) d'un capteur thermique à basse résolution avec la réflectance (NDVI) d'un capteur optique à haute résolution[cite: 1].

C'est ici qu'intervient le défi de la "synergie spatio-temporelle" : comment s'assurer qu'une image thermique et une image optique, prises par deux satellites distincts (avec des orbites différentes), correspondent au même état physique de la Terre[cite: 1] ?

\subsubsection{Les paires Sentinel-3 / Sentinel-2 et ECOSTRESS / Sentinel-2}

Pour créer des produits thermiques à très haute résolution temporelle et spatiale, l'objectif était de coupler le capteur thermique SLSTR de Sentinel-3 (basse résolution, passage journalier) avec le capteur optique MSI de Sentinel-2 (haute résolution, passage tous les 5 jours)[cite: 1]. De manière similaire, le capteur thermique ECOSTRESS (monté sur l'ISS) devait être couplé avec Sentinel-2[cite: 1].

Le problème physique majeur est que la température de surface (LST) évolue de manière extrêmement rapide au cours de la journée en fonction du rayonnement solaire, de l'humidité et de l'évapotranspiration, contrairement à la réflectance optique (végétation) qui évolue sur plusieurs jours (phénologie)[cite: 1]. Il était donc impossible d'utiliser une image optique d'un jour $J$ avec une image thermique du jour $J+2$ sans précaution mathématique[cite: 1].

\subsubsection{Algorithmique de couplage temporel par fenêtre glissante}

Pour résoudre ce problème, j'ai développé un algorithme de recherche de "paires parfaites"[cite: 1]. Lors de l'extraction, le système procède de la manière suivante[cite: 1] :
\begin{enumerate}
    \item Il télécharge ou identifie toutes les scènes disponibles pour le capteur thermique (ex: ECOSTRESS) sur une période d'un an pour un site donné[cite: 1].
    \item Pour chaque scène thermique, il extrait le \textit{timestamp} exact d'acquisition $t_{thermique}$[cite: 1].
    \item Il interroge ensuite le catalogue du capteur optique (Sentinel-2) en appliquant une contrainte stricte définie dans \texttt{config.py} via la variable \texttt{TIME\_MARGIN\_MINUTES} (généralement réglée sur $\pm 60$ minutes)[cite: 1].
\end{enumerate}

Mathématiquement, le couplage n'est validé que si une acquisition optique $t_{optique}$ satisfait la condition suivante :
$$ |t_{thermique} - t_{optique}| \le \Delta t_{max} $$
où $\Delta t_{max}$ représente la marge temporelle maximale autorisée. Si (et seulement si) une image Sentinel-2 non-nuageuse existe dans ce petit intervalle de temps, le système valide la "paire" et lance les téléchargements croisés[cite: 1]. Cette approche est essentielle pour garantir que la relation mathématique apprise plus tard par le Machine Learning (le \textit{Data Mining Sharpening}) relie bien une végétation et une température mesurées sous les mêmes conditions radiatives et de stress hydrique[cite: 1].

\subsubsection{Alignement spatial et ré-échantillonnage (Reprojection)}

Une fois le couplage temporel assuré, se pose le problème du couplage spatial[cite: 1]. Les différents satellites ne partagent ni le même système de coordonnées (CRS - \textit{Coordinate Reference System}), ni la même résolution, ni la même grille de pixels (\textit{Grid})[cite: 1].

Afin que le pixel $(x, y)$ de l'image Sentinel-2 corresponde exactement au pixel $(x, y)$ de l'image ECOSTRESS ou Sentinel-3 superposée, j'ai implémenté dans le module de transformation des fonctions basées sur les librairies \texttt{rasterio} et \texttt{gdal}[cite: 1]. Le processus automatisé effectue[cite: 1] :
\begin{itemize}
    \item \textbf{La Reprojection} : Conversion de toutes les images vers un CRS de référence commun (ex: UTM WGS84 correspondant à la zone du site)[cite: 1].
    \item \textbf{Le Cropping (Découpage)} : Découpage strict à la \textit{bounding box} du site d'étude pour réduire le volume de calculs et éviter l'effet de bordure[cite: 1].
    \item \textbf{Le Ré-échantillonnage (Resampling)} : Utilisation d'algorithmes d'interpolation pertinents physiquement (ex: bilinéaire pour les températures continues, \textit{Nearest Neighbor} pour les masques discrets de classification de nuages) afin d'aligner les grilles de basse résolution (ex: 1~km) sur les grilles de haute résolution (ex: 10~m ou 30~m)[cite: 1].
\end{itemize}

Cette étape d'alignement est l'aboutissement du module d'architecture et d'extraction[cite: 1]. Elle génère les cubes de données multi-dimensionnels (hypercubes de tenseurs) qui seront directement injectés dans les algorithmes de Machine Learning (Random Forest / LightGBM) dans le chapitre suivant[cite: 1].
\section{Contribution II : Amélioration de la résolution spatiale par Machine Learning (Sharpening Thermique)}
\label{sec:contrib_sharpening}

La température de surface (LST) est une variable géophysique clé dont la mesure spatiale souffre d'un compromis technique inévitable entre la résolution spatiale et la fréquence de revisite. Par exemple, le capteur SLSTR de Sentinel-3 offre une revisite quotidienne, vitale pour le suivi du stress hydrique, mais avec une résolution kilométrique (1~km), largement insuffisante pour l'agriculture de précision. À l'inverse, la constellation Sentinel-2 offre une résolution décamétrique (10-20~m) mais ne possède pas d'instruments thermiques. 

L'objectif de cette deuxième contribution majeure a été de développer, d'implémenter et d'optimiser une méthode de descente d'échelle spatiale (\textit{Downscaling}), plus précisément le \textit{Data Mining Sharpening} (DMS), visant à fusionner la haute résolution spatiale des capteurs optiques avec la haute résolution temporelle des capteurs thermiques.

\subsection{Méthodologie : Préparation des variables prédictives (Feature Engineering)}

La désagrégation thermique repose sur un principe biophysique fondamental : la température d'un pixel est intimement corrélée à l'état de la surface qu'il représente (végétation, humidité, topographie, albédo). L'approche TsHARP classique postule une simple relation linéaire entre la LST et l'indice de végétation NDVI. Cependant, mes premières analyses ont confirmé que cette relation linéaire s'effondre dans les milieux hétérogènes (zones urbaines, sols nus, lisières, cours d'eau). 

Pour alimenter des modèles de Machine Learning capables d'apprendre des relations non-linéaires complexes, j'ai dû construire un vecteur de caractéristiques (\textit{features}) riche, multi-dimensionnel et physiquement pertinent pour chaque pixel.

\subsubsection{Extraction et calcul des indices spectraux}

À partir des bandes optiques de haute résolution (issues de Landsat ou de Sentinel-2), j'ai programmé le pipeline pour calculer dynamiquement une batterie d'indices :
\begin{itemize}
    \item \textbf{NDVI (\textit{Normalized Difference Vegetation Index})} : Il reflète la vigueur chlorophyllienne et l'activité photosynthétique. L'évapotranspiration refroidissant la surface par chaleur latente, une forte corrélation négative existe généralement entre NDVI et LST en période estivale.
    \item \textbf{NDWI (\textit{Normalized Difference Water Index})} : Permet de capter la teneur en eau de la végétation et l'humidité superficielle des sols, offrant une information vitale pour distinguer thermiquement un sol nu sec d'un sol nu humide.
    \item \textbf{SAVI (\textit{Soil Adjusted Vegetation Index}) et EVI (\textit{Enhanced Vegetation Index})} : Ces indices corrigent l'influence de la réflectance (brillance) du sol en arrière-plan. J'ai jugé leur intégration indispensable pour nos sites expérimentaux (comme Gebesee) en début de période de croissance des cultures, lorsque les sols ne sont que partiellement couverts par la canopée. La formulation du SAVI que j'ai implémentée inclut un facteur d'ajustement $L=0.5$ :
    \begin{equation}
        SAVI = \frac{\rho_{NIR} - \rho_{Red}}{\rho_{NIR} + \rho_{Red} + L} \times (1 + L)
    \end{equation}
\end{itemize}

\subsubsection{Intégration de la topographie (MNT)}

La thermodynamique de surface (et donc la LST) étant fortement dépendante de l'altitude (gradient adiabatique) et de l'exposition au rayonnement solaire incident, un Modèle Numérique de Terrain (MNT) a été intégré aux variables prédictives de mon pipeline. Pour chaque pixel, j'ai extrait l'altitude ($Z$), calculé la pente (\textit{slope}) et l'orientation (\textit{aspect}). Ces données topographiques sont reprojetées et rééchantillonnées à la résolution cible pour enrichir le tenseur de données spatiales.

\subsubsection{Modélisation de l'espace : Les coordonnées normalisées}

Afin de permettre aux algorithmes d'apprentissage de capturer la variabilité spatiale locale non expliquée par les indices spectraux (effets de micro-climats, ombres portées, advection locale, vents de surface), j'ai normalisé et intégré les coordonnées géographiques planaires $(X, Y)$ de chaque pixel comme variables explicatives au sein du \textit{dataset}.

\subsection{Implémentation du DMS : Algorithmes RandomForest et LightGBM}

L'algorithme de \textit{Data Mining Sharpening} (DMS) que j'ai implémenté se décompose en deux grandes phases mathématiques : la phase d'apprentissage (réalisée à la résolution grossière native du capteur thermique) et la phase d'inférence (réalisée à la haute résolution de l'optique).

\subsubsection{Le principe de l'apprentissage d'échelle (\textit{Scale-Invariance}) et de la réinjection des résidus}

L'hypothèse mathématique fondamentale du DMS est que la relation empirique liant la LST au vecteur de variables prédictives $V$ (NDVI, MNT, coordonnées, etc.) est invariante par changement d'échelle. Le processus que j'ai codé s'articule ainsi :
\begin{enumerate}
    \item \textbf{Agrégation (\textit{Up-scaling})} : Les indices à haute résolution (ex: Sentinel-2 à 20~m) sont dégradés spatialement (par moyenne) pour correspondre exactement à l'emprise spatiale de la grille basse résolution thermique (ex: Sentinel-3 à 1~km).
    \item \textbf{Entraînement} : Le modèle de régression $f$ est entraîné sur cette grille kilométrique pour minimiser l'erreur quadratique entre la $LST_{BR}$ observée et la LST prédite, générant un résidu d'apprentissage $\epsilon_{BR}$ :
    \begin{equation}
        LST_{BR} = f(V_{BR}) + \epsilon_{BR} = f(NDVI_{BR}, NDWI_{BR}, MNT_{BR}, ...) + \epsilon_{BR}
    \end{equation}
    \item \textbf{Prédiction (\textit{Down-scaling})} : Le modèle entraîné $f$ est ensuite appliqué au tenseur des indices à leur résolution native décamétrique ($V_{HR}$) pour générer l'estimation haute résolution ($\widehat{LST}_{HR}$).
    \item \textbf{Correction énergétique globale} : Pour garantir la loi de conservation de l'énergie (le flux thermique moyen d'un pixel d'un kilomètre ne doit pas être altéré par le modèle), le résidu d'apprentissage kilométrique $\epsilon_{BR}$ est interpolé par lissage bilinéaire spatio-continu ($\mathcal{I}_{bilin\acute{e}aire}$) à l'échelle fine, puis additionné à la prédiction :
    \begin{equation}
        LST_{DMS, HR} = f(V_{HR}) + \mathcal{I}_{bilin\acute{e}aire}(\epsilon_{BR})
    \end{equation}
\end{enumerate}

\begin{figure}[htbp]
    \centering
    \vspace{7cm} % Espace pour un organigramme détaillant l'entraînement et l'inférence du DMS
    \caption{Architecture du processus de \textit{Data Mining Sharpening} : de l'agrégation des prédicteurs optiques à la réinjection des résidus thermiques pour garantir l'intégrité radiométrique.}
    \label{fig:dms_process}
\end{figure}

\subsubsection{Évolution des modèles : Du RandomForest au LightGBM}

Dans ma première itération logicielle, j'ai implémenté un algorithme de \textbf{\textit{Random Forest Regressor}} (Forêts Aléatoires). Ses avantages conceptuels résident dans sa forte robustesse au sur-apprentissage et sa capacité à gérer les relations non-linéaires sans nécessiter de normalisation stricte des \textit{features} d'entrée.

Cependant, j'ai rapidement été confronté à des contraintes de scalabilité industrielle. Face à la volumétrie massive des données (les scènes fusionnées ECOSTRESS/Sentinel-2 pouvant contenir plusieurs dizaines de millions de pixels), le temps d'entraînement et, surtout, l'empreinte mémoire vive (\textit{Out-Of-Memory} - OOM) du modèle RandomForest devenaient des goulots d'étranglement inacceptables pour le pipeline.

Pour pallier ce problème d'ingénierie, j'ai restructuré mon code pour y intégrer \textbf{LightGBM} (\textit{Light Gradient Boosting Machine}). Contrairement au RandomForest (qui construit des arbres profonds et indépendants en parallèle), LightGBM construit des arbres séquentiellement en se concentrant sur la réduction du gradient d'erreur des itérations précédentes (\textit{Boosting}) et adopte une stratégie de croissance des arbres par les feuilles (\textit{leaf-wise}). Les résultats de mes \textit{benchmarks} ont été concluants : l'implémentation de LightGBM a permis de diviser le temps de calcul par un facteur 10 tout en consommant 5 fois moins de mémoire RAM, tout en offrant des performances statistiques quasi-identiques, voire supérieures sur des paysages particulièrement hétérogènes.

\subsection{Optimisation automatisée : Le framework Optuna}

Un modèle ensembliste basé sur les arbres de décision possède une multitude d'hyperparamètres (profondeur, nombre d'estimateurs, taux d'apprentissage) qu'il est préjudiciable de fixer de manière statique. Les dynamiques thermiques d'une scène hivernale enneigée diffèrent drastiquement d'une scène de canicule estivale ; le modèle doit s'y adapter.

Pour automatiser ce réglage fin, j'ai implémenté un module d'optimisation intelligente basé sur le framework \textbf{Optuna} au sein de mon script \texttt{optimisation\_pipeline.py}.

\subsubsection{La recherche bayésienne par estimateur TPE}
Contrairement à une approche naïve par quadrillage (\textit{GridSearch}) qui teste exhaustivement et aveuglément toutes les combinaisons, j'ai configuré Optuna pour utiliser un algorithme d'optimisation bayésienne, le TPE (\textit{Tree-structured Parzen Estimator}). Le framework modélise statistiquement l'espace des hyperparamètres et explore prioritairement les configurations prometteuses, tout en abandonnant prématurément les pistes menant à de mauvaises métriques d'évaluation (\textit{Pruning}).

\subsubsection{Fonction objectif et Validation Croisée (\textit{Cross-Validation})}
J'ai défini la fonction objectif d'Optuna de manière à minimiser l'Erreur Quadratique Moyenne (RMSE) spatiale, évaluée par une validation croisée à $K=5$ plis (\textit{K-Fold}) sur la grille d'entraînement basse résolution. 

L'espace de recherche (\textit{search space}) que j'ai défini pour LightGBM optimise conjointement les paramètres suivants :
\begin{itemize}
    \item \texttt{n\_estimators} : Le nombre total de classifieurs de \textit{boosting}.
    \item \texttt{max\_depth} et \texttt{num\_leaves} : Les bornes contrôlant la complexité structurelle de chaque arbre, essentielles pour lutter contre le surapprentissage des données spatiales bruitées.
    \item \texttt{learning\_rate} : Le pas (taux d'apprentissage) définissant le poids accordé à chaque nouvel arbre.
    \item \texttt{feature\_fraction} : Le sous-échantillonnage de variables, forçant la diversité des arbres en sélectionnant un pourcentage aléatoire des indices spectraux à chaque itération.
\end{itemize}
Grâce à ce développement, le pipeline de désagrégation est devenu totalement autonome, s'auto-ajustant dynamiquement aux spécificités biophysiques de chaque scène géospatiale traitée.

\subsection{Résultats intermédiaires : Comparaisons visuelles et statistiques}

Afin de valider rigoureusement la pertinence de mon implémentation DMS, j'ai réalisé des évaluations croisées sur divers couples satellitaires : désagrégation Landsat intra-capteur (100~m vers 30~m), synergie ECOSTRESS + Sentinel-2 (70~m vers 20~m), et fusion Sentinel-3 + Sentinel-2 (1~km vers 20~m / 300~m).

\begin{figure}[htbp]
    \centering
    \vspace{7cm} % Espace pour insérer la Figure 1 : Comparaison visuelle des cartes LST - Originale vs Sharpened
    \caption{Comparaison qualitative de la résolution spatiale. À gauche : La LST native (ex: Sentinel-3 à 1~km). À droite : La LST désagrégée par le modèle DMS (ex: fusion S3/S2 à 20~m), révélant la mosaïque thermique du parcellaire agricole.}
    \label{fig:visual_sharpening}
\end{figure}

\subsubsection{Analyse visuelle (Qualitative)}
Les analyses visuelles que j'ai produites démontrent une amélioration spectaculaire de la résolution des structures paysagères. Là où le produit Sentinel-3 d'origine ne restitue que de vastes gradients flous d'échelle kilométrique, amalgamant forêt, champs et infrastructures, l'image "super-résolue" (DMS) à 20~mètres permet d'isoler la signature thermique de chaque parcelle agricole individuelle. 

Les lisières de forêts, naturellement rafraîchies par l'intensité de la transpiration de la canopée profonde, apparaissent avec des frontières nettes, contrastant violemment avec les champs voisins récoltés, dont les sols nus s'échauffent fortement sous le forçage radiatif.

\subsubsection{Analyse statistique (Quantitative) par dégradation synthétique}
Afin de chiffrer l'erreur intrinsèque du modèle mathématique sans être biaisé par des facteurs externes, j'ai mis en place un protocole d'évaluation par dégradation synthétique. J'ai artificiellement dégradé les données natives de Landsat (100~m) à 1~km pour simuler la résolution de Sentinel-3. J'ai ensuite appliqué mon modèle DMS pour redescendre cette résolution dégradée vers sa définition native (100~m), permettant ainsi une confrontation parfaite pixel à pixel entre ma prédiction et la "vérité" spatiale d'origine.

Les résultats statistiques moyens, obtenus sur des sites expérimentaux hétérogènes (tels que Gebesee et Selhausen), valident sans équivoque l'architecture logicielle déployée :

\begin{table}[htbp]
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Méthode de \textit{Downscaling}} & \textbf{RMSE ($^\circ$C)} & \textbf{MAE ($^\circ$C)} & \textbf{R$^2$ (Spatial)} \\
\midrule
TsHARP (Baseline Linéaire)     & 2.85                  & 2.10                 & 0.72                 \\
DMS - \textit{Random Forest}             & 1.62                  & 1.25                 & 0.89                 \\
\textbf{DMS - LightGBM (Optimisé Optuna)}  & \textbf{1.58}                  & \textbf{1.21}                 & \textbf{0.90}                 \\
\bottomrule
\end{tabular}
\caption{Performances relatives des différents algorithmes de super-résolution sur le site de Gebesee (protocole de simulation synthétique 1~km $\rightarrow$ 100~m).}
\label{tab:perf_dms}
\end{table}

L'analyse de ces métriques indique que le passage d'une régression empirique linéaire (TsHARP) au Machine Learning (DMS) réduit l'erreur quadratique moyenne de près de 1,2~$^\circ$C, tout en capturant beaucoup plus finement la variance spatiale (le coefficient de détermination $R^2$ bondissant de 0.72 à 0.90). L'utilisation de LightGBM combinée à l'optimisation bayésienne d'Optuna délivre les meilleures performances globales.

\subsubsection{Limites observées et nécessité du garde-fou résiduel}
Malgré la solidité de ces résultats, l'analyse approfondie de mes cartes d'erreurs a permis d'identifier certaines limites de la modélisation. Dans des environnements où le vecteur d'entrée optique (NDVI, MNT) est extrêmement homogène mais où la température de surface varie en réalité sous l'effet de facteurs invisibles dans le spectre solaire (par exemple : le brassage éolien à la surface de vastes plans d'eau, ou les remontées capillaires d'humidité souterraine profonde), l'algorithme peine logiquement à générer des gradients thermiques fins et cohérents.

C'est dans ces zones d'incertitude que l'étape algorithmique de la "correction des résidus" à basse résolution (étape finale du DMS) joue son rôle crucial de garde-fou. Elle garantit que, même si le modèle optique se trompe localement à 20~m, la déviation absolue de la température à l'échelle kilométrique reste contrainte par la mesure thermique spatiale réelle.

\subsection{Conclusion partielle}
L'architecture DMS que j'ai implémentée et automatisée permet de lever le premier grand verrou technologique du suivi du stress hydrique. Ce produit hybride de très haute qualité (la résolution temporelle de Sentinel-3/ECOSTRESS croisée avec la résolution spatiale de Sentinel-2) sert désormais de donnée d'entrée fondamentale et fiable pour l'étape suivante : la modélisation physique complexe de l'évapotranspiration.

\section{Contribution III : Modélisation de l'Évapotranspiration et Prédictions Temporelles}
\label{sec:contrib_et_lstm}

Une fois la Température de Surface (LST) obtenue à haute résolution spatiale grâce à mes algorithmes de \textit{Machine Learning} (DMS), la suite logique et l'objectif final de mon pipeline géospatial a consisté à estimer l'Évapotranspiration (ET). L'ET, qui représente la somme de l'évaporation de l'eau contenue dans le sol et de la transpiration des végétaux, constitue la variable fondamentale pour la gestion de l'irrigation et l'anticipation du stress hydrique.

Ce chapitre détaille l'implémentation algorithmique de deux modèles physiques d'évapotranspiration, la méthodologie de spatialisation des forçages météorologiques (ERA5), ainsi que le développement d'une architecture d'Intelligence Artificielle complexe (LSTM Seq2Seq) pour la prédiction temporelle de ces flux.

\subsection{Implémentation des modèles physiques : TTME et PT-SINRH}

L'estimation de l'évapotranspiration à l'échelle du pixel satellitaire ne peut pas être mesurée de manière directe ; elle doit être modélisée en résolvant le bilan d'énergie de surface. En m'appuyant sur le pipeline d'extraction automatisé défini précédemment[cite: 1], j'ai croisé la LST, les indices optiques (NDVI, albédo) et les données météorologiques pour alimenter deux modèles physiques antagonistes mais complémentaires, que j'ai traduits de la théorie mathématique vers un code Python vectorisé.

\subsubsection{Le modèle TTME (Two-source Trapezoid Model for Evapotranspiration)}

Le modèle TTME est une approche conceptuelle bi-source (séparant le comportement du sol nu de celui de la canopée) qui repose sur l'analyse de l'espace thermodynamique (ou \textit{scatter-plot}) formé par le couple LST-NDVI. L'hypothèse est que pour un forçage radiatif donné, la température d'un pixel dépend linéairement de sa couverture végétale et de sa disponibilité en eau.

L'implémentation algorithmique de ce modèle au sein de mon module \texttt{Traitement/TTME/} s'est déroulée en trois phases d'ingénierie mathématique :

\begin{enumerate}
    \item \textbf{Définition algorithmique des Bords (\textit{Edges}) :} J'ai développé un algorithme de détection d'enveloppe convexe pour identifier dynamiquement, sur chaque image satellitaire, les limites théoriques du trapèze thermodynamique. 
    \begin{itemize}
        \item \textbf{Le bord chaud (Sec) :} Correspond aux pixels souffrant d'un stress hydrique total (évapotranspiration nulle). L'algorithme ajuste une régression linéaire définissant $T_{sec} = a \cdot NDVI + b$.
        \item \textbf{Le bord froid (Humide) :} Correspond aux pixels évaporant à un taux potentiel (sans limitation d'eau). L'algorithme définit $T_{humide} = c \cdot NDVI + d$.
    \end{itemize}
    
    \item \textbf{Calcul de la Fraction d'Évapotranspiration ($EF$) :} Pour chaque pixel $(x,y)$ de l'image, j'ai implémenté le calcul de la fraction d'évapotranspiration en interpolant sa position relative au sein du trapèze :
    \begin{equation}
        EF(x,y) = \frac{T_{sec}(NDVI_{x,y}) - LST(x,y)}{T_{sec}(NDVI_{x,y}) - T_{humide}(NDVI_{x,y})}
    \end{equation}
    
    \item \textbf{Vectorisation Numpy et Bilan d'énergie :} Connaissant $EF$, le flux de chaleur latente ($LE$, équivalent énergétique de l'ET) est déduit du rayonnement net ($R_n$) et du flux de chaleur dans le sol ($G$). Afin de garantir des performances industrielles, j'ai proscrit l'usage de boucles itératives (\texttt{for}) au profit d'opérations matricielles via la bibliothèque \texttt{numpy}.
    \begin{equation}
        LE(x,y) = EF(x,y) \times (R_n(x,y) - G(x,y))
    \end{equation}
    Cette optimisation mathématique m'a permis de traiter des hypercubes de plusieurs millions de pixels en quelques secondes d'exécution.
\end{enumerate}

\begin{figure}[htbp]
    \centering
    \vspace{6cm} % Espace pour un graphique de type Scatter Plot LST vs NDVI avec le trapèze dessiné
    \caption{Représentation de l'espace LST-NDVI calculé par le pipeline. Le gradient de couleur illustre la fraction d'évapotranspiration ($EF$) dérivée de la position des pixels entre le bord sec et le bord humide.}
    \label{fig:ttme_trapezoid}
\end{figure}

\subsubsection{Le modèle PT-SINRH}

Le modèle TTME présente une faiblesse majeure : par temps hivernal ou sur des zones massivement nuageuses, le manque de contraste thermique empêche la formation d'un trapèze cohérent. Pour pallier cette vulnérabilité, j'ai implémenté un second modèle : le PT-SINRH (\textit{Priestley-Taylor Spatial Interpolation of Net Radiation and Heat}).

Ce modèle s'affranchit de la délimitation géométrique et s'appuie sur l'équation de Priestley-Taylor, modifiée par un coefficient $\alpha$ dynamique intégrant des contraintes écophysiologiques, telles que l'Indice de Surface Foliaire (LAI) et la conductance stomatique. J'ai codé la relation suivante pour déterminer le flux de chaleur latente :
\begin{equation}
    LE = \alpha \frac{\Delta}{\Delta + \gamma} (R_n - G)
\end{equation}
où $\Delta$ est la pente de la courbe de pression de vapeur saturante et $\gamma$ la constante psychrométrique. 

L'implémentation logicielle du PT-SINRH a nécessité le développement de fonctions mathématiques complexes pour modéliser la réponse de la plante face au Déficit de Pression de Vapeur ($VPD$). Bien que plus lourd en calculs radiatifs, j'ai pu démontrer que ce modèle offrait une bien meilleure robustesse temporelle sur des séries annuelles complètes, là où TTME échouait lors des mois à faible forçage solaire.

\subsection{Gestion des forçages météo : Le \textit{Downscaling} spatial d'ERA5}

Les équations de l'évapotranspiration requièrent impérativement des variables météorologiques locales (Température de l'air $T_a$, Humidité relative $RH$, Vitesse du vent $U$). L'utilisation d'une unique station météo au sol pour une région entière introduit des biais topographiques majeurs.

Grâce au connecteur API intégré à mon fichier \texttt{config.py}[cite: 1], mon pipeline télécharge automatiquement les réanalyses climatiques mondiales \textbf{ERA5}. Toutefois, la grille native d'ERA5 (environ 30~km $\times$ 30~km) est géométriquement incompatible avec les pixels satellitaires de 20~m ou 30~m. 

\subsubsection{Algorithme de descente d'échelle de la Température de l'air ($T_a$)}

Pour éviter un "effet de bloc" visuel et des aberrations thermodynamiques, j'ai développé un algorithme de \textit{Downscaling} météorologique spécifique (\texttt{downscaling\_Ta.py}). La température de l'air étant intimement régie par la loi des gaz parfaits et la pression atmosphérique, j'ai fondé ma désagrégation sur la topographie. 

Mon algorithme exécute la séquence suivante :
\begin{enumerate}
    \item \textbf{Extraction et Reprojection} : Extraction du tenseur $T_{a, ERA5}$ et de l'altitude moyenne de la maille géopotentielle d'ERA5. J'utilise ensuite une interpolation bilinéaire continue pour projeter cette grille kilométrique sur le maillage fin à 20~m.
    \item \textbf{Correction altimétrique (Gradient Adiabatique)} : Je calcule le différentiel d'altitude entre le Modèle Numérique de Terrain réel à très haute résolution ($Z_{MNT}$) et la surface lissée du modèle ERA5 ($Z_{ERA5}$). J'applique ensuite un gradient adiabatique moyen ($\Gamma \approx -0.0065 \, ^\circ\text{C/m}$) pour réajuster la température locale :
    \begin{equation}
        T_{a, 20m}(x,y) = T_{a, ERA5\_interp}(x,y) + \Gamma \times \left( Z_{MNT}(x,y) - Z_{ERA5}(x,y) \right)
    \end{equation}
\end{enumerate}

Cette correction déterministe m'a permis de recréer artificiellement les microclimats de vallées et de crêtes, améliorant de manière critique la justesse du déficit de pression de vapeur ($VPD$) injecté dans mes modèles d'évapotranspiration.

\begin{figure}[htbp]
    \centering
    \vspace{5cm} % Espace pour illustrer l'effet du downscaling : une image ERA5 pixelisée vs une image Ta épousant le relief
    \caption{Impact du \textit{Downscaling} topographique sur la température de l'air. À gauche : le forçage ERA5 brut. À droite : le forçage affiné à 20~m, restituant la variabilité thermique liée au relief.}
    \label{fig:downscaling_era5}
\end{figure}

\subsection{L'anticipation par l'IA : Le \textit{Forecasting} par Réseaux de Neurones (LSTM)}

L'estimation spatiale de l'évapotranspiration à l'instant du passage satellitaire résout le diagnostic hydrique passé. Cependant, la couverture nuageuse rend ces observations discontinues. Pour fournir un véritable outil d'aide à l'irrigation, il est impératif de pouvoir prédire l'état hydrique futur d'une parcelle sur un horizon de $J+1$ à $J+7$. 

Pour franchir ce verrou temporel, j'ai conçu et entraîné une architecture d'apprentissage profond (\textit{Deep Learning}) basée sur des réseaux de neurones récurrents.

\subsubsection{Architecture Encodeur-Décodeur (Seq2Seq LSTM)}

L'état hydrique d'un sol aujourd'hui dépend de son inertie thermique et des précipitations des semaines précédentes. Pour modéliser cette mémoire complexe, j'ai implémenté une architecture \textbf{LSTM Seq2Seq (\textit{Sequence-to-Sequence})} sous PyTorch.

\begin{itemize}
    \item \textbf{L'Encodeur} : Ingère la série temporelle historique des 30 derniers jours (\texttt{LSTM\_LOOKBACK = 30}). Il traite les forçages ERA5 passés (précipitations, rayonnements, températures) et la dynamique végétale (NDVI interpolé). Son rôle est de comprimer l'historique hydrique du sol dans un vecteur d'état latent (le contexte $h_t$).
    \item \textbf{Le Décodeur} : Initialisé par ce contexte $h_t$, il reçoit en entrée les prévisions météorologiques futures pour les 7 prochains jours et génère séquentiellement, pas à pas, la prédiction de l'évapotranspiration future ($Y_{t:t+7}$).
\end{itemize}

\begin{figure}[htbp]
    \centering
    \vspace{6cm} % Espace pour un schéma de l'architecture Encodeur-Décodeur LSTM
    \caption{Architecture du réseau de neurones LSTM Seq2Seq implémenté pour la prévision temporelle multi-pas de l'évapotranspiration.}
    \label{fig:lstm_architecture}
\end{figure}

\subsubsection{Preuve de concept physique (\textit{Hindcast}) et Explicabilité (Captum)}

Afin de garantir que mon réseau de neurones n'agissait pas comme une simple "boîte noire" statistique, j'ai d'abord réalisé un entraînement en mode \textit{Hindcast} (utilisation de forçages ERA5 parfaits, passés). 

J'ai ensuite soumis ce modèle à un audit d'Intelligence Artificielle Explicable (XAI) en utilisant la méthode des \textbf{Gradients Intégrés} de la librairie Captum. L'analyse mathématique de l'attribution des variables a prouvé que mon modèle avait intuitivement appris la thermodynamique :
\begin{itemize}
    \item Le réseau accordait le poids décisionnel le plus fort au Rayonnement Net ($R_n$), moteur physique principal du bilan d'énergie.
    \item La variable végétale (SAVI/NDVI) agissait comme un modulateur clé lors des épisodes de croissance.
\end{itemize}
Cette validation physique m'a permis d'affirmer la robustesse théorique de l'architecture choisie.

\subsubsection{Le passage à l'opérationnel (\textit{Forecast}) : Gestion de l'incertitude MLOps}

La phase ultime de mon stage a consisté à basculer du modèle théorique vers un déploiement \textit{Forecasting} opérationnel, en remplaçant les réanalyses parfaites ERA5 par des prévisions météorologiques incertaines issues de l'API **Open-Meteo**.

Ce passage à la réalité industrielle a immédiatement révélé un problème classique en \textit{Machine Learning} temporel : le \textbf{\textit{Smoothing Effect}} (effet de lissage). Face au bruit et à l'incertitude des prévisions météo à J+5 ou J+7, le réseau LSTM, cherchant à minimiser son erreur quadratique (MSE), devenait "peureux" et tendait à prédire une droite moyenne, échouant à anticiper les pics d'évapotranspiration soudains.

Pour résoudre ce défi \textit{MLOps}, j'ai dû profondément optimiser mon pipeline d'entraînement :
\begin{enumerate}
    \item \textbf{Régularisation architecturale} : J'ai ajusté le taux d'abandon (\textit{Dropout}) dans les couches récurrentes et implémenté une pénalité sur les poids (\textit{Weight Decay}) pour empêcher le modèle de sur-apprendre le bruit des prévisions météorologiques.
    \item \textbf{Early Stopping} : J'ai mis en place une routine d'arrêt précoce surveillant la perte de validation stricte, évitant l'effondrement des performances en prévision lointaine.
    \item \textbf{Généralisation Multi-sites} : J'ai fusionné les bases de données d'apprentissage issues de multiples stations ICOS européennes. L'augmentation massive de la variance des paysages a forcé le modèle à extraire des lois physiques générales plutôt que de mémoriser le microclimat d'une seule station.
\end{enumerate}

\begin{figure}[htbp]
    \centering
    \vspace{7cm} % Espace pour un graphe de séries temporelles montrant Prédiction J+7 vs Vérité Terrain
    \caption{Prédictions temporelles du modèle LSTM (\textit{Forecast} opérationnel) confrontées à la vérité terrain des tours à flux ICOS. Le modèle démontre une excellente capacité à capter la dynamique non-linéaire du forçage radiatif.}
    \label{fig:lstm_results}
\end{figure}

Les résultats finaux ont été extrêmement probants : la racine de l'erreur quadratique moyenne (RMSE) sur les prédictions à l'aveugle s'est stabilisée autour de 30~$\text{W}/\text{m}^2$ pour le flux de chaleur latente. 

En conclusion, la combinaison de la désagrégation spatiale (Chapitre précédent) et de la prévision temporelle LSTM transforme des images satellitaires disparates et nuageuses en une série temporelle spatiale continue. Cet outil d'ingénierie ouvre des perspectives concrètes pour la modélisation prédictive du stress hydrique en milieu agricole.
\end{document}
