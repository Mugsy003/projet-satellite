import os
import subprocess
import sys
import argparse
import time

# Dictionnaire des scripts de visualisation
SCRIPTS = {
    "1": {
        "name": "1. Graphiques : Graphes temporels par Site (generer_graphes_par_site.py)",
        "path": os.path.join("Analyse", "Visualisations", "generer_graphes_par_site.py"),
        "description": "Génère les séries temporelles et les scatter plots pour chaque site."
    },
    "2": {
        "name": "2. Graphiques : Performances Globales ET (visualisation_performances_ET.py)",
        "path": os.path.join("Analyse", "Visualisations", "visualisation_performances_ET.py"),
        "description": "Génère les graphiques en barres des performances (RMSE, Biais) globales et par site."
    },
    "3": {
        "name": "3. Graphiques : Amélioration ET (plot_ET_improvement.py)",
        "path": os.path.join("Analyse", "Visualisations", "plot_ET_improvement.py"),
        "description": "Génère le graphique illustrant le gain de performance de l'ET."
    },
    "4": {
        "name": "4. Graphiques : Amélioration Ta (plot_Ta_improvement.py)",
        "path": os.path.join("Analyse", "Visualisations", "plot_Ta_improvement.py"),
        "description": "Génère le graphique illustrant le gain de performance de la température de l'air."
    }
}

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(script_key):
    script_info = SCRIPTS.get(script_key)
    if not script_info:
        print(f"[ ERROR ] Script invalide : {script_key}")
        return False
    
    script_path = script_info["path"]
    print(f"\n[ RUN ] Lancement de : {script_info['name']}")
    print(f"   Chemin : {script_path}")
    print("-" * 50)
    
    try:
        # Exécuter le script en s'assurant que le répertoire racine est dans le PYTHONPATH
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        
        result = subprocess.run([sys.executable, script_path], env=env, check=True)
        print("-" * 50)
        print(f"[ OK ] Terminé avec succès : {script_info['name']}\n")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 50)
        print(f"[ FAIL ] Erreur lors de l'exécution de {script_info['name']}. Code de retour : {e.returncode}\n")
        return False

def interactive_menu():
    while True:
        clear_terminal()
        print("=" * 60)
        print("   PIPELINE DE VISUALISATION ET D'ANALYSE")
        print("=" * 60)
        print("Sélectionnez les scripts à lancer :\n")
        
        for key, info in SCRIPTS.items():
            print(f" [{key}] {info['name']}")
            print(f"     -> {info['description']}")
            print()
            
        print(" [A] Lancer TOUT le pipeline (1 à 4 dans l'ordre)")
        print(" [Q] Quitter")
        print("=" * 60)
        
        choix = input("\n[>] Votre choix (ex: '1', '3 4 5', 'A', 'Q') : ").strip().upper()
        
        if choix == 'Q':
            print("Au revoir !")
            break
        elif choix == 'A':
            print("\n[INFO] Lancement du pipeline complet...")
            for key in sorted(SCRIPTS.keys()):
                run_script(key)
            input("\nAppuyez sur Entrée pour continuer...")
        else:
            # Séparer les choix s'il y a des espaces
            cles = choix.split()
            valides = [c for c in cles if c in SCRIPTS]
            invalides = [c for c in cles if c not in SCRIPTS]
            
            if invalides:
                print(f"\n[WARN] Choix ignorés (invalides) : {', '.join(invalides)}")
                
            if valides:
                for c in valides:
                    run_script(c)
                input("\nAppuyez sur Entrée pour continuer...")
            else:
                print("\nAucun choix valide effectué.")
                time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Lanceur pour les scripts d'analyse et de visualisation.")
    parser.add_argument("--all", action="store_true", help="Lance tous les scripts dans l'ordre.")
    parser.add_argument("--scripts", nargs="+", help="Liste des numéros de scripts à lancer (ex: --scripts 1 3 4).")
    
    args = parser.parse_args()
    
    # Si des arguments sont passés en ligne de commande, on bypass le menu
    if args.all:
        for key in sorted(SCRIPTS.keys()):
            run_script(key)
    elif args.scripts:
        for key in args.scripts:
            run_script(key)
    else:
        # Mode interactif par défaut
        import time
        interactive_menu()

if __name__ == "__main__":
    # Assurer que le PYTHONPATH contient le dossier racine du projet
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    main()
