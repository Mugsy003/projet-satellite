"""
main.py - Pipeline Principal du Projet Satellite
=================================================
Script orchestrateur qui exécute toutes les étapes du pipeline
dans l'ordre : Extraction → Transformation → Machine Learning → Validation → Visualisation.

Chaque étape peut être activée/désactivée individuellement via
PIPELINE_STEPS dans config.py.

Usage :
    python main.py
"""
import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

from config import LOGGER, PIPELINE_STEPS


def run_step(name, func):
    """Exécute une étape du pipeline avec gestion d'erreurs et chronométrage."""
    if not PIPELINE_STEPS.get(name, False):
        LOGGER.info(f"⏭️  Étape '{name}' désactivée dans config.py. Skip.")
        return True
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"▶️  DÉMARRAGE : {name}")
    LOGGER.info(f"{'='*60}")
    
    start = time.time()
    try:
        func()
        elapsed = time.time() - start
        LOGGER.info(f"✅ {name} terminé en {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        LOGGER.error(f"❌ {name} a échoué après {elapsed:.1f}s : {e}")
        return False


def main():
    LOGGER.info("🚀 DÉMARRAGE DU PIPELINE COMPLET")
    LOGGER.info(f"   Étapes activées : {[k for k, v in PIPELINE_STEPS.items() if v]}")
    
    resultats = {}
    start_global = time.time()

    # ========================================
    # 1. EXTRACTION
    # ========================================
    
    # 1a. Extraction Landsat
    def step_extraction_landsat():
        from Extraction.Landsat.main_extract_landsat import main as extract_landsat
        extract_landsat()
    resultats["extraction_landsat"] = run_step("extraction_landsat", step_extraction_landsat)

    # 1b. Extraction ECOSTRESS
    def step_extraction_ecostress():
        from Extraction.ECOSTRESS.main_extract_ecostress import main as extract_ecostress
        extract_ecostress()
    resultats["extraction_ecostress"] = run_step("extraction_ecostress", step_extraction_ecostress)

    # 1c. Extraction Sentinel-2
    def step_extraction_sentinel():
        from Extraction.Sentinel2.main_extract_sentinel import main as extract_sentinel
        extract_sentinel()
    resultats["extraction_sentinel"] = run_step("extraction_sentinel", step_extraction_sentinel)

    # 1d. Extraction Paires ECOSTRESS + Sentinel-2
    def step_extraction_paires():
        from Extraction.Sentinel2.main_extract_paires_eco_s2 import main as extract_paires
        extract_paires()
    resultats["extraction_paires_eco_s2"] = run_step("extraction_paires_eco_s2", step_extraction_paires)

    # 1e. Extraction Sentinel-3
    def step_extraction_sentinel3():
        from Extraction.Sentinel3.main_extract_sentinel3 import main as extract_s3
        extract_s3()
    resultats["extraction_sentinel3"] = run_step("extraction_sentinel3", step_extraction_sentinel3)

    # 1f. Extraction Paires Sentinel-3 + Sentinel-2
    def step_extraction_paires_s3_s2():
        from Extraction.Sentinel2.main_extract_paires_s3_s2 import main as extract_paires_s3
        extract_paires_s3()
    resultats["extraction_paires_s3_s2"] = run_step("extraction_paires_s3_s2", step_extraction_paires_s3_s2)

    # ========================================
    # 2. TRANSFORMATION
    # ========================================
    
    # 2a. Transformation Landsat
    def step_transform_landsat():
        from Transform.Landsat.main_transform_landsat import main as transform_landsat
        transform_landsat()
    resultats["transform_landsat"] = run_step("transform_landsat", step_transform_landsat)

    # 2b. Transformation ECOSTRESS
    def step_transform_ecostress():
        from Transform.ECOSTRESS.main_transform_ecostress import main as transform_ecostress
        transform_ecostress()
    resultats["transform_ecostress"] = run_step("transform_ecostress", step_transform_ecostress)

    # 2c. Transformation Sentinel-2
    def step_transform_sentinel():
        from Transform.Sentinel2.main_transform_sentinel import main as transform_sentinel
        transform_sentinel()
    resultats["transform_sentinel"] = run_step("transform_sentinel", step_transform_sentinel)

    # 2d. Transformation Sentinel-3
    def step_transform_sentinel3():
        from Transform.Sentinel3.main_transform_sentinel3 import main as transform_s3
        transform_s3()
    resultats["transform_sentinel3"] = run_step("transform_sentinel3", step_transform_sentinel3)

    # ========================================
    # 3. MACHINE LEARNING (Sharpening)
    # ========================================
    
    # 3a. DMS Sharpening Landsat
    def step_dms_landsat():
        from Transform.Landsat.dms_sharpening_landsat import main as dms_landsat
        dms_landsat()
    resultats["sharpening_dms_landsat"] = run_step("sharpening_dms_landsat", step_dms_landsat)

    # 3b. TsHARP Sharpening Landsat
    def step_tsharp_landsat():
        from Transform.Landsat.tsharp_landsat import main as tsharp_landsat
        tsharp_landsat()
    resultats["sharpening_tsharp_landsat"] = run_step("sharpening_tsharp_landsat", step_tsharp_landsat)

    # 3c. DMS Fusion (ECOSTRESS + S2)
    def step_fusion_dms():
        from Transform.Fusion.dms_sharpening_fusion import main as fusion_dms
        fusion_dms()
    resultats["fusion_dms"] = run_step("fusion_dms", step_fusion_dms)

    # 3d. TsHARP Fusion (ECOSTRESS + S2)
    def step_fusion_tsharp():
        from Transform.Fusion.tsharp_fusion import main as fusion_tsharp
        fusion_tsharp()
    resultats["fusion_tsharp"] = run_step("fusion_tsharp", step_fusion_tsharp)

    # 3e. DMS Sharpening Sentinel-3 (pur)
    def step_dms_sentinel3():
        from Transform.Sentinel3.dms_sharpening_sentinel3 import main as dms_s3
        dms_s3()
    resultats["sharpening_dms_sentinel3"] = run_step("sharpening_dms_sentinel3", step_dms_sentinel3)

    # 3f. DMS Fusion (Sentinel-3 + S2)
    def step_fusion_dms_s3_s2():
        from Transform.Fusion.dms_sharpening_s3_s2 import main as dms_s3_s2
        dms_s3_s2()
    resultats["fusion_dms_s3_s2"] = run_step("fusion_dms_s3_s2", step_fusion_dms_s3_s2)

    # ========================================
    # 4. VALIDATION
    # ========================================
    
    def step_comparaison():
        # comparaison_ICOS est un script, on l'exécute via exec
        exec(open("comparaison_ICOS.py", encoding="utf-8").read())
    resultats["comparaison_icos"] = run_step("comparaison_icos", step_comparaison)

    # ========================================
    # 5. VISUALISATION
    # ========================================
    
    def step_visualisation():
        exec(open("visualisation_performances.py", encoding="utf-8").read())
    resultats["visualisation"] = run_step("visualisation", step_visualisation)

    # ========================================
    # RÉSUMÉ FINAL
    # ========================================
    elapsed_global = time.time() - start_global
    
    print(f"\n{'='*60}")
    print(f"📊 RÉSUMÉ DU PIPELINE (durée totale : {elapsed_global:.1f}s)")
    print(f"{'='*60}")
    
    for step_name, success in resultats.items():
        status = "✅" if success else "❌"
        print(f"   {status} {step_name}")
    
    nb_ok = sum(1 for v in resultats.values() if v)
    nb_total = len(resultats)
    print(f"\n   {nb_ok}/{nb_total} étapes réussies.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
