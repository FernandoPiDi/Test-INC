import pickle


# ============================================================================
# CARGAR MODELO Y FEATURES
# ============================================================================

# Cargar modelo Random Forest entrenado
try:
    with open("modelo_rf.pkl", "rb") as f:
        modelo = pickle.load(f)
    print("✓ Modelo Random Forest cargado exitosamente")
except FileNotFoundError:
    print("⚠ WARNING: modelo_rf.pkl no encontrado. Ejecute data_pipeline.py primero.")
    modelo = None

# Cargar nombres de features esperados
try:
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    print(f"✓ Features cargados: {len(feature_names)} columnas")
except FileNotFoundError:
    print("⚠ WARNING: feature_names.pkl no encontrado.")
    feature_names = None
