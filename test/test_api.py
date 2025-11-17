"""
Script de prueba para la API de Predicción de Adherencia
Realiza llamadas de ejemplo a los endpoints de la API
"""

import requests
import json

# URL base de la API
BASE_URL = "http://localhost:8000"

def print_section(title):
    """Imprime una sección con formato"""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def test_root():
    """Prueba el endpoint raíz"""
    print_section("TEST 1: Root Endpoint")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_health():
    """Prueba el health check"""
    print_section("TEST 2: Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_predict_individual():
    """Prueba predicción individual"""
    print_section("TEST 3: Predicción Individual")
    
    # Datos de ejemplo del paciente
    paciente = {
        "sexo": "Femenino",
        "edad": 55,
        "zona_residencia": "Urbana",
        "tipo_cancer": "Mama",
        "estadio": "II",
        "aseguradora": "EPS_A",
        "count_consultas": 5,
        "count_laboratorios": 3,
        "avg_Biopsia": 1.2,
        "avg_Marcador_CA125": 35.5
    }
    
    print(f"Datos de entrada:")
    print(json.dumps(paciente, indent=2, ensure_ascii=False))
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=paciente,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nStatus: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_predict_batch():
    """Prueba predicción en batch"""
    print_section("TEST 4: Predicción Batch")
    
    # Datos de ejemplo de múltiples pacientes
    batch = {
        "pacientes": [
            {
                "sexo": "Femenino",
                "edad": 55,
                "zona_residencia": "Urbana",
                "tipo_cancer": "Mama",
                "estadio": "II",
                "aseguradora": "EPS_A",
                "count_consultas": 5,
                "count_laboratorios": 3,
                "avg_Biopsia": 1.2,
                "avg_Marcador_CA125": 35.5
            },
            {
                "sexo": "Masculino",
                "edad": 62,
                "zona_residencia": "Rural",
                "tipo_cancer": "Próstata",
                "estadio": "III",
                "aseguradora": "EPS_B",
                "count_consultas": 2,
                "count_laboratorios": 1,
                "avg_PSA": 8.5
            }
        ]
    }
    
    print(f"Número de pacientes: {len(batch['pacientes'])}")
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=batch,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nStatus: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def main():
    """Ejecuta todas las pruebas"""
    print("\n" + "="*80)
    print("INICIANDO PRUEBAS DE LA API")
    print("="*80)
    print(f"URL Base: {BASE_URL}")
    
    try:
        # Ejecutar pruebas
        test_root()
        test_health()
        test_predict_individual()
        test_predict_batch()
        
        print("\n" + "="*80)
        print("TODAS LAS PRUEBAS COMPLETADAS")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: No se pudo conectar a la API")
        print("Asegúrese de que la API está ejecutándose en http://localhost:8000")
        print("\nPara iniciar la API, ejecute:")
        print("  python main.py")
        print("o")
        print("  uvicorn main:app --host 0.0.0.0 --port 8000")
        
    except Exception as e:
        print(f"\n❌ ERROR inesperado: {str(e)}")

if __name__ == "__main__":
    main()

