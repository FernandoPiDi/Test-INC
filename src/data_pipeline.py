"""
Data Exploration and Processing Pipeline
Prueba Técnica - Gestión de Datos en Salud

Este script realiza:
1. Consultas SQL de exploración (como strings)
2. Carga de datos desde Excel
3. Limpieza y transformación de datos
4. Feature engineering y agregación
5. One-hot encoding para modelado ML
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# ============================================================================
# PARTE 1: CONSULTAS SQL DE EXPLORACIÓN (PostgreSQL)
# ============================================================================

# Query 1: Tiempo Diagnóstico-Prueba
# Calcula el tiempo en días entre la primera fecha de muestra de laboratorio 
# y la fecha de diagnóstico para cada paciente
SQL_QUERY_1_TIEMPO_DIAGNOSTICO = """
SELECT 
    p.id_paciente,
    p.fecha_dx,
    MIN(l.fecha_muestra) AS primera_muestra,
    (p.fecha_dx - MIN(l.fecha_muestra)) AS dias_diagnostico_prueba
FROM 
    Pacientes p
INNER JOIN 
    Laboratorios l ON p.id_paciente = l.id_paciente
GROUP BY 
    p.id_paciente, p.fecha_dx
ORDER BY 
    p.id_paciente;
"""

# Query 2: Inactividad
# Identifica pacientes sin consultas en los últimos 90 días
# Fecha de referencia: 2025-12-31
SQL_QUERY_2_INACTIVIDAD = """
SELECT 
    p.id_paciente,
    p.sexo,
    p.edad,
    p.tipo_cancer,
    MAX(c.fecha_consulta) AS ultima_consulta
FROM 
    Pacientes p
LEFT JOIN 
    Consultas c ON p.id_paciente = c.id_paciente
GROUP BY 
    p.id_paciente, p.sexo, p.edad, p.tipo_cancer
HAVING 
    MAX(c.fecha_consulta) IS NULL 
    OR MAX(c.fecha_consulta) < DATE '2025-12-31' - INTERVAL '90 days'
ORDER BY 
    p.id_paciente;
"""

# Query 3: Métricas por Cáncer (Dashboard)
# Genera métricas agregadas por tipo de cáncer para dashboard
SQL_QUERY_3_METRICAS_CANCER = """
WITH fecha_limite_6m AS (
    SELECT DATE '2025-12-31' - INTERVAL '6 months' AS fecha_corte
)
SELECT 
    p.tipo_cancer,
    COUNT(DISTINCT p.id_paciente) AS total_pacientes,
    ROUND(AVG(p.edad), 2) AS edad_promedio,
    COUNT(DISTINCT c.oid_consulta) AS total_consultas_6m,
    COUNT(DISTINCT l.id_lab) AS total_laboratorios
FROM 
    Pacientes p
LEFT JOIN 
    Consultas c ON p.id_paciente = c.id_paciente 
        AND c.fecha_consulta >= (SELECT fecha_corte FROM fecha_limite_6m)
LEFT JOIN 
    Laboratorios l ON p.id_paciente = l.id_paciente
GROUP BY 
    p.tipo_cancer
ORDER BY 
    total_pacientes DESC;
"""

print("="*80)
print("CONSULTAS SQL DE EXPLORACIÓN")
print("="*80)
print("\n--- Query 1: Tiempo Diagnóstico-Prueba ---")
print(SQL_QUERY_1_TIEMPO_DIAGNOSTICO)
print("\n--- Query 2: Inactividad (últimos 90 días) ---")
print(SQL_QUERY_2_INACTIVIDAD)
print("\n--- Query 3: Métricas por Cáncer ---")
print(SQL_QUERY_3_METRICAS_CANCER)

# ============================================================================
# PARTE 2: CARGA DE DATOS DESDE EXCEL
# ============================================================================

print("\n" + "="*80)
print("CARGA DE DATOS")
print("="*80)

# Cargar las tres hojas del archivo Excel
archivo_excel = 'Dataset_prueba.xlsx'

df_pacientes = pd.read_excel(archivo_excel, sheet_name='Pacientes')
df_consultas = pd.read_excel(archivo_excel, sheet_name='Consultas')
df_laboratorios = pd.read_excel(archivo_excel, sheet_name='Laboratorios')

print(f"\n✓ Pacientes cargados: {df_pacientes.shape}")
print(f"✓ Consultas cargadas: {df_consultas.shape}")
print(f"✓ Laboratorios cargados: {df_laboratorios.shape}")

# ============================================================================
# PARTE 3: LIMPIEZA DE DATOS
# ============================================================================

print("\n" + "="*80)
print("LIMPIEZA DE DATOS")
print("="*80)

# 3.1 Estandarización de columnas categóricas
# Aplicar lowercase y luego Title Case a la columna 'motivo' en consultas
print("\n1. Estandarizando columna 'motivo' en Consultas...")
if 'motivo' in df_consultas.columns:
    df_consultas['motivo'] = df_consultas['motivo'].str.lower().str.title()
    print(f"   ✓ Valores únicos en motivo: {df_consultas['motivo'].unique()}")

# 3.2 Imputación de valores faltantes
# Imputar valores nulos de 'estadio' con la moda
print("\n2. Imputando valores nulos en 'estadio' con la moda...")
if 'estadio' in df_pacientes.columns:
    nulos_antes = df_pacientes['estadio'].isna().sum()
    if nulos_antes > 0:
        moda_estadio = df_pacientes['estadio'].mode()[0]
        df_pacientes['estadio'] = df_pacientes['estadio'].fillna(moda_estadio)
        print(f"   ✓ Nulos imputados: {nulos_antes} (moda: {moda_estadio})")
    else:
        print(f"   ✓ No hay valores nulos en 'estadio'")

# 3.3 Variable objetivo
# Eliminar filas con nulos en 'adherencia_12m'
print("\n3. Eliminando filas con nulos en 'adherencia_12m'...")
if 'adherencia_12m' in df_pacientes.columns:
    filas_antes = len(df_pacientes)
    df_pacientes = df_pacientes.dropna(subset=['adherencia_12m'])
    filas_eliminadas = filas_antes - len(df_pacientes)
    print(f"   ✓ Filas eliminadas: {filas_eliminadas}")
    print(f"   ✓ Filas restantes: {len(df_pacientes)}")

# ============================================================================
# PARTE 4: FEATURE ENGINEERING Y AGREGACIÓN
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING Y AGREGACIÓN")
print("="*80)

# 4.1 Contar consultas por paciente
print("\n1. Calculando conteo de consultas por paciente...")
df_count_consultas = df_consultas.groupby('id_paciente').size().reset_index(name='count_consultas')
print(f"   ✓ Agregado creado: {df_count_consultas.shape}")

# 4.2 Contar laboratorios por paciente
print("\n2. Calculando conteo de laboratorios por paciente...")
df_count_labs = df_laboratorios.groupby('id_paciente').size().reset_index(name='count_laboratorios')
print(f"   ✓ Agregado creado: {df_count_labs.shape}")

# 4.3 Calcular promedio de resultado_numerico por paciente y tipo de prueba
# Pivotar para crear columnas como avg_Biopsia, avg_VPH, etc.
print("\n3. Calculando promedio de resultado_numerico por tipo_prueba...")
if 'resultado_numerico' in df_laboratorios.columns and 'tipo_prueba' in df_laboratorios.columns:
    # Calcular promedio por paciente y tipo de prueba
    df_avg_labs = df_laboratorios.groupby(['id_paciente', 'tipo_prueba'])['resultado_numerico'].mean().reset_index()
    
    # Pivotar para crear columnas separadas por tipo de prueba
    df_avg_labs_pivot = df_avg_labs.pivot(
        index='id_paciente', 
        columns='tipo_prueba', 
        values='resultado_numerico'
    ).reset_index()
    
    # Renombrar columnas para añadir prefijo 'avg_'
    columnas_renombradas = {col: f'avg_{col}' for col in df_avg_labs_pivot.columns if col != 'id_paciente'}
    df_avg_labs_pivot.rename(columns=columnas_renombradas, inplace=True)
    
    print(f"   ✓ Promedio por tipo de prueba: {df_avg_labs_pivot.shape}")
    print(f"   ✓ Columnas creadas: {[col for col in df_avg_labs_pivot.columns if col != 'id_paciente']}")

# 4.4 Consolidar todo en un DataFrame único por paciente
print("\n4. Consolidando todos los agregados en df_final...")

# Comenzar con df_pacientes
df_final = df_pacientes.copy()

# Merge con count de consultas
df_final = df_final.merge(df_count_consultas, on='id_paciente', how='left')

# Merge con count de laboratorios
df_final = df_final.merge(df_count_labs, on='id_paciente', how='left')

# Merge con promedios de laboratorio pivotados
if 'resultado_numerico' in df_laboratorios.columns:
    df_final = df_final.merge(df_avg_labs_pivot, on='id_paciente', how='left')

# Rellenar NaN en conteos con 0 (pacientes sin consultas/labs)
df_final['count_consultas'] = df_final['count_consultas'].fillna(0)
df_final['count_laboratorios'] = df_final['count_laboratorios'].fillna(0)

print(f"   ✓ df_final creado: {df_final.shape}")
print(f"   ✓ Columnas totales: {len(df_final.columns)}")

# ============================================================================
# PARTE 5: ONE-HOT ENCODING
# ============================================================================

print("\n" + "="*80)
print("ONE-HOT ENCODING")
print("="*80)

print("\n1. Identificando columnas categóricas...")

# Identificar columnas categóricas (object o category dtype)
columnas_categoricas = df_final.select_dtypes(include=['object', 'category']).columns.tolist()

# Excluir columnas que no deben ser codificadas (por ejemplo, id_paciente si es string)
columnas_a_codificar = [col for col in columnas_categoricas if col != 'id_paciente']

print(f"   ✓ Columnas categóricas identificadas: {columnas_a_codificar}")

# 2. Aplicar One-Hot Encoding
print("\n2. Aplicando One-Hot Encoding...")

df_modelado_final = pd.get_dummies(
    df_final, 
    columns=columnas_a_codificar,
    drop_first=False,  # Mantener todas las categorías
    dtype=int  # Usar enteros (0, 1) en lugar de bool
)

print(f"   ✓ df_modelado_final creado: {df_modelado_final.shape}")
print(f"   ✓ Columnas después de encoding: {len(df_modelado_final.columns)}")

# ============================================================================
# PARTE 6: RESULTADO FINAL
# ============================================================================

print("\n" + "="*80)
print("DATAFRAME FINAL LISTO PARA MODELADO")
print("="*80)

print(f"\nShape: {df_modelado_final.shape}")
print(f"Filas: {df_modelado_final.shape[0]}")
print(f"Columnas: {df_modelado_final.shape[1]}")

print("\nInformación del DataFrame:")
print(df_modelado_final.info())

print("\nPrimeras 5 filas:")
print(df_modelado_final.head())

print("\nEstadísticas descriptivas (columnas numéricas):")
print(df_modelado_final.describe())

print("\n" + "="*80)
print("PIPELINE DE PROCESAMIENTO COMPLETADO")
print("="*80)
print("\nEl DataFrame 'df_modelado_final' está listo para entrenamiento de modelos ML")
print("Todas las variables categóricas han sido codificadas con One-Hot Encoding")
print("="*80)

# ============================================================================
# PARTE 7: MODELADO PREDICTIVO
# ============================================================================

print("\n" + "="*80)
print("MODELADO PREDICTIVO")
print("="*80)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# 7.1 Preparación: Separar features y target
print("\n1. Separando features (X) y target (y)...")

# La variable objetivo es 'adherencia_12m'
target_column = 'adherencia_12m'

# Verificar que la columna objetivo existe
if target_column not in df_modelado_final.columns:
    raise ValueError(f"La columna objetivo '{target_column}' no existe en el DataFrame")

# Separar X (features) y y (target)
y = df_modelado_final[target_column]
X = df_modelado_final.drop(columns=[target_column])

# Excluir columnas que no son features (id, fechas)
columnas_no_features = ['id_paciente', 'fecha_dx']
for col in columnas_no_features:
    if col in X.columns:
        X = X.drop(columns=[col])

# Rellenar valores NaN con 0 (para promedios de laboratorios faltantes)
X = X.fillna(0)

print(f"   ✓ Features (X): {X.shape}")
print(f"   ✓ Target (y): {y.shape}")
print(f"   ✓ Distribución de clases: \n{y.value_counts()}")

# 7.2 División en conjuntos de entrenamiento y prueba (80/20)
print("\n2. Dividiendo en train/test (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Mantener proporción de clases
)

print(f"   ✓ X_train: {X_train.shape}")
print(f"   ✓ X_test: {X_test.shape}")
print(f"   ✓ y_train: {y_train.shape}")
print(f"   ✓ y_test: {y_test.shape}")

# ============================================================================
# MODELO 1: RANDOM FOREST CLASSIFIER
# ============================================================================

print("\n" + "="*80)
print("MODELO 1: RANDOM FOREST CLASSIFIER")
print("="*80)

# Inicializar Random Forest
print("\n1. Inicializando Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
print(f"   ✓ Modelo inicializado: {rf_model}")

# Entrenar Random Forest
print("\n2. Entrenando Random Forest...")
rf_model.fit(X_train, y_train)
print(f"   ✓ Modelo entrenado exitosamente")

# Predicciones
print("\n3. Generando predicciones...")
y_pred_rf = rf_model.predict(X_test)
y_pred_rf_proba = rf_model.predict_proba(X_test)[:, 1]  # Probabilidades para clase positiva
print(f"   ✓ Predicciones generadas: {len(y_pred_rf)}")

# ============================================================================
# MODELO 2: RED NEURONAL SIMPLE (KERAS)
# ============================================================================

print("\n" + "="*80)
print("MODELO 2: RED NEURONAL SIMPLE (KERAS)")
print("="*80)

# Construir arquitectura de red neuronal
print("\n1. Construyendo arquitectura de red neuronal...")

nn_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), name='capa_entrada'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu', name='capa_oculta'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid', name='capa_salida')
])

print(f"   ✓ Red neuronal construida con 3 capas densas")
nn_model.summary()

# Compilar el modelo
print("\n2. Compilando modelo...")
nn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print(f"   ✓ Modelo compilado (optimizer=adam, loss=binary_crossentropy)")

# Entrenar la red neuronal
print("\n3. Entrenando red neuronal...")
history = nn_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0  # Sin output detallado por epoch
)
print(f"   ✓ Entrenamiento completado ({len(history.history['loss'])} epochs)")
print(f"   ✓ Loss final: {history.history['loss'][-1]:.4f}")
print(f"   ✓ Accuracy final (train): {history.history['accuracy'][-1]:.4f}")

# Predicciones de la red neuronal
print("\n4. Generando predicciones...")
y_pred_nn_proba = nn_model.predict(X_test, verbose=0).flatten()
y_pred_nn = (y_pred_nn_proba > 0.5).astype(int)
print(f"   ✓ Predicciones generadas: {len(y_pred_nn)}")

# ============================================================================
# EVALUACIÓN DE MODELOS
# ============================================================================

print("\n" + "="*80)
print("EVALUACIÓN DE MODELOS")
print("="*80)

def evaluar_modelo(y_true, y_pred, y_pred_proba, nombre_modelo):
    """
    Calcula y muestra métricas de evaluación para un modelo
    """
    print(f"\n--- {nombre_modelo} ---")
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calcular AUC solo si no hay NaN
    if np.isnan(y_pred_proba).any():
        auc = 0.0
        print(f"⚠ WARNING: Probabilidades contienen NaN, AUC = 0.0")
    else:
        auc = roc_auc_score(y_true, y_pred_proba)
    
    # Mostrar métricas
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    
    # Matriz de confusión
    print(f"\nMatriz de Confusión:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Reporte de clasificación
    print(f"\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

# Evaluar Random Forest
metricas_rf = evaluar_modelo(y_test, y_pred_rf, y_pred_rf_proba, "RANDOM FOREST")

# Evaluar Red Neuronal
metricas_nn = evaluar_modelo(y_test, y_pred_nn, y_pred_nn_proba, "RED NEURONAL")

# Comparación de modelos
print("\n" + "="*80)
print("COMPARACIÓN DE MODELOS")
print("="*80)

comparacion = pd.DataFrame({
    'Random Forest': metricas_rf,
    'Red Neuronal': metricas_nn
}).T

print("\n", comparacion)

# Determinar mejor modelo (por AUC)
if metricas_rf['auc'] >= metricas_nn['auc']:
    mejor_modelo = "Random Forest"
    mejor_auc = metricas_rf['auc']
else:
    mejor_modelo = "Red Neuronal"
    mejor_auc = metricas_nn['auc']

print(f"\n✓ Mejor modelo: {mejor_modelo} (AUC: {mejor_auc:.4f})")

# ============================================================================
# GUARDAR MODELO
# ============================================================================

print("\n" + "="*80)
print("GUARDANDO MODELO")
print("="*80)

# Guardar Random Forest (asumimos que es el mejor modelo)
modelo_filename = 'modelo_rf.pkl'

print(f"\n1. Guardando Random Forest como '{modelo_filename}'...")
with open(modelo_filename, 'wb') as f:
    pickle.dump(rf_model, f)

print(f"   ✓ Modelo guardado exitosamente")

# Verificar archivo guardado
import os
if os.path.exists(modelo_filename):
    file_size = os.path.getsize(modelo_filename) / 1024  # KB
    print(f"   ✓ Archivo: {modelo_filename} ({file_size:.2f} KB)")

# Guardar también los nombres de las features para validación en producción
feature_names = list(X.columns)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print(f"   ✓ Nombres de features guardados: feature_names.pkl")

print("\n" + "="*80)
print("PIPELINE COMPLETO FINALIZADO")
print("="*80)
print("\n✓ Procesamiento de datos completado")
print("✓ Modelos entrenados y evaluados")
print("✓ Random Forest guardado para despliegue")
print("✓ Listo para fase de deployment (FastAPI + Docker)")
print("="*80)

