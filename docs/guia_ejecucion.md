# Guía General de Ejecución del Proyecto

Esta guía describe el orden y propósito de los módulos CLI e interfaces GUI desarrolladas en el proyecto.

---

## 1. Módulos de Operación y Pipeline (CLI)

Ejecutar en orden secuencial para generar los datos y modelos necesarios:

1. **Purga y Reinicio**:
   ```bash
   python src/util/00_reset_entorno.py
   ```
2. **Inspección Exploratoria por Consola (Apartado 1)**:
   ```bash
   python src/analisisDatos/cli_resumen_datos.py
   ```
3. **Preparación y Transformación de Datos**:
   ```bash
   python src/util/01_cli_preparar_datos.py
   ```
4. **Cálculo de Bayes por Conteos**:
   ```bash
   python src/bayes/cli_teorema_bayes.py
   ```
5. **Construcción de Red Naive Bayes y Tablas Laplace**:
   ```bash
   python src/redBayesiana/cli_pipeline_red_naive_bayes.py
   ```
6. **Entrenamiento Random Forest**:
   ```bash
   python src/modelosML/randoomForest/cli_entrenar_rf.py
   ```
7. **Entrenamiento MLP con Early Stopping**:
   ```bash
   python src/modelosML/mLP/cli_entrenar_mlp.py
   ```
8. **Evaluación Comparativa en Test (Umbral 0.50)**:
   ```bash
   python src/comparacion/cli_evaluar_todos.py
   ```

---

## 2. Lanzador Maestro y Módulos Visuales (GUI)

### 🌟 Panel Maestro de Navegación (Recomendado)
Permite navegar y lanzar visualmente todas las interfaces del proyecto desde una sola ventana:
```bash
python src/main_dashboard.py
# O su alias directo:
python src/main.py
```

---

### Módulos Visuales Autónomos por Apartado

Cada interfaz es independiente y puede ejecutarse por separado:

* **Apartado 1 (Jaime / Cesar): Exploración de Datos y Mapa Conceptual**:
  - **Mapa Conceptual Dedicado (NetworkX)**:
    ```bash
    python src/analisisDatos/mapa_conceptual.py
    ```
  - **Distribución de Datos Dedicada**:
    ```bash
    python src/analisisDatos/distribucion_datos.py
    ```
  - **Panel Integrado (Ambas Vistas y Lanzadores)**:
    ```bash
    python src/analisisDatos/exploracion_mapa.py
    ```

* **Apartado 2 (Jaime): Teorema de Bayes por Conteos**:
  ```bash
  python src/bayes/teorema_bayes.py
  ```

* **Apartado 3 (Jaime): Grafo Naive Bayes y Tablas CPT con Laplace**:
  *(Incluye persistencia JSON/CSV y Visor de Matriz CPT Completa)*
  ```bash
  python src/redBayesiana/red_naive_bayes.py
  ```

* **Apartado 4 (Demostración): Consultas Progresivas y Casos de Prueba**:
  ```bash
  python src/redBayesiana/consultas_demostracion.py
  ```
  *(Clasificador complementario simple: `python src/redBayesiana/gui_04_clasificador_bayes.py`)*

* **Apartado 5 (Cesar): Comparativa Final y Evaluación**:
  - **Panel Comparativo y Matrices de Confusión Heatmaps**:
    ```bash
    python src/comparacion/comparativa_modelos.py
    ```
  - **Comparador Multimodelo en Vivo (Manejo de Evidencia Incompleta)**:
    ```bash
    python src/comparacion/gui_comparativa_predicciones.py
    ```
