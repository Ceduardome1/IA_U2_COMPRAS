# Guía General de Ejecución del Proyecto

Esta guía describe el orden y propósito de los módulos CLI e interfaces GUI desarrolladas en el proyecto.

---

## 1. Módulos de Operación y Pipeline (CLI)

Ejecutar en orden secuencial para generar los datos y modelos necesarios:

1. **Purga y Reinicio**:
   ```bash
   python src/util/00_reset_entorno.py
   ```
2. **Preparación y Transformación de Datos**:
   ```bash
   python src/util/01_cli_preparar_datos.py
   ```
3. **Cálculo de Bayes por Conteos**:
   ```bash
   python src/bayes/cli_02_teorema_bayes.py
   ```
4. **Construcción de Red Naive Bayes y Tablas Laplace**:
   ```bash
   python src/redBayesiana/cli_03_red_naive_bayes.py
   ```
5. **Entrenamiento Random Forest**:
   ```bash
   python src/modelosML/randoomForest/02_cli_entrenar_rf.py
   ```
6. **Entrenamiento MLP con Early Stopping**:
   ```bash
   python src/modelosML/mLP/03_cli_entrenar_mlp.py
   ```
7. **Evaluación Comparativa en Test (Umbral 0.50)**:
   ```bash
   python src/comparacion/04_cli_evaluar_todos.py
   ```

---

## 2. Módulos Visuales Autónomos (GUI)

Cada interfaz es independiente y puede ejecutarse por separado:

* **Apartado 1 (Jaime): Exploración y Mapa Conceptual**:
  ```bash
  python src/analisisDatos/gui_01_exploracion_mapa.py
  ```
* **Apartado 2 (Jaime): Teorema de Bayes por Conteos**:
  ```bash
  python src/bayes/gui_02_teorema_bayes.py
  ```
* **Apartado 3 (Jaime): Grafo Naive Bayes y Tablas CPT con Laplace**:
  ```bash
  python src/redBayesiana/gui_03_red_naive_bayes.py
  ```
* **Apartado 4 (Demostración): Consultas Progresivas y Casos de Prueba**:
  ```bash
  python src/redBayesiana/gui_04_consultas_demostracion.py
  ```
* **Apartado 5 (Cesar): Comparativa Final y Matrices de Confusión**:
  ```bash
  python src/comparacion/gui_05_comparativa_modelos.py
  ```
