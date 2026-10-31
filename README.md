# Proyecto Final - Inteligencia Artificial (Unidad 2)
## Sistema de Inferencia Probabilística y Comparativa con Modelos Supervisados (ML)
### Predicción de Intención de Compra en Comercio Electrónico (*Online Shoppers Purchasing Intention*)

---

## 📋 Descripción General

Este proyecto implementa y compara modelos de inferencia probabilística y aprendizaje supervisado para predecir si un visitante en un sitio de comercio electrónico completará una compra (`Revenue = True/False`), utilizando el dataset benchmark de UCI Machine Learning Repository (*Online Shoppers Purchasing Intention Dataset*).

El sistema cubre de forma rigurosa los 5 apartados de la Unidad 2:
1. **Apartado 1: Exploración y Representación del Conocimiento**: Estadísticas descriptivas, distribuciones de variables y mapa conceptual generado con NetworkX.
2. **Apartado 2: Teorema de Bayes por Conteos**: Estimación frecuentista exacta $P(\text{Compra}|\text{Visitante Recurrente})$, matriz de contingencia $2 \times 2$ y validación de la identidad matemática.
3. **Apartado 3: Red Naive Bayes y Tablas CPT con Suavizado de Laplace (+1)**: Construcción de grafo dirigido, cálculo de verosimilitudes con corrección por cero frecuencias y persistencia en JSON/CSV.
4. **Apartado 4: Inferencia Progresiva y Casos de Prueba**: Motor de inferencia en tres etapas incrementales y validación de casos de prueba obligatorios (parcial, completo y error controlado).
5. **Apartado 5: Comparativa de Modelos y Evaluación**: Random Forest (con balanceo de clases) y Perceptrón Multicapa (MLP con early stopping) comparados contra Naive Bayes bajo métricas F1, Precisión, Sensibilidad (Recall), matrices de confusión (Heatmaps) y comparador en vivo con soporte de evidencia incompleta.

---

## 🛠️ Requisitos e Instalación

### Requisitos Previos
- **Python**: Versión 3.10 o superior (probado en Python 3.14).
- **Tkinter**: Interfaz gráfica nativa de Python (en sistemas Linux basados en Debian/Ubuntu: `sudo apt install python3-tk`).

### Instalación de Dependencias

1. Clonar el repositorio y posicionarse en la carpeta raíz del proyecto:
   ```bash
   cd /ruta/hacia/Final
   ```

2. (Opcional pero recomendado) Crear y activar un entorno virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # En Linux / macOS
   # .venv\Scripts\activate     # En Windows
   ```

3. Instalar las dependencias listadas en `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Guía de Ejecución

Existen tres formas de operar el proyecto, según el flujo deseado:

### 🌟 Opción A: Panel Maestro Gráfico (Recomendada)
Para explorar todo el proyecto de manera centralizada mediante una interfaz gráfica integral:

```bash
python src/main.py
```
*(O alternativamente: `python src/main_dashboard.py`)*

Desde el Panel Maestro podrás:
- Supervisar en tiempo real el estado de todos los artefactos generados (`.csv`, `.joblib`, `.json`).
- Lanzar con un solo clic cualquiera de las interfaces gráficas de los 5 apartados.
- Ejecutar el pipeline de evaluación con un botón de acceso directo.

---

### ⚙️ Opción B: Ejecución Secuencial del Pipeline por Consola (CLI)

Si deseas regenerar desde cero los datos, calcular las probabilidades, entrenar los modelos y evaluar las métricas, ejecuta los siguientes scripts en orden:

1. **Reinicio y Purga del Entorno** *(Limpia artefactos previos y datasets derivados)*:
   ```bash
   python src/util/00_reset_entorno.py
   ```

2. **Inspección Exploratoria por Consola (Apartado 1)**:
   ```bash
   python src/analisisDatos/cli_resumen_datos.py
   ```

3. **Partición y Discretización de Datos (80% Train / 20% Test)**:
   ```bash
   python src/util/01_cli_preparar_datos.py
   ```
   *Genera los cortes P33 y P67 en train y produce los archivos categóricos para Bayes y numéricos (One-Hot Encoded) para ML.*

4. **Cálculo Analítico de Bayes por Conteos (Apartado 2)**:
   ```bash
   python src/bayes/cli_teorema_bayes.py
   ```
   *Muestra conteos de frecuencias, cálculo de $P(C|R)$ y verificación de identidad con $P_{\text{directa}}$.*

5. **Construcción de Red Naive Bayes y Tablas CPT con Laplace (Apartado 3)**:
   ```bash
   python src/redBayesiana/cli_pipeline_red_naive_bayes.py
   ```
   *Genera `tablas_cpt_laplace.json` y ejecuta la inferencia progresiva en 3 etapas.*

6. **Entrenamiento de Random Forest (Apartado 5)**:
   ```bash
   python src/modelosML/randoomForest/cli_entrenar_rf.py
   ```
   *Ajusta un ensamble balanceado con validación cruzada estratificada (5-Fold CV).*

7. **Entrenamiento de Perceptrón Multicapa - MLP (Apartado 5)**:
   ```bash
   python src/modelosML/mLP/cli_entrenar_mlp.py
   ```
   *Estandariza con `StandardScaler`, entrena con parada temprana (Early Stopping) y serializa el modelo.*

8. **Evaluación Comparativa en Conjunto de Prueba (Test 20%)**:
   ```bash
   python src/comparacion/cli_evaluar_todos.py
   ```
   *Evalúa Naive Bayes, Random Forest y MLP frente a la clase mayoritaria bajo un umbral de 0.50 y guarda `metricas_evaluacion.json`.*

---

### 🖥️ Opción C: Ejecución de Interfaces Gráficas Autónomas (GUI)

Cada vista del proyecto es un módulo desacoplado e independiente:

#### Apartado 1: Exploración y Representación
- **Mapa Conceptual Interactivo (NetworkX)**:
  ```bash
  python src/analisisDatos/mapa_conceptual.py
  ```
- **Distribución de Frecuencias y Percentiles**:
  ```bash
  python src/analisisDatos/distribucion_datos.py
  ```
- **Panel Integrado de Exploración**:
  ```bash
  python src/analisisDatos/exploracion_mapa.py
  ```

#### Apartado 2: Teorema de Bayes
- **Visualizador del Teorema de Bayes por Conteos**:
  ```bash
  python src/bayes/teorema_bayes.py
  ```

#### Apartado 3: Red Naive Bayes
- **Grafo Dirigido y Visor de Tablas CPT con Suavizado Laplace**:
  ```bash
  python src/redBayesiana/red_naive_bayes.py
  ```

#### Apartado 4: Demostración y Pruebas
- **Inferencia Progresiva y Casos de Prueba (TC-01, TC-02, TC-03)**:
  ```bash
  python src/redBayesiana/consultas_demostracion.py
  ```
- **Clasificador Interactivo Naive Bayes**:
  ```bash
  python src/redBayesiana/clasificador_bayes.py
  ```

#### Apartado 5: Comparativa y Predicción Multimodelo
- **Panel Comparativo y Matrices de Confusión (Heatmaps)**:
  ```bash
  python src/comparacion/comparativa_modelos.py
  ```
- **Comparador Multimodelo en Vivo (Manejo de Evidencia Incompleta)**:
  ```bash
  python src/comparacion/gui_comparativa_predicciones.py
  ```
  *Permite ingresar valores arbitrarios y contrastar el comportamiento de Naive Bayes (que marginaliza variables omitidas) frente a Random Forest y MLP (que exigen evidencia completa).*

---

## 🧪 Casos de Prueba Obligatorios

El sistema valida formalmente tres escenarios de prueba fundamentales según la especificación del proyecto (detallados en `docs/tabla_pruebas.md`):

| ID | Tipo de Caso | Entrada de Evidencia | Comportamiento Esperado |
| :--- | :--- | :--- | :--- |
| **TC-01** | **Consulta Parcial** | `VisitorType = "Returning_Visitor"`<br>`ProductRelated` omitido<br>`ProductRelated_Duration` omitido | Naive Bayes marginaliza las variables ausentes usando factor neutro ($1.0$), calculando $P(\text{Compra}) \approx 0.1398$. Los modelos ML señalan la ausencia de atributos obligatorios. |
| **TC-02** | **Consulta Completa** | `VisitorType = "Returning_Visitor"`<br>`ProductRelated = 35` (alto)<br>`ProductRelated_Duration = 1500.0` (alto) | Evaluación en 3 etapas progresivas para Naive Bayes ($P > 0.30$) y estimación directa con probabilidades en Random Forest y MLP. |
| **TC-03** | **Entrada Inválida** | `VisitorType = "Returning_Visitor"`<br>`ProductRelated = -5`<br>`ProductRelated_Duration = "duracion_invalida"` | Validación estricta con rechazo controlado mediante mensajes de error descriptivos sin colapsar el sistema. |

---

## 📁 Estructura del Repositorio

```text
Final/
├── artifacts/              # Artefactos persistidos (.json, .joblib, .csv, imágenes)
│   ├── cortes_percentiles.json
│   ├── tablas_cpt_laplace.json
│   ├── analisis_estadistico_cpt.csv
│   ├── random_forest.joblib
│   ├── mlp_model.joblib
│   ├── scaler_mlp.joblib
│   └── metricas_evaluacion.json
├── data/                   # Datasets original y particionados
│   ├── online_shoppers_intention.csv
│   ├── train_categorico.csv
│   ├── test_categorico.csv
│   ├── train_numerico.csv
│   └── test_numerico.csv
├── docs/                   # Documentación técnica
│   ├── guia_ejecucion.md   # Guía rápida de comandos
│   └── tabla_pruebas.md    # Especificación de casos de prueba
├── src/                    # Código fuente modular
│   ├── config.py           # Rutas centrales, constantes y semillas
│   ├── preprocessing.py    # Discretización y carga de percentiles
│   ├── main.py             # Alias de entrada al Dashboard Maestro
│   ├── main_dashboard.py   # Panel Maestro Gráfico (GUI)
│   ├── analisisDatos/      # Apartado 1: Estadísticas, mapa y distribuciones
│   ├── bayes/              # Apartado 2: Teorema de Bayes por conteos
│   ├── redBayesiana/       # Apartado 3 y 4: Red Naive Bayes, Laplace y pruebas
│   ├── modelosML/          # Apartado 5: Random Forest y MLP
│   ├── comparacion/        # Apartado 5: Evaluación y comparativa multimodelo
│   └── util/               # Scripts de utilería (reset y preparación de datos)
├── requirements.txt        # Dependencias de Python del proyecto
└── README.md               # Documentación principal del repositorio
```

---

## 👥 Autores y Créditos
- **Jaime / Cesar** — Inteligencia Artificial, Unidad 2.
