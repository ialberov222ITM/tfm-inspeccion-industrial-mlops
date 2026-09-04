# 🏭 Gemelo Digital: Inspección de Calidad en Superficies Metálicas

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)](#)

**Trabajo de Fin de Máster - Máster en Inteligencia Artificial**

Sistema MLOps completo para detección, clasificación y explicabilidad de defectos en superficies metálicas mediante Deep Learning, Transfer Learning, Visión Artificial Explicable (XAI) y monitoreo en tiempo real con rutas **100% portables**.

---

## 📋 Tabla de Contenidos

- [✨ Características Principales](#-características-principales)
- [🎥 Demostración](#-demostración)
- [📂 Estructura del Repositorio](#-estructura-del-repositorio)
- [🚀 Arquitectura de Versiones](#-arquitectura-de-versiones)
- [🔧 Sistema de Rutas Portables](#-sistema-de-rutas-portables)
- [📥 Instalación](#-instalación)
- [🎯 Inicio Rápido](#-inicio-rápido)
- [🔧 Uso Avanzado](#-uso-avanzado)
- [📊 Dashboard MLOps](#-dashboard-mlops)
- [🧠 Explicabilidad (XAI - Grad-CAM)](#-explicabilidad-xai---grad-cam)
- [⚠️ Solución de Problemas](#-solución-de-problemas)
- [📚 Referencia Técnica](#-referencia-técnica)
- [👤 Autor](#-autor)

---

## ✨ Características Principales

### 🤖 **Inteligencia Artificial**
- **3 Arquitecturas Comparativas:** ResNet50, MobileNetV2, EfficientNetB0
- **Transfer Learning:** Adaptación a defectología de superficies metálicas
- **Consenso Multi-Modelo:** Decisión de planta por mayoría de 3 modelos
- **Inferencia en Tiempo Real:** <500ms por imagen (CPU compatible)

### 📊 **Explicabilidad (XAI)**
- **Grad-CAM:** Visualización térmica del campo receptivo de la red
- **Auditoría Visual:** Identificación de zonas críticas en decisiones
- **Trazabilidad Completa:** Cada predicción linkea a la imagen de origen

### 🏭 **MLOps y Productivización**
- **Simulador de Planta en Vivo:** 5 líneas de producción paralelas
- **Dashboard de Calidad:** KPIs en tiempo real (FP, FN, Tasa de Defectos)
- **Base de Datos SQLite:** Auditoría permanente de decisiones
- **Comparativa de Modelos:** Visualización side-by-side de predicciones

### 🔄 **Reproducibilidad y Portabilidad**
- **Rutas Dinámicas:** Funciona en cualquier equipo sin cambios manuales
- **Detección Automática:** Adapta modo operación según disponibilidad de datos
- **Dos Versiones Independientes:** Evaluación (GitHub) vs Producción (Local)
- **Mensajes Informativos:** Sidebar muestra status automático del entorno

---

## 🎥 Demostración

Para ver el sistema en acción, consulta la demostración interactiva:

📺 **[Ver vídeo demostrativo en YouTube](https://youtu.be/QClQ1FfFdSk)**

---

## 📂 Estructura del Repositorio

```
📦 TFM_MetalesSinteticos/
│
├── 📁 data/
│   └── sample_images/          # Dataset de muestra (defectos sintéticos)
│       ├── crack/
│       ├── hole/
│       ├── normal/
│       ├── rust/
│       └── scratch/
│
├── 📁 models/                   # Pesos entrenados
│   ├── modelo_optimo_resnet50.keras       # Descargar de GitHub Releases (v1.0.0)
│   ├── modelo_optimo_mobilenetv2.keras    # ✅ Incluido en repositorio
│   └── modelo_optimo_efficientnetb0.keras # ✅ Incluido en repositorio
│
├── 📁 notebooks/                # Scripts principales y cuadernos Jupyter
│   ├── 01_eda_dataset.ipynb
│   ├── 02_data_augmentation.ipynb
│   ├── 03_entrenamiento_comparativo_tl.ipynb
│   ├── App_planta_industrial.py              # Versión Evaluación (GitHub - Ligera)
│   ├── App_planta_industrialCompleta.py      # Versión Completa (Producción Local)
│   └── 04_gradcam_autocalibrado.py           # Script de auditoría XAI en lote
│
├── 📁 results/                  # Evidencias visuales
│   ├── confusion_matrices/      # Matrices de confusión por modelo
│   ├── gradcam_audit/           # Reportes Grad-CAM seleccionados
│   └── reporte_gradcam/         # Salida automatizada de auditoría
│
├── 📄 requirements.txt           # Dependencias Python
├── 📄 README.md                  # Este archivo (raíz del proyecto)
└── 📄 LICENSE                    # MIT License

```

**Nota:** `industrial_defect_dataset/val/` no está en GitHub (demasiado pesado). Se descarga por separado o se agregará localmente.

---

## 🚀 Arquitectura de Versiones

El proyecto incluye **dos versiones independientes** optimizadas para diferentes contextos:

### 1. 🟢 Versión de Evaluación (RECOMENDADA PARA GITHUB)

**Archivo:** `src/App_planta_industrialDemo.py`

**Características:**
- ✅ Dataset de muestra reducido (`data/sample_images/`)
- ✅ Base de datos ligera (`produccion_planta_tfm.db`)
- ✅ Tamaño total: < 500 MB
- ✅ Reproducible inmediatamente
- ✅ Ideal para tribunal evaluador

**Ejecución:**
```bash
streamlit run src/App_planta_industrialDemo.py
```

---

### 2. 🔵 Versión Completa (ENTORNO DE PRODUCCIÓN LOCAL)

**Archivo:** `src/App_planta_industrialProduccion.py`

**Características:**
- ✅ Dataset masivo local (miles de imágenes)
- ✅ Base de datos completa (`produccion_planta.db`)
- ✅ Almacenamiento persistente
- ✅ Detección automática de dataset
- ✅ Mensajes informativos en sidebar

**Ejecución:**
```bash
streamlit run src/App_planta_industrialProduccion.py
```

---

### 📋 Diferencia Estructural de Versiones

| Aspecto | Evaluación (GitHub) | Completa (Local) |
|--------|-------------------|------------------|
| **Archivo** | `App_planta_industrialDemo.py` | `App_planta_industrialProduccion.py` |
| **Dataset** | `data/sample_images/` (muestra) | `industrial_defect_dataset/val/` (si existe) |
| **BD Principal** | `produccion_planta_tfm.db` | `produccion_planta.db` |
| **Tamaño Total** | < 500 MB | +10 GB |
| **Persistencia** | Ligera | Completa |
| **Target** | Tribunal evaluador | Desarrollo/Demostrativo |

---

## 🔧 Sistema de Rutas Portables

### ⭐ **Cómo Funciona la Portabilidad**

Ambas versiones utilizan **rutas dinámicas calculadas en tiempo de ejecución**, lo que garantiza que funcionan en cualquier equipo sin cambios manuales.

**Lógica de Rutas:**

```python
# BASE_DIR se calcula automáticamente (2 niveles arriba del script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Todas las rutas se construyen relativamente a BASE_DIR
DATASET_PATH = os.path.join(BASE_DIR, "industrial_defect_dataset", "val")
DB_FILE = os.path.join(BASE_DIR, "produccion_planta.db")
MODELS_PATH = os.path.join(BASE_DIR, "models", "*.keras")
```

### 🎯 **Detección Automática de Dataset**

La versión completa detecta automáticamente si el dataset masivo está disponible:

```python
DATASET_EXISTS = os.path.exists(DATASET_PATH)

if DATASET_EXISTS:
    # Muestra: 🟢 "Modo LOCAL (Dataset Completo)"
    # Usa dataset masivo local
else:
    # Muestra: 🟡 "Modo EVALUACIÓN (Dataset de Muestra)"
    # Fallback a data/sample_images/ (si es necesario)
```

**Resultado en Sidebar:**

- **🟢 Modo LOCAL:** Dataset masivo encontrado → Simulación con miles de imágenes
- **🟡 Modo EVALUACIÓN:** Dataset no encontrado → Advertencia + instrucciones

### 📍 **Estructura Esperada en Diferentes Contextos**

**En Tu Local (Desarrollo):**
```
TFM_MetalesSinteticos/
├── industrial_defect_dataset/val/  ✅ Existe
├── models/
├── data/sample_images/
└── notebooks/
    └── App_planta_industrialCompleta.py
        → Detecta dataset masivo automáticamente
        → Muestra: 🟢 "Modo LOCAL"
```

**En GitHub / Otro Ordenador (Sin Dataset Masivo):**
```
TFM_MetalesSinteticos/
├── models/
├── data/sample_images/             ✅ Incluido en GitHub
└── notebooks/
    └── App_planta_industrialCompleta.py
        → Dataset masivo no existe
        → Muestra: 🟡 "Modo EVALUACIÓN"
        → Instrucciones en sidebar para agregar dataset
```

**Si Agregan Dataset en Otro Ordenador:**
```
TFM_MetalesSinteticos/
├── industrial_defect_dataset/val/  ✅ Usuario agregó
├── models/
├── data/sample_images/
└── notebooks/
    └── App_planta_industrialCompleta.py
        → Auto-detecta la nueva carpeta
        → Auto-cambia a: 🟢 "Modo LOCAL"
        → Sin necesidad de reiniciar o cambiar código
```

---

## 📥 Instalación

### Paso 1: Clona el Repositorio

```bash
git clone https://github.com/TU_USUARIO/TFM_MetalesSinteticos.git
cd TFM_MetalesSinteticos
```

### Paso 2: Crea un Entorno Virtual (Recomendado)

```bash
# Con Python 3.8+
python -m venv venv

# Activar entorno
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### Paso 3: Instala Dependencias

```bash
pip install -r requirements.txt
```

**Contenido esperado de `requirements.txt`:**
```
tensorflow==2.14.0
streamlit==1.59.0
pandas==2.0.0
numpy==1.24.0
Pillow==10.0.0
plotly==5.17.0
matplotlib==3.8.0
scikit-learn==1.3.0
```

### Paso 4: Descarga los Modelos Entrenados

> ⚠️ **IMPORTANTE:** Los modelos `.keras` son pesados (~350 MB combinados)

**Opción A:** Descargar desde GitHub Releases (Recomendado)

ResNet50 (archivo grande) se descarga desde GitHub Releases:
```bash
cd models
# Descargar ResNet50 desde Releases
wget https://github.com/ialberov222ITM/tfm-inspeccion-industrial-mlops/releases/download/v1.0.0/modelo_optimo_resnet50.keras
cd ..
```

MobileNetV2 y EfficientNetB0 ya están incluidos en el repositorio:
```
models/
├── modelo_optimo_resnet50.keras          # Descargar de Releases (arriba)
├── modelo_optimo_mobilenetv2.keras       # ✅ Incluido en repositorio
└── modelo_optimo_efficientnetb0.keras    # ✅ Incluido en repositorio
```

**Opción B:** Re-entrenar localmente
```bash
cd notebooks
jupyter notebook 03_entrenamiento_comparativo_tl.ipynb
```

---

## 🎯 Inicio Rápido

### Para Evaluadores (Versión Ligera - GitHub)

```bash
# 1. Instala (ver sección anterior)
pip install -r requirements.txt

# 2. Ejecuta directamente
streamlit run src/App_planta_industrialDemo.py

# 3. Abre en el navegador
# http://localhost:8501
```

**Tiempo esperado:**
- ⏱️ Carga de modelos: ~10-15 segundos
- ⏱️ Primera predicción: <1 segundo
- 💾 Espacio en disco: ~500 MB

---

### Para Desarrolladores (Versión Completa Local)

```bash
# 1. Instala como arriba
pip install -r requirements.txt

# 2. Ejecuta
streamlit run src/App_planta_industrialProduccion.py

# 3. La app automáticamente:
# - Detecta si existe industrial_defect_dataset/val/
# - Muestra 🟢 "Modo LOCAL" si existe
# - Muestra 🟡 "Modo DEMO" si no existe
```

**Con Dataset Masivo Local:**
```
✅ Verifica que tienes: TFM_MetalesSinteticos/industrial_defect_dataset/val/
   - crack/
   - hole/
   - normal/
   - rust/
   - scratch/

✅ La app detectará automáticamente y mostrará: 🟢 "Modo LOCAL"
```

---

## 🔧 Uso Avanzado

### Script de Auditoría XAI en Lote

Genera reportes Grad-CAM automáticos para un conjunto de imágenes:

```bash
cd notebooks
python 04_gradcam_autocalibrado.py
```

**Salida:**
```
results/
└── reporte_gradcam/
    ├── crack/
    ├── hole/
    ├── normal/
    ├── rust/
    └── scratch/
```

---

### Personalización del Dataset Local

Para usar tu propio dataset masivo:

1. **Estructura esperada:**
   ```
   industrial_defect_dataset/val/
   ├── crack/
   │   ├── crack_001.jpg
   │   └── ...
   ├── hole/
   ├── normal/
   ├── rust/
   └── scratch/
   ```

2. **Ubicación:**
   ```
   TFM_MetalesSinteticos/
   ├── industrial_defect_dataset/val/  ← Coloca aquí
   ├── models/
   ├── data/
   └── notebooks/
   ```

3. **Reinicia Streamlit (o recarga la página):**
   ```bash
   streamlit run src/App_planta_industrialProduccion.py
   ```

   La app detectará automáticamente el nuevo dataset.

---

## 📊 Dashboard MLOps

### Página 1: Simulador de Planta (En Vivo)

**Propósito:** Emular una línea de producción industrial real

**Funcionalidades:**
- 5 líneas de producción paralelas
- Generación aleatoria de piezas (defectuosas vs. normales)
- Evaluación comparativa de 3 modelos por pieza
- Consenso de decisión (rechazo si 2+ modelos votan rechazo)
- Visualización Grad-CAM side-by-side

**Controles:**
- ▶️ **Toggle:** Activar/detener simulación
- 🎛️ **Slider:** Configurar tasa de defectos objetivo (5%-70%)
- 🗑️ **Botón:** Resetear base de datos

---

### Página 2: Dashboard de Resultados

**Propósito:** Análisis de calidad y auditoría post-simulación

**Métricas Principales:**
- Total de piezas analizadas
- Tasa de defectos real vs. objetivo
- Falsos Positivos / Falsos Negativos
- Errores de Clasificación

**Visualizaciones:**
- 📈 Distribución de defectos (pie chart)
- 📊 Rendimiento por línea (bar chart)
- 🔍 Tabla de auditoría con filtro por modelo
- 📉 Comparativa de fiabilidad entre modelos

---

### Página 3: Inspección Manual (Subir Imagen)

**Propósito:** Evaluar una imagen de prueba con los 3 modelos

**Pasos:**
1. Haz clic en **"Cargar imagen"** 📤
2. Selecciona una imagen (JPG/PNG)
3. El sistema evalúa automáticamente con los 3 modelos
4. Compara resultados side-by-side
5. Descarga Grad-CAM de cada modelo

---

## 🧠 Explicabilidad (XAI - Grad-CAM)

### ¿Qué es Grad-CAM?

**Gradient-weighted Class Activation Mapping** es una técnica de XAI que:
1. Calcula gradientes de la clase predicha respecto a la última capa convolucional
2. Pondera cada mapa de características por su gradiente
3. Genera un mapa térmico que muestra **dónde se fijó la red** para decidir

### Visualización Interactiva

En el simulador y la inspección manual verás:

```
┌─────────────────────────────────────────┐
│ 📷 Imagen Original    🧠 Grad-CAM       │
│ (Superficie real)     (Zona crítica)    │
│                                         │
│ Región roja/naranja = Alta importancia  │
│ Región azul = Baja importancia          │
└─────────────────────────────────────────┘
```

### Auditoría en Lote

Para generar reportes visuales de múltiples imágenes:

```bash
python src/Gradcam_autocalibrado.py
```

---

## ⚠️ Solución de Problemas

### ❌ Error: "ModuleNotFoundError: No module named 'tensorflow'"

**Solución:**
```bash
pip install tensorflow==2.14.0
```

---

### ❌ Error: "No se encontraron imágenes en './data/sample_images'"

**Causa:** El dataset de muestra no está en la ruta esperada

**Soluciones:**
1. **Verifica la estructura:**
   ```bash
   ls data/sample_images/  # macOS/Linux
   dir data\sample_images\ # Windows
   ```

2. **Si no existe, descárgalo:**
   - Desde Google Drive: [Enlace al dataset](#)
   - O re-ejecuta el EDA: `notebooks/01_eda_dataset.ipynb`

---

### ❌ Error: "Los modelos .keras no se encontraron"

**Causa:** Los pesos no fueron descargados

**Solución:**
1. Descarga desde Google Drive: [Enlace a modelos](#)
2. Coloca en `models/`:
   ```
   models/
   ├── modelo_optimo_resnet50.keras
   ├── modelo_optimo_mobilenetv2.keras
   └── modelo_optimo_efficientnetb0.keras
   ```

---

### ⚠️ "Modo EVALUACIÓN" en sidebar pero tengo dataset local

**Causa:** El script no encuentra `industrial_defect_dataset/val/` en la ruta esperada

**Soluciones:**
1. Verifica que la carpeta esté en `TFM_MetalesSinteticos/` (no dentro de `notebooks/`)
2. Recarga la página de Streamlit (Ctrl+R)
3. Si sigue sin detectarse, verifica permisos de lectura

---

### 🐢 Aplicación lenta (primeras predicciones >5 segundos)

**Normal en primera ejecución:**
- TensorFlow necesita compilar kernels CUDA/CPU
- Espera de 10-15 segundos en inicio
- Predicciones posteriores serán < 500ms

---

## 📚 Referencia Técnica

### Arquitecturas de Modelos

| Modelo | Parámetros | Velocidad | Precisión | Ideal para |
|--------|-----------|-----------|-----------|-----------|
| **ResNet50** | 23.6M | Medio | ⭐⭐⭐⭐⭐ | Máxima precisión |
| **MobileNetV2** | 3.5M | ⚡ Rápido | ⭐⭐⭐⭐ | Producción en tiempo real |
| **EfficientNetB0** | 5.3M | Rápido | ⭐⭐⭐⭐⭐ | Balance óptimo |

**Consenso:** Rechazo si ≥2 de 3 modelos votan "Defecto"

---

### Base de Datos SQLite

**Esquema de `produccion_planta.db` / `produccion_planta_tfm.db`:**

```sql
CREATE TABLE produccion (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    linea INTEGER,                    -- 1-5 (línea de producción)
    defecto TEXT,                     -- Predicción: Crack, Hole, Normal, Rust, Scratch
    confianza REAL,                   -- 0-100 (%)
    estado TEXT,                      -- Aceptada / Rechazada (Defecto)
    clase_real TEXT,                  -- Ground truth del dataset
    estado_real TEXT,                 -- Ground truth: Aceptada / Rechazada
    modelo_ia TEXT,                   -- ResNet50 / MobileNetV2 / EfficientNetB0
    tasa_defectos_objetivo REAL,      -- % configurado en slider
    nombre_fichero TEXT,              -- Ruta relativa en dataset
    pieza_id TEXT UNIQUE,             -- ID de la pieza física (agrupa 3 evaluaciones)
    estado_consenso TEXT              -- Decisión final de planta
);
```

---

### Clases de Defectos

```python
CLASS_NAMES = {
    0: 'Crack (Grieta)',           # Fractura superficial
    1: 'Hole (Perforación)',       # Orificio
    2: 'Normal (Sin defectos)',    # Superficie sana
    3: 'Rust (Óxido)',             # Oxidación / corrosión
    4: 'Scratch (Arañazo)'         # Marca superficial
}
```

---

### Sistema de Rutas (Detalle Técnico)

**Cálculo de BASE_DIR:**
```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#                          ↑ Nivel 1: notebooks/
#                                              ↑ Nivel 2: TFM_MetalesSinteticos/
```

**Construcción de rutas:**
```python
DATASET_PATH = os.path.join(BASE_DIR, "industrial_defect_dataset", "val")
#             └─ Resulta en: TFM_MetalesSinteticos/industrial_defect_dataset/val

DB_FILE = os.path.join(BASE_DIR, "produccion_planta.db")
#         └─ Resulta en: TFM_MetalesSinteticos/produccion_planta.db

MODELS_PATH = os.path.join(BASE_DIR, "models", "*.keras")
#            └─ Resulta en: TFM_MetalesSinteticos/models/*.keras
```

**Por qué es portable:**
- No importa dónde esté `notebooks/` dentro del proyecto
- No importa dónde esté el proyecto en el disco
- Las rutas se construyen **relativamente** desde la ubicación del script
- Funciona idéntico en Windows, macOS, Linux

---

## 👤 Autor

**Ismael Albero Verdú**

- 📧 Email: `ialberov222@gmail.com`
- 🎓 Máster en Inteligencia Artificial
- 🔬 Especialización: MLOps, Deep Learning, Visión Artificial Explicable (XAI)

---

## 📜 Licencia

Este proyecto está bajo licencia **MIT**. Ver archivo `LICENSE` para más detalles.

---

## 🙏 Agradecimientos

- Universidad: Por la formación en IA y metodología de investigación
- Tribunal Evaluador: Por la retroalimentación y supervisión
- Comunidad Open Source: TensorFlow, Streamlit, pandas, scikit-learn

---

## 📞 Contacto y Soporte

¿Preguntas o problemas?

1. **Issues en GitHub:** Crea un issue detallando el problema
2. **Email:** `ialberov222@gmail.com`
3. **Troubleshooting:** Ver sección [⚠️ Solución de Problemas](#-solución-de-problemas)

---

**Última actualización:** Agosto 2026 | **Versión:** 1.1.0 (Rutas Portables)
