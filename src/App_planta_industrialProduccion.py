
import os
import io
# Estas variables deben fijarse ANTES de "import tensorflow": sus logs de arranque
# (CUDA, oneDNN, instrucciones de CPU) se emiten en el momento del import, así que
# configurarlas después no tiene ningún efecto sobre esos mensajes.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'       # silencia logs INFO/WARNING del backend C++ de TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'      # evita el aviso de oneDNN sobre variaciones numéricas
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'      # sin GPU en este equipo: que ni intente inicializar CUDA

import warnings
warnings.filterwarnings("ignore", message=".*chardet or charset_normalizer.*")
# Aviso benigno de Keras: el modelo tiene una entrada nombrada ("input_rgb") y el
# Grad-CAM la invoca con un tensor plano en vez de un dict; Keras hace el match por
# posición igualmente y el resultado es correcto, solo avisa de la discrepancia de forma.
warnings.filterwarnings("ignore", message=".*structure of `inputs` doesn't match.*")

import time
import random
import glob
import sqlite3
import uuid
from collections import Counter
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib

tf.get_logger().setLevel('ERROR')
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold('error')
except ImportError:
    pass

# ==========================================
# 0. CONFIGURACIÓN INICIAL Y ESTILO
# ==========================================
st.set_page_config(page_title="Planta Industrial IA", page_icon="🏭", layout="wide")

# CSS de compactación: recorta paddings y tamaños de fuente para que tanto el
# contenido principal como la barra lateral quepan en una única pantalla, sin scroll.
# Las reglas "grandes" (para el Simulador) se aplican solo dentro de .main, y la barra
# lateral tiene su propio set de reglas, más compacto, para que no herede esos tamaños.
st.markdown("""
<style>
    /* --- Contenido principal --- */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
    }
    .main h1 {
        font-size: 2.4rem !important;
        margin-bottom: 0rem !important;
        margin-top: 0rem !important;
    }
    .main [data-testid="stVerticalBlockBorderWrapper"] {
        margin: 0rem !important;
    }
    .main h3 {
        font-size: 1.8rem !important;
        margin-bottom: 0.4rem !important;
        margin-top: 0.4rem !important;
    }
    .main div[data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.4rem;
        font-size: 1.4rem;
    }
    .main [data-testid="stMetric"] {
        padding: 0.4rem 0.8rem;
    }
    .main [data-testid="stMetricValue"] {
        font-size: 2.2rem;
    }
    .main [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
    }
    .main .stAlert {
        padding: 0.6rem 1.2rem;
        margin-bottom: 0.6rem;
        font-size: 1.2rem;
    }
    .main hr {
        margin: 0.6rem 0 !important;
    }
    div[data-testid="stImage"] img {
        max-height: 360px;
        object-fit: contain;
    }
    .main div[data-testid="stImage"] figcaption {
        font-size: 1.1rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0.6rem;
    }
    .main button[data-testid="stBaseButton-secondary"] p,
    .main label[data-testid="stWidgetLabel"] p {
        font-size: 1.2rem;
    }

    /* --- Barra lateral: compacta, independiente del tamaño del contenido principal --- */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.3rem !important;
        margin-bottom: 0.2rem !important;
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 1rem !important;
        margin-top: 0.3rem !important;
        margin-bottom: 0.1rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        font-size: 0.85rem;
        margin-bottom: 0.1rem;
    }
    section[data-testid="stSidebar"] hr {
        margin: 0.4rem 0 !important;
    }
    section[data-testid="stSidebar"] .stAlert {
        padding: 0.3rem 0.6rem;
        font-size: 0.8rem;
    }
    section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] button p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONFIGURACIÓN DE RUTAS (Portable + Local)
# ==========================================
# Este script está diseñado para ser PORTABLE: funciona tanto en desarrollo local
# como en otros ordenadores descargados desde GitHub. Las rutas se calculan
# dinámicamente usando BASE_DIR (dos niveles arriba del script) para que sean
# relativamente independientes de dónde esté el script dentro del proyecto.
#
# Estructura esperada (mismo nivel que BASE_DIR):
# ├── industrial_defect_dataset/val/  ← Dataset masivo local (si existe)
# ├── data/sample_images/              ← Dataset de muestra (fallback en GitHub)
# └── models/                           ← Modelos entrenados
#
# Si DATASET_EXISTS = False, muestra advertencia y instructions en la sidebar.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "industrial_defect_dataset", "val")
DB_FILE = os.path.join(BASE_DIR, "produccion_planta.db")

DATASET_EXISTS = os.path.exists(DATASET_PATH)
MODO_OPERACION = "🟢 LOCAL (Dataset masivo)" if DATASET_EXISTS else "🟡 EVALUACIÓN (Dataset de muestra)"

CLASS_NAMES = {0: 'Crack (Grieta)', 1: 'Hole (Perforación)', 2: 'Normal (Sin defectos)', 3: 'Rust (Óxido)', 4: 'Scratch (Arañazo)'}
FOLDER_TO_CLASS_NAME = {'crack': CLASS_NAMES[0], 'hole': CLASS_NAMES[1], 'normal': CLASS_NAMES[2], 'rust': CLASS_NAMES[3], 'scratch': CLASS_NAMES[4]}

MODEL_CONFIGS = {
    "EfficientNetB0": {"path": os.path.join(BASE_DIR, "models", "modelo_optimo_efficientnetb0.keras"), "last_conv_layer": "top_activation"},
    "ResNet50": {"path": os.path.join(BASE_DIR, "models", "modelo_optimo_resnet50.keras"), "last_conv_layer": "conv5_block3_out"},
    "MobileNetV2": {"path": os.path.join(BASE_DIR, "models", "modelo_optimo_mobilenetv2.keras"), "last_conv_layer": "out_relu"},
}
TARGET_SIZE = (224, 224)
DURACION_SIMULACION_SEG = 60 * 60

# ==========================================
# 1. GESTIÓN DE BASE DE DATOS (SQLITE)
# ==========================================
def init_db():
    """Inicializa la base de datos y crea la tabla si no existe."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS produccion (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            linea INTEGER,
            defecto TEXT,
            confianza REAL,
            estado TEXT
        )
    ''')
    # Migración: añade las columnas de etiqueta real y modelo usado (necesarias para
    # calcular falsos positivos/negativos y filtrar por modelo) si la BD es de una versión anterior.
    columnas_existentes = {fila[1] for fila in c.execute("PRAGMA table_info(produccion)").fetchall()}
    if 'clase_real' not in columnas_existentes:
        c.execute("ALTER TABLE produccion ADD COLUMN clase_real TEXT")
    if 'estado_real' not in columnas_existentes:
        c.execute("ALTER TABLE produccion ADD COLUMN estado_real TEXT")
    if 'modelo_ia' not in columnas_existentes:
        c.execute("ALTER TABLE produccion ADD COLUMN modelo_ia TEXT")
    if 'tasa_defectos_objetivo' not in columnas_existentes:
        c.execute("ALTER TABLE produccion ADD COLUMN tasa_defectos_objetivo REAL")
    if 'imagen_original' not in columnas_existentes:
        c.execute("ALTER TABLE produccion ADD COLUMN imagen_original BLOB")
    if 'imagen_gradcam' not in columnas_existentes:
        c.execute("ALTER TABLE produccion ADD COLUMN imagen_gradcam BLOB")
    if 'nombre_fichero' not in columnas_existentes:
        c.execute("ALTER TABLE produccion ADD COLUMN nombre_fichero TEXT")
    if 'pieza_id' not in columnas_existentes:
        # Agrupa las 3 filas (una por modelo) que corresponden a la MISMA pieza física, para
        # poder contar piezas producidas de verdad (deduplicando) en vez de filas de la tabla.
        c.execute("ALTER TABLE produccion ADD COLUMN pieza_id TEXT")
    if 'estado_consenso' not in columnas_existentes:
        # Decisión real de la planta (mayoría de los 3 modelos), igual en las 3 filas de una
        # misma pieza. Antes esta decisión solo vivía en memoria y nunca se guardaba en la BD.
        c.execute("ALTER TABLE produccion ADD COLUMN estado_consenso TEXT")
    conn.commit()
    conn.close()

def imagen_a_bytes(img_pil):
    """Codifica una imagen PIL a bytes PNG, para poder guardarla en una columna BLOB de SQLite."""
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return buffer.getvalue()

def insertar_registro(linea, defecto, confianza, estado, clase_real, estado_real, modelo_ia,
                       tasa_defectos_objetivo, imagen_original_bytes=None, imagen_gradcam_bytes=None,
                       nombre_fichero=None, pieza_id=None, estado_consenso=None):
    """Inserta la evaluación de UN modelo sobre una pieza en la base de datos, junto con su
    etiqueta real (ground truth del dataset), el modelo de IA que la clasificó, la tasa de
    defectos objetivo (%) configurada en el slider en el momento de generarla, las imágenes
    (original + Grad-CAM de ESE modelo), el nombre/ruta del fichero original del dataset, y
    dos campos compartidos por las 3 filas de la misma pieza física: `pieza_id` (para poder
    contar piezas reales deduplicando, en vez de contar filas = piezas × modelos) y
    `estado_consenso` (la decisión real de aceptar/rechazar de la planta, por mayoría de los
    3 modelos, no la de un modelo individual)."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO produccion (timestamp, linea, defecto, confianza, estado, clase_real, estado_real,
                                 modelo_ia, tasa_defectos_objetivo, imagen_original, imagen_gradcam,
                                 nombre_fichero, pieza_id, estado_consenso)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, linea, defecto, confianza, estado, clase_real, estado_real, modelo_ia,
          tasa_defectos_objetivo, imagen_original_bytes, imagen_gradcam_bytes, nombre_fichero,
          pieza_id, estado_consenso))
    conn.commit()
    conn.close()

def cargar_datos():
    """Carga los datos de producción en un DataFrame de Pandas (sin las imágenes: son BLOBs
    pesados que solo se cargan bajo demanda para un registro concreto, ver cargar_imagenes_registro).
    Cada fila es la evaluación de UN modelo sobre una pieza (hay 3 filas por pieza física);
    usar `pieza_id` para deduplicar cuando se necesite contar piezas reales, no filas."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query('''
        SELECT id, timestamp, linea, defecto, confianza, estado, clase_real, estado_real,
               modelo_ia, tasa_defectos_objetivo, nombre_fichero, pieza_id, estado_consenso
        FROM produccion
    ''', conn)
    conn.close()
    return df

def cargar_imagenes_registro(id_registro):
    """Recupera las imágenes (original y Grad-CAM) y el fichero de origen guardados para un
    registro concreto de la tabla, identificado por su id, para poder inspeccionarlos manualmente."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT imagen_original, imagen_gradcam, modelo_ia, defecto, confianza, clase_real, nombre_fichero "
              "FROM produccion WHERE id = ?", (int(id_registro),))
    fila = c.fetchone()
    conn.close()
    return fila

def limpiar_base_datos():
    """Borra todos los registros para reiniciar la simulación."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('DELETE FROM produccion')
    conn.commit()
    conn.close()

# Inicializar DB al arrancar la app
init_db()

# ==========================================
# 2. LÓGICA DE SIMULACIÓN Y ESTADO
# ==========================================
# Inicializar el estado de las 5 líneas de producción
if 'lineas' not in st.session_state:
    st.session_state.lineas = {
        i: {
            # Tiempo aleatorio inicial para la primera pieza (entre 5 y 20 segundos)
            "proximo_procesamiento": time.time() + random.uniform(5, 20),
            "ultimo_resultado": None,
            "piezas_totales": 0
        } for i in range(1, 6)
    }

# Última pieza procesada globalmente (de cualquier línea), la que se muestra en grande
if 'ultima_pieza_global' not in st.session_state:
    st.session_state.ultima_pieza_global = None

# Instante en que se activó la simulación (None = simulación detenida), para poder pararla sola tras 1 hora
if 'tiempo_inicio_simulacion' not in st.session_state:
    st.session_state.tiempo_inicio_simulacion = None

# ==========================================
# 3. INTERFAZ DE NAVEGACIÓN Y SELECCIÓN DE MODELO (SIDEBAR)
# ==========================================
st.sidebar.title("🏭 Gemelo Digital: Inspección de Calidad en Superficies Metálicas")
st.sidebar.caption("🚀 Prototipo MLOps y Visión Artificial (Trabajo de Fin de Máster)")
st.sidebar.divider()

if DATASET_EXISTS:
    st.sidebar.success("🟢 Modo LOCAL (Dataset Completo)")
else:
    st.sidebar.warning("🟡 Modo EVALUACIÓN (Dataset de Muestra)")

st.sidebar.divider()
st.sidebar.markdown("**Selecciona el módulo a visualizar:**")
pagina = st.sidebar.radio("Módulos", ["Simulador de Planta (En Vivo)", "Dashboard de Resultados", "Inspección Manual (Subir Imagen)"])

st.sidebar.divider()
st.sidebar.caption("🧠 Tanto el Simulador como la Inspección Manual evalúan siempre con "
                    "los 3 modelos entrenados a la vez (EfficientNetB0, ResNet50, MobileNetV2).")

st.sidebar.divider()
st.sidebar.markdown("### 🎛️ Simulación")
tasa_defectos_pct = st.sidebar.slider(
    "Tasa de defectos objetivo (%)",
    min_value=5, max_value=70, value=50, step=5,
    help="Probabilidad de que la siguiente pieza entrante tenga realmente un defecto. "
         "5-15% simula un entorno de producción real; valores altos (50-70%) fuerzan un "
         "modo demostración intensivo con más rechazos. Se aplica en tiempo real, a partir "
         "de la siguiente pieza que procese cada línea."
)
tasa_defectos = tasa_defectos_pct / 100

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Administración")
if st.sidebar.button("🗑️ Resetear Base de Datos"):
    limpiar_base_datos()
    for i in range(1, 6):
        st.session_state.lineas[i]["piezas_totales"] = 0
        st.session_state.lineas[i]["ultimo_resultado"] = None
    st.session_state.ultima_pieza_global = None
    st.sidebar.success("Base de datos reiniciada.")

# ==========================================
# 4. MODELO REAL Y GRAD-CAM (idéntico a 05_app_despliegue_streamlit / 04_gradcam_autocalibrado)
# ==========================================
@st.cache_resource
def load_classification_model(model_path):
    """Carga el modelo entrenado en caché (una instancia por cada modelo distinto que se seleccione)."""
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Error al cargar el modelo '{model_path}': {e}")
        st.info(f"Por favor, asegúrate de que el archivo '{model_path}' esté en la misma carpeta que este script.")
        return None

@st.cache_resource
def construir_grad_model(nombre_modelo):
    """Construye UNA SOLA VEZ (cacheado) el sub-modelo de Grad-CAM que devuelve logits puros."""
    modelo = modelos_cargados.get(nombre_modelo)
    if modelo is None:
        return None
    last_conv_layer_name = MODEL_CONFIGS[nombre_modelo]["last_conv_layer"]
    modelo.layers[-1].activation = None
    grad_model = tf.keras.models.Model(
        inputs=[modelo.inputs],
        outputs=[modelo.get_layer(last_conv_layer_name).output, modelo.output]
    )
    return grad_model

@st.cache_data
def listar_imagenes_dataset():
    """Indexa una sola vez las imágenes disponibles como 'piezas entrantes',
    separadas en dos pools (normales vs. con defecto) según su carpeta (etiqueta real).
    Intenta con dataset masivo primero, fallback a dataset de muestra si no existe."""
    imagenes_normales, imagenes_defecto = [], []

    ruta_a_buscar = DATASET_PATH if os.path.exists(DATASET_PATH) else os.path.join(BASE_DIR, "data", "sample_images")

    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for ruta in glob.glob(os.path.join(ruta_a_buscar, "**", ext), recursive=True):
            carpeta = os.path.basename(os.path.dirname(ruta)).lower()
            if carpeta == 'normal':
                imagenes_normales.append(ruta)
            else:
                imagenes_defecto.append(ruta)
    return imagenes_normales, imagenes_defecto

# Tanto el Simulador en Vivo como la Inspección Manual evalúan cada pieza con los 3
# modelos a la vez (para compararlos sobre exactamente la misma imagen), así que se
# cargan los 3 de entrada, junto con su grad_model correspondiente (ver construir_grad_model).
modelos_cargados = {nombre: load_classification_model(cfg["path"]) for nombre, cfg in MODEL_CONFIGS.items()}
grad_models_cargados = {nombre: construir_grad_model(nombre) for nombre in MODEL_CONFIGS}
imagenes_normales, imagenes_defecto = listar_imagenes_dataset()
imagenes_dataset = imagenes_normales + imagenes_defecto

def make_gradcam_heatmap(img_array, grad_model):
    """Calcula el mapa de calor matemático de los gradientes (Grad-CAM real) usando un
    grad_model ya construido de antemano (ver construir_grad_model) — no muta ningún
    modelo compartido en cada llamada."""
    img_array = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap_max = tf.math.reduce_max(heatmap)
    heatmap = tf.cond(heatmap_max > 0, lambda: tf.maximum(heatmap, 0) / heatmap_max, lambda: tf.maximum(heatmap, 0))

    return heatmap.numpy(), int(pred_index.numpy()), preds[0]

def create_superimposed_image(img_pil, heatmap, alpha=0.5):
    """Superpone el mapa térmico sobre la imagen original utilizando un colormap Jet."""
    img_array = img_to_array(img_pil)

    heatmap = np.uint8(255 * heatmap)
    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img_array
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

def preparar_imagen_para_modelo(img_pil):
    """Redimensiona una imagen PIL a TARGET_SIZE de forma IDÉNTICA para el Simulador y la
    Inspección Manual. Antes divergían: el Simulador cargaba con `load_img(..., target_size=...)`
    (interpolación 'nearest' por defecto en Keras) mientras que la Inspección Manual usaba
    `Image.resize()` directamente (que en Pillow reciente usa 'bicubic' por defecto). Esa
    diferencia de método de redimensionado puede cambiar sutilmente los píxeles en casos límite
    y hacer que el mismo fichero dé una predicción distinta según por qué camino se procese —
    justo el síntoma reportado (falso negativo en el Simulador, resultado distinto al subirla
    a mano). Se usa 'bilinear' porque es el interpolador por defecto de
    `image_dataset_from_directory`, la forma habitual de cargar datasets en carpetas como este."""
    img_resized = img_pil.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    img_array = img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_resized, img_batch

def inspeccionar_pieza_multi_modelo(modelos_cargados, tasa_defectos):
    """Toma UNA imagen real del dataset de validación (sorteada del pool de defectuosas
    con probabilidad `tasa_defectos` y del pool de normales en caso contrario, ver el
    slider de la barra lateral) y la evalúa con LOS 3 MODELOS ENTRENADOS a la vez, cada
    uno con su propio Grad-CAM real sobre esa misma imagen. Así se puede comparar cómo
    decide cada arquitectura ante exactamente la misma pieza. La decisión de planta
    (aceptar/rechazar) se toma por consenso: se rechaza si 2 de los 3 modelos rechazan.
    Como la imagen procede de una carpeta con etiqueta conocida, también se devuelve la
    clase real (ground truth) para poder auditar falsos positivos/negativos por modelo, y
    el nombre/ruta del fichero de origen para poder trazar cada registro hasta su imagen
    original en el dataset."""
    if imagenes_defecto and (not imagenes_normales or random.random() < tasa_defectos):
        img_path = random.choice(imagenes_defecto)
    else:
        img_path = random.choice(imagenes_normales)
    carpeta_real = os.path.basename(os.path.dirname(img_path)).lower()
    clase_real = FOLDER_TO_CLASS_NAME.get(carpeta_real, carpeta_real)
    estado_real = "Aceptada" if clase_real == 'Normal (Sin defectos)' else "Rechazada (Defecto)"
    nombre_fichero = os.path.relpath(img_path, DATASET_PATH)  # ej. "crack/crack_0042.jpg", para trazabilidad

    img_sin_redimensionar = Image.open(img_path).convert('RGB')
    img_original, img_batch = preparar_imagen_para_modelo(img_sin_redimensionar)

    resultados_por_modelo = {}
    for nombre_modelo in MODEL_CONFIGS:
        modelo_obj = modelos_cargados.get(nombre_modelo)
        grad_model = grad_models_cargados.get(nombre_modelo)
        if modelo_obj is None or grad_model is None:
            continue  # este modelo no se pudo cargar (ver error mostrado al arrancar la app)

        heatmap, pred_index, preds = make_gradcam_heatmap(img_batch, grad_model)
        confianza = float(tf.nn.softmax(preds)[pred_index]) * 100
        defecto_predicho = CLASS_NAMES.get(pred_index, f"Clase_{pred_index}")
        estado_pieza = "Aceptada" if defecto_predicho == 'Normal (Sin defectos)' else "Rechazada (Defecto)"
        img_gradcam = create_superimposed_image(img_original, heatmap)

        resultados_por_modelo[nombre_modelo] = {
            "defecto": defecto_predicho,
            "confianza": confianza,
            "estado": estado_pieza,
            "imagen_gradcam": img_gradcam,
        }

    rechazos = sum(1 for r in resultados_por_modelo.values() if r["estado"] == "Rechazada (Defecto)")
    estado_consenso = "Rechazada (Defecto)" if rechazos >= 2 else "Aceptada"

    return img_original, clase_real, estado_real, resultados_por_modelo, estado_consenso, nombre_fichero

# ==========================================
# 5. PÁGINA 1: SIMULADOR DE PLANTA EN VIVO
# ==========================================
if pagina == "Simulador de Planta (En Vivo)":
    # Cabecera compacta: título + toggle + cronómetro en una sola fila para ahorrar espacio vertical.
    col_titulo, col_toggle, col_timer = st.columns([2.2, 1.3, 1.3])
    with col_titulo:
        st.markdown("### 🏭 Monitor en Vivo · Comparativa de los 3 modelos")

    # Si la simulación llevaba encendida más de DURACION_SIMULACION_SEG, se apaga sola
    # ANTES de dibujar el toggle (no se puede tocar st.session_state de un widget ya instanciado).
    simulacion_detenida_por_tiempo = False
    if (st.session_state.tiempo_inicio_simulacion is not None and
            time.time() - st.session_state.tiempo_inicio_simulacion >= DURACION_SIMULACION_SEG):
        st.session_state["simulacion_activa_widget"] = False
        st.session_state.tiempo_inicio_simulacion = None
        simulacion_detenida_por_tiempo = True

    with col_toggle:
        simulacion_activa = st.toggle("▶️ Activar Simulación", key="simulacion_activa_widget")

    if simulacion_activa and st.session_state.tiempo_inicio_simulacion is None:
        st.session_state.tiempo_inicio_simulacion = time.time()  # se acaba de activar: arranca el cronómetro
    elif not simulacion_activa:
        st.session_state.tiempo_inicio_simulacion = None

    with col_timer:
        if simulacion_detenida_por_tiempo:
            st.caption("⏱️ Detenida automáticamente tras 1 hora.")
        elif simulacion_activa:
            restante = max(0, DURACION_SIMULACION_SEG - (time.time() - st.session_state.tiempo_inicio_simulacion))
            horas_rest, resto_seg = divmod(int(restante), 3600)
            min_rest, seg_rest = divmod(resto_seg, 60)
            st.caption(f"⏱️ Parada automática en: {horas_rest}:{min_rest:02d}:{seg_rest:02d}")

    modelos_disponibles = {n: m for n, m in modelos_cargados.items() if m is not None}
    if not modelos_disponibles:
        st.error("No se puede simular: ninguno de los 3 modelos se cargó correctamente.")
    elif len(modelos_disponibles) < len(MODEL_CONFIGS):
        faltantes = ", ".join(n for n in MODEL_CONFIGS if n not in modelos_disponibles)
        st.warning(f"Modelo(s) no disponibles (se excluyen de la comparación): {faltantes}")
    if not imagenes_dataset:
        st.error(f"No se encontraron imágenes en '{DATASET_PATH}'. Verifica la ruta del dataset.")

    # Contenedores para las 5 líneas
    columnas_lineas = st.columns(5)

    # Evaluar si toca procesar alguna pieza
    if simulacion_activa and modelos_disponibles and imagenes_dataset:
        tiempo_actual = time.time()

        for i in range(1, 6):
            if tiempo_actual >= st.session_state.lineas[i]["proximo_procesamiento"]:
                # --- INFERENCIA REAL CON LOS 3 MODELOS SOBRE LA MISMA PIEZA (imagen real del dataset) ---
                (img_original_pieza, clase_real, estado_real, resultados_modelos,
                 estado_consenso, nombre_fichero) = inspeccionar_pieza_multi_modelo(modelos_disponibles, tasa_defectos)

                # Guardar en Base de Datos: una fila por modelo (misma pieza, misma línea, mismo timestamp),
                # con la imagen original, el Grad-CAM de ESE modelo y el fichero de origen, para poder
                # auditar y comparar cada modelo individualmente en el Dashboard (incluida una revisión
                # visual manual y la trazabilidad hasta la imagen del dataset). Las 3 filas comparten el
                # mismo `pieza_id` (para poder contar piezas físicas, no filas) y el mismo
                # `estado_consenso` (la decisión real de aceptar/rechazar de la planta).
                pieza_id = str(uuid.uuid4())
                imagen_original_bytes = imagen_a_bytes(img_original_pieza)
                for nombre_modelo, res in resultados_modelos.items():
                    imagen_gradcam_bytes = imagen_a_bytes(res["imagen_gradcam"])
                    insertar_registro(i, res["defecto"], res["confianza"], res["estado"],
                                       clase_real, estado_real, nombre_modelo, tasa_defectos_pct,
                                       imagen_original_bytes, imagen_gradcam_bytes, nombre_fichero,
                                       pieza_id, estado_consenso)

                # Actualizar estado de la sesión (la decisión de planta es el consenso de los 3 modelos)
                resultado = {
                    "linea": i,
                    "estado": estado_consenso,
                    "hora": datetime.now().strftime("%H:%M:%S"),
                    "clase_real": clase_real,
                    "estado_real": estado_real,
                }
                st.session_state.lineas[i]["ultimo_resultado"] = resultado
                st.session_state.lineas[i]["piezas_totales"] += 1
                # Esta es la última pieza producida en toda la planta: la que se muestra en grande,
                # con el detalle de los 3 modelos para poder compararlos.
                st.session_state.ultima_pieza_global = {
                    **resultado,
                    "imagen_original": img_original_pieza,
                    "resultados_modelos": resultados_modelos,
                }
                # Configurar el tiempo para la SIGUIENTE pieza (aleatorio entre 8 y 20 segs)
                st.session_state.lineas[i]["proximo_procesamiento"] = tiempo_actual + random.uniform(8, 20)

    # Renderizar UI de las líneas (solo estado y métricas, sin imagen individual)
    for i, col in enumerate(columnas_lineas, 1):
        with col:
            estado_linea = st.session_state.lineas[i]
            estado_cinta = "🔄 En movimiento" if simulacion_activa else "⏸️ Detenida"

            # Mostrar último resultado (texto) de esta línea, todo en una única tarjeta compacta
            if estado_linea["ultimo_resultado"]:
                res = estado_linea["ultimo_resultado"]
                color = "green" if res["estado"] == "Aceptada" else "red"
                st.markdown(f"""
                <div style="border:1px solid {color}; padding:12px 16px; border-radius:8px; font-size:1.05rem; line-height:1.4;">
                    <strong>Línea {i}</strong> · {estado_cinta} · Total: {estado_linea['piezas_totales']}<br>
                    <span style="color:{color}; font-weight:bold;">{res['estado']}</span> ({res['hora']})<br>
                    Consenso de los 3 modelos
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="border:1px dashed gray; padding:12px 16px; border-radius:8px; font-size:1.05rem; text-align:center; line-height:1.4;">
                    <strong>Línea {i}</strong> · {estado_cinta} · Total: {estado_linea['piezas_totales']}<br>
                    Esperando primera pieza...
                </div>
                """, unsafe_allow_html=True)

    # ==========================================
    # Pieza activa: pieza original + comparación de los 3 modelos sobre esa misma
    # pieza, en 4 tarjetas idénticas en tamaño y estilo (una fila, mismo ancho,
    # mismo alto de imagen) para que quede alineado y visualmente coherente.
    # ==========================================
    pieza = st.session_state.ultima_pieza_global
    if pieza:
        color = "green" if pieza["estado"] == "Aceptada" else "red"
        st.markdown(f"""
        <div style="border:1px solid {color}; padding:8px 18px; border-radius:8px; font-size:1.15rem; margin-top:0.6rem; white-space:nowrap; overflow-x:auto;">
            🔎 <strong>Línea {pieza['linea']} · {pieza['hora']}:</strong>
            Decisión de planta (consenso de los 3 modelos):
            <span style="color:{color}; font-weight:bold;"> {pieza['estado']}</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Misma pieza evaluada por los 3 modelos · Grad-CAM = zona en la que se ha fijado cada modelo para decidir")

        GRIS_NEUTRO = "#616161"
        columnas_pieza = st.columns(1 + len(MODEL_CONFIGS))

        # Las 4 tarjetas usan siempre EXACTAMENTE 3 líneas (nombre / decisión / estado de
        # acierto), así su altura es idéntica sin importar la longitud del texto de cada una
        # (ej. "MobileNetV2" + "Rechazada (Defecto)" es más largo que "ResNet50" + "Aceptada"),
        # y las imágenes de debajo arrancan siempre alineadas en la misma fila.
        with columnas_pieza[0]:
            st.markdown(f"""
            <div style="border:1px solid {GRIS_NEUTRO}; padding:6px 10px; border-radius:6px; font-size:1rem; text-align:center; line-height:1.35; white-space:nowrap; overflow-x:auto;">
                <strong>📷 Pieza Original</strong><br>
                <span style="color:{GRIS_NEUTRO}; font-weight:bold;">ENTRADA</span><br>
                Clase real: {pieza['clase_real']}
            </div>
            """, unsafe_allow_html=True)
            st.image(pieza["imagen_original"], caption="Superficie", width='stretch')

        for col, nombre_modelo in zip(columnas_pieza[1:], MODEL_CONFIGS.keys()):
            with col:
                res = pieza["resultados_modelos"].get(nombre_modelo)
                if res is None:
                    st.warning(f"{nombre_modelo}: no disponible")
                    continue
                acierto = res["defecto"] == pieza["clase_real"]
                color_m = "green" if res["estado"] == "Aceptada" else "red"
                # La decisión (verde/rojo) y el acierto son cosas distintas: un modelo puede
                # "Rechazar" (rojo) y aun así equivocarse de defecto, o "Aceptar" (verde) una
                # pieza que en realidad es defectuosa. El borde ámbar + la 3ª línea resaltan
                # el ERROR en sí, independientemente del color de la decisión.
                if acierto:
                    borde_tarjeta = f"2px solid {color_m}"
                    fondo_tarjeta = "transparent"
                    linea_acierto = '<span style="color:green; font-weight:bold;">✅ Correcto</span>'
                else:
                    borde_tarjeta = "3px solid #ff9800"
                    fondo_tarjeta = "rgba(255,152,0,0.12)"
                    linea_acierto = '<span style="color:#e65100; font-weight:bold;">⚠️ ERROR</span>'
                st.markdown(f"""
                <div style="border:{borde_tarjeta}; background:{fondo_tarjeta}; padding:6px 10px; border-radius:6px; font-size:1rem; text-align:center; line-height:1.35; white-space:nowrap; overflow-x:auto;">
                    <strong>🧠 {nombre_modelo}</strong><br>
                    <span style="color:{color_m}; font-weight:bold;">{res['estado']}</span> · {res['defecto']} ({res['confianza']:.1f}%)<br>
                    {linea_acierto}
                </div>
                """, unsafe_allow_html=True)
                st.image(res["imagen_gradcam"], caption="Grad-CAM", width='stretch')
    else:
        st.markdown("<div style='border:1px dashed gray; padding:28px; border-radius:8px; text-align:center; font-size:1.5rem; margin-top:0.6rem;'>Esperando la primera pieza de cualquier línea...</div>", unsafe_allow_html=True)

    # Bucle de refresco automático si la simulación está activa
    if simulacion_activa:
        time.sleep(1.5) # Espera corta para no saturar la CPU
        st.rerun()      # Vuelve a ejecutar el script para actualizar la UI

# ==========================================
# 6. PÁGINA 2: DASHBOARD DE RESULTADOS (BI)
# ==========================================
elif pagina == "Dashboard de Resultados":
    st.title("📊 Dashboard de Calidad MLOps")
    
    df = cargar_datos()

    if df.empty:
        st.info("La base de datos está vacía. Ve al 'Simulador de Planta' y activa la producción para generar datos.")
    else:
        # Copia sin filtrar por modelo, para poder mostrar más abajo una comparativa directa
        # entre los 3 modelos sin depender de qué opción tenga seleccionada el desplegable.
        df_sin_filtrar = df.copy()

        # Filtro por modelo de IA: cada modelo puede tener una fiabilidad distinta,
        # así que las métricas no deben mezclar piezas inspeccionadas por modelos diferentes.
        modelos_en_bd = sorted(df['modelo_ia'].dropna().unique().tolist())
        filtro_modelo = st.selectbox("🧠 Filtrar por modelo de IA", ["Todos los modelos"] + modelos_en_bd)
        if filtro_modelo != "Todos los modelos":
            df = df[df['modelo_ia'] == filtro_modelo]

        # 6.1. KPIs (Key Performance Indicators) Principales
        st.subheader("Indicadores Globales")
        col1, col2, col3, col4 = st.columns(4)

        # Cada pieza genera 3 filas (una por modelo evaluado): hay que deduplicar por
        # pieza_id para contar piezas físicas de verdad, no filas. La decisión de aceptar/
        # rechazar a nivel de planta es el consenso guardado (estado_consenso), no el
        # 'estado' de un modelo concreto (que puede diferir de los otros dos).
        df_piezas = df.dropna(subset=['pieza_id']).drop_duplicates(subset='pieza_id')

        total_piezas = len(df_piezas)
        piezas_rechazadas = len(df_piezas[df_piezas['estado_consenso'].str.contains("Rechazada", na=False)])
        porcentaje_rechazo = (piezas_rechazadas / total_piezas) * 100 if total_piezas > 0 else 0

        # Encontrar la línea con más rechazos (a nivel de planta, por consenso)
        rechazos_por_linea = df_piezas[df_piezas['estado_consenso'].str.contains("Rechazada", na=False)]['linea'].value_counts()
        linea_problematica = rechazos_por_linea.idxmax() if not rechazos_por_linea.empty else "N/A"

        col1.metric("Total Piezas Analizadas", total_piezas)
        col2.metric("Piezas Rechazadas", piezas_rechazadas)
        col3.metric("Tasa de Defectos", f"{porcentaje_rechazo:.1f}%")
        col4.metric("Línea con más fallos", f"Línea {linea_problematica}")
        st.caption(f"({len(df)} filas de evaluación individual por modelo en la BD para estas {total_piezas} piezas)")

        # 6.1.bis Fiabilidad del modelo: Falsos Positivos y Falsos Negativos
        # (comparando la predicción de la IA contra la etiqueta real del dataset)
        df_auditado = df.dropna(subset=['estado_real'])
        st.markdown("#### Fiabilidad del Modelo (predicción IA vs. etiqueta real)")

        # Comparativa directa entre los 3 modelos, SIEMPRE sobre el total de piezas (independiente
        # de la opción elegida en "Filtrar por modelo de IA" de arriba) — para poder ver de un
        # vistazo qué modelo falla más sin tener que ir alternando el desplegable uno a uno.
        df_auditado_todos = df_sin_filtrar.dropna(subset=['estado_real'])
        if not df_auditado_todos.empty:
            filas_comparativa = []
            for nombre_m in sorted(df_auditado_todos['modelo_ia'].dropna().unique()):
                d = df_auditado_todos[df_auditado_todos['modelo_ia'] == nombre_m]
                fp_m = len(d[(d['estado_real'] == 'Aceptada') & (d['estado'] == 'Rechazada (Defecto)')])
                fn_m = len(d[(d['estado_real'] == 'Rechazada (Defecto)') & (d['estado'] == 'Aceptada')])
                # Error de Clasificación: acertó que había defecto (mismo estado binario que la
                # realidad), pero confundió DE QUÉ defecto se trataba (ej. real Crack, predijo
                # Scratch). No es FP ni FN porque el binario Aceptada/Rechazada coincide.
                ec_m = len(d[(d['estado_real'] == 'Rechazada (Defecto)') & (d['estado'] == 'Rechazada (Defecto)') & (d['defecto'] != d['clase_real'])])
                filas_comparativa.append({
                    "Modelo": nombre_m,
                    "Piezas evaluadas": len(d),
                    "Falsos Positivos": fp_m,
                    "Falsos Negativos": fn_m,
                    "Errores de Clasificación": ec_m,
                    "Total errores": fp_m + fn_m + ec_m,
                    "Tasa de error (%)": round((fp_m + fn_m + ec_m) / len(d) * 100, 1) if len(d) > 0 else 0,
                })
            df_comparativa = pd.DataFrame(filas_comparativa).sort_values("Total errores", ascending=False)
            st.markdown("###### 📊 Comparativa de fiabilidad por modelo (todas las piezas, sin filtrar)")
            st.dataframe(df_comparativa, width='stretch', hide_index=True)

        if df_auditado.empty:
            st.info("Aún no hay piezas con etiqueta real registrada. Procesa piezas nuevas en el simulador para calcular estos indicadores.")
        else:
            piezas_sanas_reales = len(df_auditado[df_auditado['estado_real'] == 'Aceptada'])
            piezas_defecto_reales = len(df_auditado[df_auditado['estado_real'] == 'Rechazada (Defecto)'])

            falsos_positivos = len(df_auditado[(df_auditado['estado_real'] == 'Aceptada') & (df_auditado['estado'] == 'Rechazada (Defecto)')])
            falsos_negativos = len(df_auditado[(df_auditado['estado_real'] == 'Rechazada (Defecto)') & (df_auditado['estado'] == 'Aceptada')])
            # Error de Clasificación: acertó el binario Aceptada/Rechazada (detectó que había
            # defecto), pero confundió DE QUÉ defecto se trataba (ej. real Crack, predijo
            # Scratch). Antes esto no se contaba en ningún sitio porque FP/FN solo miran el
            # binario, no la clase exacta.
            errores_clasificacion = len(df_auditado[(df_auditado['estado_real'] == 'Rechazada (Defecto)') & (df_auditado['estado'] == 'Rechazada (Defecto)') & (df_auditado['defecto'] != df_auditado['clase_real'])])

            tasa_fp = (falsos_positivos / piezas_sanas_reales * 100) if piezas_sanas_reales > 0 else 0
            tasa_fn = (falsos_negativos / piezas_defecto_reales * 100) if piezas_defecto_reales > 0 else 0
            tasa_ec = (errores_clasificacion / piezas_defecto_reales * 100) if piezas_defecto_reales > 0 else 0

            col_fp, col_fn, col_ec = st.columns(3)
            col_fp.metric("Falsos Positivos", falsos_positivos, delta=f"{tasa_fp:.1f}% de piezas sanas", delta_color="off",
                          help="Piezas realmente SIN defecto que la IA rechazó por error (falsa alarma).")
            col_fn.metric("Falsos Negativos", falsos_negativos, delta=f"{tasa_fn:.1f}% de piezas defectuosas", delta_color="off",
                          help="Piezas realmente CON defecto que la IA aceptó por error (el caso más crítico: un defecto se cuela en producción).")
            col_ec.metric("Errores de Clasificación", errores_clasificacion, delta=f"{tasa_ec:.1f}% de piezas defectuosas", delta_color="off",
                          help="Piezas realmente defectuosas que la IA rechazó correctamente, pero confundiendo de qué defecto se trataba (ej. predijo Arañazo siendo en realidad Grieta).")

            # --- Desglose detallado: exactamente qué tipos de pieza confunde el modelo ---
            df_fp = df_auditado[(df_auditado['estado_real'] == 'Aceptada') & (df_auditado['estado'] == 'Rechazada (Defecto)')]
            df_fn = df_auditado[(df_auditado['estado_real'] == 'Rechazada (Defecto)') & (df_auditado['estado'] == 'Aceptada')]
            df_ec = df_auditado[(df_auditado['estado_real'] == 'Rechazada (Defecto)') & (df_auditado['estado'] == 'Rechazada (Defecto)') & (df_auditado['defecto'] != df_auditado['clase_real'])]

            if falsos_positivos > 0 or falsos_negativos > 0 or errores_clasificacion > 0:
                st.markdown("##### 🔬 Desglose detallado de errores por tipo de pieza")

                col_graf_fn, col_graf_fp, col_graf_ec = st.columns(3)
                with col_graf_fn:
                    st.caption("Falsos Negativos: qué defecto real se le coló a la IA")
                    if not df_fn.empty:
                        conteo_fn = df_fn['clase_real'].value_counts().rename_axis('Tipo de defecto real').reset_index(name='Veces no detectado')
                        fig_fn = px.bar(conteo_fn, x='Tipo de defecto real', y='Veces no detectado',
                                        color_discrete_sequence=['#d62728'])
                        st.plotly_chart(fig_fn, width='stretch')
                    else:
                        st.write("Sin falsos negativos con los filtros actuales.")
                with col_graf_fp:
                    st.caption("Falsos Positivos: qué defecto 'vio' la IA sin existir")
                    if not df_fp.empty:
                        conteo_fp = df_fp['defecto'].value_counts().rename_axis('Defecto predicho por error').reset_index(name='Veces')
                        fig_fp = px.bar(conteo_fp, x='Defecto predicho por error', y='Veces',
                                        color_discrete_sequence=['#ff7f0e'])
                        st.plotly_chart(fig_fp, width='stretch')
                    else:
                        st.write("Sin falsos positivos con los filtros actuales.")
                with col_graf_ec:
                    st.caption("Errores de Clasificación: qué defecto real confunde más de tipo")
                    if not df_ec.empty:
                        conteo_ec = df_ec['clase_real'].value_counts().rename_axis('Tipo de defecto real').reset_index(name='Veces confundido')
                        fig_ec = px.bar(conteo_ec, x='Tipo de defecto real', y='Veces confundido',
                                        color_discrete_sequence=['#9467bd'])
                        st.plotly_chart(fig_ec, width='stretch')
                    else:
                        st.write("Sin errores de clasificación con los filtros actuales.")

                tipo_error_filtro = st.radio(
                    "Ver registros individuales de:",
                    ["Todos", "Solo Falsos Positivos", "Solo Falsos Negativos", "Solo Errores de Clasificación"],
                    horizontal=True
                )
                if tipo_error_filtro == "Solo Falsos Positivos":
                    df_errores = df_fp.copy()
                elif tipo_error_filtro == "Solo Falsos Negativos":
                    df_errores = df_fn.copy()
                elif tipo_error_filtro == "Solo Errores de Clasificación":
                    df_errores = df_ec.copy()
                else:
                    df_errores = pd.concat([df_fp, df_fn, df_ec]).copy()

                if df_errores.empty:
                    st.success("No hay errores de este tipo con los filtros actuales.")
                else:
                    condiciones_tipo_error = [
                        df_errores['estado_real'] == 'Aceptada',
                        (df_errores['estado_real'] == 'Rechazada (Defecto)') & (df_errores['estado'] == 'Aceptada'),
                    ]
                    df_errores['tipo_error'] = np.select(
                        condiciones_tipo_error, ['Falso Positivo', 'Falso Negativo'],
                        default='Error de Clasificación'
                    )
                    columnas_mostrar = ['id', 'timestamp', 'linea', 'modelo_ia', 'tipo_error', 'clase_real', 'defecto', 'confianza', 'nombre_fichero']
                    df_errores_mostrar = df_errores[columnas_mostrar].sort_values(by='timestamp', ascending=False)
                    st.dataframe(df_errores_mostrar, width='stretch')

                    # Auditoría visual directa de un error: elige uno de los FP/FN de arriba y
                    # mira la imagen original + el Grad-CAM que se guardaron en ese momento.
                    st.markdown("###### 🖼️ Ver la imagen guardada de uno de estos errores")
                    etiquetas_error = {
                        int(fila['id']): (f"ID {int(fila['id'])} · {fila['timestamp']} · Línea {int(fila['linea'])} · "
                                           f"{fila['modelo_ia']} · {fila['tipo_error']} · "
                                           f"real: {fila['clase_real']} → predijo: {fila['defecto']}")
                        for _, fila in df_errores_mostrar.iterrows()
                    }
                    id_error_elegido = st.selectbox(
                        "Selecciona un registro de error",
                        options=list(etiquetas_error.keys()),
                        format_func=lambda x: etiquetas_error[x],
                    )
                    fila_imagenes_error = cargar_imagenes_registro(id_error_elegido)
                    if fila_imagenes_error is None or fila_imagenes_error[0] is None:
                        st.info("Este registro no tiene imagen guardada (es anterior a activar esta funcionalidad).")
                    else:
                        (img_original_error, img_gradcam_error, modelo_ia_error, defecto_error,
                         confianza_error, clase_real_error, nombre_fichero_error) = fila_imagenes_error
                        if nombre_fichero_error:
                            st.caption(f"📄 Fichero de origen: `{nombre_fichero_error}`")
                        col_ve1, col_ve2 = st.columns(2)
                        with col_ve1:
                            st.image(io.BytesIO(img_original_error), caption="Superficie original guardada", width='stretch')
                        with col_ve2:
                            if img_gradcam_error:
                                st.image(io.BytesIO(img_gradcam_error), caption=f"Grad-CAM guardado ({modelo_ia_error})", width='stretch')

        st.divider()

        # 6.1.ter Tasa de Rechazo Real vs. Tasa de Defectos Objetivo (slider de la barra lateral)
        st.markdown("#### 🎯 Tasa de Rechazo Real vs. Tasa de Defectos Objetivo (Slider)")
        st.caption("Para cada valor configurado en el slider 'Tasa de defectos objetivo', compara qué "
                   "porcentaje de piezas eran realmente defectuosas (ground truth del dataset) frente al "
                   "porcentaje que el modelo de IA acabó rechazando (incluye sus falsos positivos/negativos).")

        df_tasa = df.dropna(subset=['tasa_defectos_objetivo'])
        if df_tasa.empty:
            st.info("Aún no hay piezas registradas con una tasa de defectos objetivo asociada (funcionalidad "
                     "añadida recientemente). Genera piezas nuevas en el simulador para ver esta comparación.")
        else:
            # OJO: esto agrupa filas de evaluación (una por modelo), no piezas físicas — con
            # "Todos los modelos" seleccionado hay 3 filas por pieza. Se llama "Evaluaciones"
            # a propósito (no "Piezas") para no dar la impresión de que hay 3 veces más piezas
            # de las que realmente pasaron por la planta.
            resumen_tasa = df_tasa.groupby('tasa_defectos_objetivo').agg(
                Evaluaciones=('id', 'count'),
                **{'Piezas realmente defectuosas (ground truth)': ('estado_real', lambda s: (s == 'Rechazada (Defecto)').mean() * 100)},
                **{'Piezas rechazadas por la IA': ('estado', lambda s: (s == 'Rechazada (Defecto)').mean() * 100)}
            ).reset_index().rename(columns={'tasa_defectos_objetivo': 'Tasa objetivo (%)'})
            resumen_tasa = resumen_tasa.sort_values('Tasa objetivo (%)')

            df_tasa_melt = resumen_tasa.melt(
                id_vars=['Tasa objetivo (%)', 'Evaluaciones'],
                value_vars=['Piezas realmente defectuosas (ground truth)', 'Piezas rechazadas por la IA'],
                var_name='Serie', value_name='Porcentaje (%)'
            )

            fig_tasa = px.bar(
                df_tasa_melt, x='Tasa objetivo (%)', y='Porcentaje (%)', color='Serie', barmode='group',
                color_discrete_map={
                    'Piezas realmente defectuosas (ground truth)': '#1f77b4',
                    'Piezas rechazadas por la IA': '#d62728'
                }
            )
            fig_tasa.add_scatter(
                x=resumen_tasa['Tasa objetivo (%)'], y=resumen_tasa['Tasa objetivo (%)'],
                mode='lines+markers', name='Objetivo (y = x)', line=dict(color='gray', dash='dash')
            )
            fig_tasa.update_layout(xaxis_title='Tasa de defectos objetivo configurada (%)',
                                    yaxis_title='Porcentaje observado (%)')
            st.plotly_chart(fig_tasa, width='stretch')
            st.caption("Número de evaluaciones registradas por cada tasa objetivo (una por modelo, no por pieza física): " +
                       ", ".join(f"{int(r['Tasa objetivo (%)'])}% → {int(r['Evaluaciones'])} evaluaciones" for _, r in resumen_tasa.iterrows()))

        st.divider()

        # 6.2. Gráficos Analíticos (Plotly)
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            # Gráfico de Torta: Distribución de tipos de defectos REALES (ground truth) por
            # pieza física, deduplicando por pieza_id — usa clase_real, no la predicción de un
            # modelo, para que no se triplique el conteo (antes contaba una vez por cada una
            # de las 3 filas de evaluación de la misma pieza).
            st.markdown("#### Distribución de Anomalías (piezas reales)")
            df_defectos = df_piezas[df_piezas['clase_real'] != 'Normal (Sin defectos)']
            if not df_defectos.empty:
                fig_pie = px.pie(df_defectos, names='clase_real', hole=0.4,
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, width='stretch')
            else:
                st.write("Aún no se han detectado defectos.")
                
        with col_graf2:
            # Gráfico de Barras: Aceptadas vs Rechazadas por Línea, por PIEZA física (consenso
            # de planta), no por fila de evaluación de un modelo — si no, saldrían triplicadas.
            st.markdown("#### Rendimiento por Línea de Producción")
            fig_bar = px.histogram(df_piezas, x="linea", color="estado_consenso", barmode="group",
                                   category_orders={"estado_consenso": ["Aceptada", "Rechazada (Defecto)"]},
                                   color_discrete_map={"Aceptada": "green", "Rechazada (Defecto)": "red"})
            fig_bar.update_layout(xaxis_title="Número de Línea", yaxis_title="Cantidad de Piezas")
            st.plotly_chart(fig_bar, width='stretch')

        # 6.3. Tabla de Datos Crudos
        st.subheader("Registro Histórico de Auditoría")
        st.caption("Cada fila es la evaluación de UN modelo sobre una pieza (3 filas por pieza física, "
                   "agrupadas por 'pieza_id'); el campo 'estado' es la decisión de ESE modelo, mientras "
                   "que 'estado_consenso' es la decisión real de la planta (mayoría de los 3).")
        # Mostrar los últimos 100 registros ordenados por el más reciente
        st.dataframe(df.sort_values(by="id", ascending=False).head(100), width='stretch')

        # 6.4. Auditoría visual manual: recupera la imagen original y el Grad-CAM que se
        # guardaron para un registro concreto, para poder comprobar a simple vista si una
        # predicción sospechosa (ej. muchos fallos de un modelo) es un fallo real del modelo.
        st.markdown("#### 🖼️ Auditoría visual de un registro concreto")
        st.caption("Introduce el ID de un registro de la tabla de arriba (columna 'id') para ver la imagen "
                   "original y el Grad-CAM guardados en el momento de esa inspección. Solo disponible para "
                   "piezas generadas por el Simulador después de activar este guardado de imágenes.")
        id_maximo = int(df['id'].max()) if not df.empty else 0
        id_consulta = st.number_input("ID del registro", min_value=0, max_value=id_maximo, value=id_maximo, step=1)
        if id_consulta > 0:
            fila_imagenes = cargar_imagenes_registro(id_consulta)
            if fila_imagenes is None:
                st.warning(f"No existe ningún registro con ID {id_consulta}.")
            else:
                (img_original_bytes, img_gradcam_bytes, modelo_ia_fila, defecto_fila,
                 confianza_fila, clase_real_fila, nombre_fichero_fila) = fila_imagenes
                if img_original_bytes is None:
                    st.info("Este registro no tiene imagen guardada (es anterior a activar esta funcionalidad).")
                else:
                    st.markdown(f"**Modelo:** {modelo_ia_fila} · **Predicción:** {defecto_fila} "
                                f"({confianza_fila:.1f}%) · **Clase real:** {clase_real_fila}")
                    if nombre_fichero_fila:
                        st.caption(f"📄 Fichero de origen: `{nombre_fichero_fila}`")
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.image(io.BytesIO(img_original_bytes), caption="Superficie original guardada", width='stretch')
                    with col_v2:
                        if img_gradcam_bytes:
                            st.image(io.BytesIO(img_gradcam_bytes), caption="Grad-CAM guardado", width='stretch')

# ==========================================
# 7. PÁGINA 3: INSPECCIÓN MANUAL (idéntico a 05_app_despliegue_streamlit, integrado en la app)
# ==========================================
elif pagina == "Inspección Manual (Subir Imagen)":
    st.title("🔍 Inspección Manual de Piezas")
    st.markdown("Sube una imagen de una superficie metálica y se evaluará con **los 3 modelos entrenados a la vez** "
                "(igual que en el Simulador en Vivo), para comparar cómo decide cada arquitectura ante la misma pieza.")

    modelos_disponibles_manual = {n: m for n, m in modelos_cargados.items() if m is not None}
    if not modelos_disponibles_manual:
        st.error("No se puede inspeccionar: ninguno de los 3 modelos se cargó correctamente.")
    else:
        uploaded_file = st.file_uploader("Cargar imagen de inspección (JPG/PNG)...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            imagen_subida = Image.open(uploaded_file).convert('RGB')
            st.sidebar.image(imagen_subida, caption="Imagen de entrada original", width='stretch')

            img_resized, img_batch = preparar_imagen_para_modelo(imagen_subida)

            with st.spinner('Analizando topología superficial con los 3 modelos...'):
                try:
                    # --- INFERENCIA REAL CON LOS 3 MODELOS SOBRE LA MISMA IMAGEN SUBIDA ---
                    resultados_manual = {}
                    for nombre_modelo in MODEL_CONFIGS:
                        modelo_obj = modelos_disponibles_manual.get(nombre_modelo)
                        grad_model = grad_models_cargados.get(nombre_modelo)
                        if modelo_obj is None or grad_model is None:
                            continue
                        heatmap, pred_index, preds = make_gradcam_heatmap(img_batch, grad_model)
                        confianza = float(tf.nn.softmax(preds)[pred_index]) * 100
                        defecto_predicho = CLASS_NAMES.get(pred_index, f"Clase_{pred_index}")
                        img_xai = create_superimposed_image(img_resized, heatmap)
                        resultados_manual[nombre_modelo] = {
                            "defecto": defecto_predicho,
                            "confianza": confianza,
                            "imagen_gradcam": img_xai,
                        }

                    # Como aquí no hay etiqueta real (es una imagen subida por el usuario, sin ground
                    # truth), en vez de marcar "acierto/error" se marca "coincide/discrepa con la
                    # mayoría de los 3 modelos" — el mismo lenguaje visual que en el Simulador.
                    conteo_clases = Counter(r["defecto"] for r in resultados_manual.values())
                    clase_mayoritaria, votos_mayoria = conteo_clases.most_common(1)[0]
                    hay_unanimidad = votos_mayoria == len(resultados_manual)

                    st.subheader("Resultados de la Auditoría (comparativa de los 3 modelos)")
                    if hay_unanimidad:
                        st.success(f"✅ Los 3 modelos coinciden: **{clase_mayoritaria}**")
                    else:
                        st.warning(f"⚠️ Los modelos no coinciden entre sí. Mayoría ({votos_mayoria}/{len(resultados_manual)}): **{clase_mayoritaria}**")

                    GRIS_NEUTRO = "#616161"
                    columnas_manual = st.columns(1 + len(MODEL_CONFIGS))

                    with columnas_manual[0]:
                        st.markdown(f"""
                        <div style="border:1px solid {GRIS_NEUTRO}; padding:6px 10px; border-radius:6px; font-size:1rem; text-align:center; line-height:1.35; white-space:nowrap; overflow-x:auto;">
                            <strong>📷 Imagen Subida</strong><br>
                            <span style="color:{GRIS_NEUTRO}; font-weight:bold;">ENTRADA</span><br>
                            Mayoría: {clase_mayoritaria}
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(imagen_subida, caption="Superficie", width='stretch')

                    for col, nombre_modelo in zip(columnas_manual[1:], MODEL_CONFIGS.keys()):
                        with col:
                            res = resultados_manual.get(nombre_modelo)
                            if res is None:
                                st.warning(f"{nombre_modelo}: no disponible")
                                continue
                            estado_modelo = "Aceptada" if res["defecto"] == 'Normal (Sin defectos)' else "Rechazada (Defecto)"
                            color_m = "green" if estado_modelo == "Aceptada" else "red"
                            de_acuerdo = res["defecto"] == clase_mayoritaria
                            if de_acuerdo:
                                borde_tarjeta = f"2px solid {color_m}"
                                fondo_tarjeta = "transparent"
                                linea_consenso = '<span style="color:green; font-weight:bold;">✅ Coincide</span>'
                            else:
                                borde_tarjeta = "3px solid #ff9800"
                                fondo_tarjeta = "rgba(255,152,0,0.12)"
                                linea_consenso = '<span style="color:#e65100; font-weight:bold;">⚠️ Discrepa</span>'
                            st.markdown(f"""
                            <div style="border:{borde_tarjeta}; background:{fondo_tarjeta}; padding:6px 10px; border-radius:6px; font-size:1rem; text-align:center; line-height:1.35; white-space:nowrap; overflow-x:auto;">
                                <strong>🧠 {nombre_modelo}</strong><br>
                                <span style="color:{color_m}; font-weight:bold;">{estado_modelo}</span> · {res['defecto']} ({res['confianza']:.1f}%)<br>
                                {linea_consenso}
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(res["imagen_gradcam"], caption="Grad-CAM", width='stretch')
                except Exception as e:
                    st.error(f"Error durante el análisis: {e}")
        else:
            st.info("Esperando carga de imagen... Por favor, selecciona un archivo arriba.")