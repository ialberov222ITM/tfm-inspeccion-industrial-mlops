
import os
import time
import random
import glob
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.cm as cm

# ==========================================
# 0. CONFIGURACIÓN INICIAL Y ESTILO
# ==========================================
st.set_page_config(page_title="Planta Industrial IA", page_icon="🏭", layout="wide")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Mismo mapeo de clases que en 05_app_despliegue_streamlit / 04_gradcam_autocalibrado
CLASS_NAMES = {0: 'Crack (Grieta)', 1: 'Hole (Perforación)', 2: 'Normal (Sin defectos)', 3: 'Rust (Óxido)', 4: 'Scratch (Arañazo)'}
# Mapea el nombre de la subcarpeta del dataset (etiqueta real) al nombre mostrado en pantalla
FOLDER_TO_CLASS_NAME = {'crack': CLASS_NAMES[0], 'hole': CLASS_NAMES[1], 'normal': CLASS_NAMES[2], 'rust': CLASS_NAMES[3], 'scratch': CLASS_NAMES[4]}

# Los 3 modelos entrenados en 03_entrenamiento_comparativo_tl.ipynb, con la capa
# convolucional objetivo de Grad-CAM correcta para cada arquitectura.
MODEL_CONFIGS = {
    "EfficientNetB0": {"path": "modelo_optimo_efficientnetb0.keras", "last_conv_layer": "top_activation"},
    "ResNet50": {"path": "modelo_optimo_resnet50.keras", "last_conv_layer": "conv5_block3_out"},
    "MobileNetV2": {"path": "modelo_optimo_mobilenetv2.keras", "last_conv_layer": "out_relu"},
}
TARGET_SIZE = (224, 224)

# Límite de duración de la simulación: se detiene sola al cumplirse este tiempo
DURACION_SIMULACION_SEG = 10 * 60  # 10 minutos

# Carpeta de piezas "entrantes": imágenes reales de validación (no vistas en entrenamiento)
DATASET_PATH = os.path.join("..", "industrial_defect_dataset", "val")

DB_FILE = 'produccion_planta.db'

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
    conn.commit()
    conn.close()

def insertar_registro(linea, defecto, confianza, estado, clase_real, estado_real, modelo_ia):
    """Inserta una nueva pieza procesada en la base de datos, junto con su
    etiqueta real (ground truth del dataset) y el modelo de IA que la clasificó."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO produccion (timestamp, linea, defecto, confianza, estado, clase_real, estado_real, modelo_ia)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, linea, defecto, confianza, estado, clase_real, estado_real, modelo_ia))
    conn.commit()
    conn.close()

def cargar_datos():
    """Carga los datos de producción en un DataFrame de Pandas."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM produccion", conn)
    conn.close()
    return df

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

# Instante en que se activó la simulación (None = simulación detenida), para poder pararla sola tras 10 minutos
if 'tiempo_inicio_simulacion' not in st.session_state:
    st.session_state.tiempo_inicio_simulacion = None

# ==========================================
# 3. INTERFAZ DE NAVEGACIÓN Y SELECCIÓN DE MODELO (SIDEBAR)
# ==========================================
st.sidebar.title("🏭 Navegación MLOps")
st.sidebar.markdown("Selecciona el módulo a visualizar:")
pagina = st.sidebar.radio("Módulos", ["Simulador de Planta (En Vivo)", "Dashboard de Resultados", "Inspección Manual (Subir Imagen)"])

st.sidebar.divider()
st.sidebar.markdown("### 🧠 Modelo de IA")
nombre_modelo_seleccionado = st.sidebar.selectbox(
    "Modelo entrenado (Transfer Learning)",
    list(MODEL_CONFIGS.keys()),
    help="Los 3 modelos comparados en 03_entrenamiento_comparativo_tl.ipynb. Cambiar de modelo se aplica a la siguiente pieza inspeccionada."
)
config_modelo_activo = MODEL_CONFIGS[nombre_modelo_seleccionado]

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

@st.cache_data
def listar_imagenes_dataset():
    """Indexa una sola vez las imágenes reales de validación disponibles como 'piezas entrantes'."""
    rutas = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        rutas.extend(glob.glob(os.path.join(DATASET_PATH, "**", ext), recursive=True))
    return rutas

modelo = load_classification_model(config_modelo_activo["path"])
imagenes_dataset = listar_imagenes_dataset()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Calcula el mapa de calor matemático de los gradientes (Grad-CAM real)."""
    model.layers[-1].activation = None

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    model.layers[-1].activation = tf.keras.activations.softmax

    return heatmap.numpy(), int(pred_index.numpy()), preds[0]

def create_superimposed_image(img_pil, heatmap, alpha=0.5):
    """Superpone el mapa térmico sobre la imagen original utilizando un colormap Jet."""
    img_array = img_to_array(img_pil)

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img_array
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

def inspeccionar_pieza_real(modelo, last_conv_layer):
    """Toma una imagen real aleatoria del dataset de validación, la clasifica con
    el modelo entrenado seleccionado y genera su Grad-CAM real, igual que en
    05_app_despliegue_streamlit. Como la imagen procede de una carpeta con etiqueta
    conocida, también se devuelve la clase real (ground truth) para poder auditar
    falsos positivos/negativos."""
    img_path = random.choice(imagenes_dataset)
    carpeta_real = os.path.basename(os.path.dirname(img_path)).lower()
    clase_real = FOLDER_TO_CLASS_NAME.get(carpeta_real, carpeta_real)
    estado_real = "Aceptada" if clase_real == 'Normal (Sin defectos)' else "Rechazada (Defecto)"

    img_original = load_img(img_path, target_size=TARGET_SIZE)
    img_array = img_to_array(img_original)
    img_batch = np.expand_dims(img_array, axis=0)

    heatmap, pred_index, preds = make_gradcam_heatmap(img_batch, modelo, last_conv_layer)
    confianza = float(tf.nn.softmax(preds)[pred_index]) * 100
    defecto_predicho = CLASS_NAMES.get(pred_index, f"Clase_{pred_index}")
    img_gradcam = create_superimposed_image(img_original, heatmap)

    return img_original, img_gradcam, defecto_predicho, confianza, clase_real, estado_real

# ==========================================
# 5. PÁGINA 1: SIMULADOR DE PLANTA EN VIVO
# ==========================================
if pagina == "Simulador de Planta (En Vivo)":
    st.title("🏭 Planta de Producción - Monitor en Vivo")
    st.markdown(f"Visualización en tiempo real de las 5 líneas de ensamblaje. El modelo **{nombre_modelo_seleccionado}** inspecciona cada pieza de forma automatizada y renderiza su auditoría XAI instantáneamente.")

    # Si la simulación llevaba encendida más de DURACION_SIMULACION_SEG, se apaga sola
    # ANTES de dibujar el toggle (no se puede tocar st.session_state de un widget ya instanciado).
    simulacion_detenida_por_tiempo = False
    if (st.session_state.tiempo_inicio_simulacion is not None and
            time.time() - st.session_state.tiempo_inicio_simulacion >= DURACION_SIMULACION_SEG):
        st.session_state["simulacion_activa_widget"] = False
        st.session_state.tiempo_inicio_simulacion = None
        simulacion_detenida_por_tiempo = True

    simulacion_activa = st.toggle("▶️ Activar Simulación de Líneas de Producción", key="simulacion_activa_widget")

    if simulacion_activa and st.session_state.tiempo_inicio_simulacion is None:
        st.session_state.tiempo_inicio_simulacion = time.time()  # se acaba de activar: arranca el cronómetro
    elif not simulacion_activa:
        st.session_state.tiempo_inicio_simulacion = None

    if simulacion_detenida_por_tiempo:
        st.info("⏱️ Simulación detenida automáticamente tras 10 minutos.")
    elif simulacion_activa:
        restante = max(0, DURACION_SIMULACION_SEG - (time.time() - st.session_state.tiempo_inicio_simulacion))
        st.caption(f"⏱️ Tiempo restante antes de la parada automática: {int(restante // 60)}:{int(restante % 60):02d}")

    if modelo is None:
        st.error("No se puede simular: el modelo no se cargó correctamente.")
    elif not imagenes_dataset:
        st.error(f"No se encontraron imágenes en '{DATASET_PATH}'. Verifica la ruta del dataset.")

    # Contenedores para las 5 líneas
    columnas_lineas = st.columns(5)

    # Evaluar si toca procesar alguna pieza
    if simulacion_activa and modelo is not None and imagenes_dataset:
        tiempo_actual = time.time()

        for i in range(1, 6):
            if tiempo_actual >= st.session_state.lineas[i]["proximo_procesamiento"]:
                # --- INFERENCIA REAL DE IA SOBRE UNA PIEZA (imagen real del dataset) ---
                (img_original_pieza, img_gradcam_pieza, defecto_predicho, confianza_prediccion,
                 clase_real, estado_real) = inspeccionar_pieza_real(modelo, config_modelo_activo["last_conv_layer"])
                estado_pieza = "Aceptada" if defecto_predicho == 'Normal (Sin defectos)' else "Rechazada (Defecto)"

                # Guardar en Base de Datos (incluyendo la etiqueta real y el modelo, para auditar FP/FN por modelo)
                insertar_registro(i, defecto_predicho, confianza_prediccion, estado_pieza, clase_real, estado_real, nombre_modelo_seleccionado)

                # Actualizar estado de la sesión
                resultado = {
                    "linea": i,
                    "defecto": defecto_predicho,
                    "confianza": confianza_prediccion,
                    "estado": estado_pieza,
                    "hora": datetime.now().strftime("%H:%M:%S"),
                    "clase_real": clase_real,
                    "estado_real": estado_real,
                    "modelo_ia": nombre_modelo_seleccionado,
                }
                st.session_state.lineas[i]["ultimo_resultado"] = resultado
                st.session_state.lineas[i]["piezas_totales"] += 1
                # Esta es la última pieza producida en toda la planta: la que se muestra en grande
                st.session_state.ultima_pieza_global = {
                    **resultado,
                    "imagen_original": img_original_pieza,
                    "imagen_gradcam": img_gradcam_pieza
                }
                # Configurar el tiempo para la SIGUIENTE pieza (aleatorio entre 8 y 20 segs)
                st.session_state.lineas[i]["proximo_procesamiento"] = tiempo_actual + random.uniform(8, 20)

    # Renderizar UI de las líneas (solo estado y métricas, sin imagen individual)
    for i, col in enumerate(columnas_lineas, 1):
        with col:
            st.markdown(f"### Línea {i}")
            estado_linea = st.session_state.lineas[i]

            st.metric("Total procesadas", estado_linea["piezas_totales"])

            # Caja de estado visual
            if simulacion_activa:
                st.info("🔄 Cinta en movimiento...")
            else:
                st.warning("⏸️ Cinta detenida")

            # Mostrar último resultado (texto) de esta línea
            if estado_linea["ultimo_resultado"]:
                res = estado_linea["ultimo_resultado"]
                color = "green" if res["estado"] == "Aceptada" else "red"
                st.markdown(f"""
                <div style="border:1px solid {color}; padding:10px; border-radius:5px; margin-bottom: 10px;">
                    <strong>Última pieza ({res['hora']}):</strong><br>
                    <span style="color:{color}; font-weight:bold;">{res['estado']}</span><br>
                    Defecto: {res['defecto']}<br>
                    Confianza IA: {res['confianza']:.1f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='border:1px dashed gray; padding:10px; border-radius:5px; text-align:center;'>Esperando primera pieza...</div>", unsafe_allow_html=True)

    st.divider()

    # ==========================================
    # Pieza activa: auditoría visual (Original vs Grad-CAM)
    # ==========================================
    st.subheader("🔎 Última Pieza Inspeccionada (Auditoría Visual)")

    pieza = st.session_state.ultima_pieza_global
    if pieza:
        color = "green" if pieza["estado"] == "Aceptada" else "red"
        acierto = pieza["defecto"] == pieza["clase_real"]
        etiqueta_acierto = "✅ Predicción correcta" if acierto else "⚠️ Predicción incorrecta (ver etiqueta real)"
        st.markdown(f"""
        <div style="border:1px solid {color}; padding:10px; border-radius:5px; margin-bottom: 10px;">
            <strong>Línea {pieza['linea']} · {pieza['hora']} · Modelo: {pieza['modelo_ia']}:</strong>
            <span style="color:{color}; font-weight:bold;"> {pieza['estado']}</span><br>
            Defecto (predicción IA): {pieza['defecto']}<br>
            Confianza IA: {pieza['confianza']:.1f}%<br>
            Clase real (dataset): {pieza['clase_real']}<br>
            {etiqueta_acierto}
        </div>
        """, unsafe_allow_html=True)

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(pieza["imagen_original"], caption="1. Superficie Original", use_container_width=True)
        with col_img2:
            st.image(pieza["imagen_gradcam"], caption="2. Mapa de Activación Grad-CAM (Zona de decisión)", use_container_width=True)
    else:
        st.markdown("<div style='border:1px dashed gray; padding:20px; border-radius:5px; text-align:center;'>Esperando la primera pieza de cualquier línea...</div>", unsafe_allow_html=True)

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
        # Filtro por modelo de IA: cada modelo puede tener una fiabilidad distinta,
        # así que las métricas no deben mezclar piezas inspeccionadas por modelos diferentes.
        modelos_en_bd = sorted(df['modelo_ia'].dropna().unique().tolist())
        filtro_modelo = st.selectbox("🧠 Filtrar por modelo de IA", ["Todos los modelos"] + modelos_en_bd)
        if filtro_modelo != "Todos los modelos":
            df = df[df['modelo_ia'] == filtro_modelo]

        # 6.1. KPIs (Key Performance Indicators) Principales
        st.subheader("Indicadores Globales")
        col1, col2, col3, col4 = st.columns(4)
        
        total_piezas = len(df)
        piezas_rechazadas = len(df[df['estado'].str.contains("Rechazada")])
        porcentaje_rechazo = (piezas_rechazadas / total_piezas) * 100 if total_piezas > 0 else 0
        
        # Encontrar la línea con más rechazos
        rechazos_por_linea = df[df['estado'].str.contains("Rechazada")]['linea'].value_counts()
        linea_problematica = rechazos_por_linea.idxmax() if not rechazos_por_linea.empty else "N/A"
        
        col1.metric("Total Piezas Analizadas", total_piezas)
        col2.metric("Piezas Rechazadas", piezas_rechazadas)
        col3.metric("Tasa de Defectos", f"{porcentaje_rechazo:.1f}%")
        col4.metric("Línea con más fallos", f"Línea {linea_problematica}")

        # 6.1.bis Fiabilidad del modelo: Falsos Positivos y Falsos Negativos
        # (comparando la predicción de la IA contra la etiqueta real del dataset)
        df_auditado = df.dropna(subset=['estado_real'])
        st.markdown("#### Fiabilidad del Modelo (predicción IA vs. etiqueta real)")
        if df_auditado.empty:
            st.info("Aún no hay piezas con etiqueta real registrada. Procesa piezas nuevas en el simulador para calcular estos indicadores.")
        else:
            piezas_sanas_reales = len(df_auditado[df_auditado['estado_real'] == 'Aceptada'])
            piezas_defecto_reales = len(df_auditado[df_auditado['estado_real'] == 'Rechazada (Defecto)'])

            falsos_positivos = len(df_auditado[(df_auditado['estado_real'] == 'Aceptada') & (df_auditado['estado'] == 'Rechazada (Defecto)')])
            falsos_negativos = len(df_auditado[(df_auditado['estado_real'] == 'Rechazada (Defecto)') & (df_auditado['estado'] == 'Aceptada')])

            tasa_fp = (falsos_positivos / piezas_sanas_reales * 100) if piezas_sanas_reales > 0 else 0
            tasa_fn = (falsos_negativos / piezas_defecto_reales * 100) if piezas_defecto_reales > 0 else 0

            col_fp, col_fn = st.columns(2)
            col_fp.metric("Falsos Positivos", falsos_positivos, delta=f"{tasa_fp:.1f}% de piezas sanas", delta_color="off",
                          help="Piezas realmente SIN defecto que la IA rechazó por error (falsa alarma).")
            col_fn.metric("Falsos Negativos", falsos_negativos, delta=f"{tasa_fn:.1f}% de piezas defectuosas", delta_color="off",
                          help="Piezas realmente CON defecto que la IA aceptó por error (el caso más crítico: un defecto se cuela en producción).")

            # --- Desglose detallado: exactamente qué tipos de pieza confunde el modelo ---
            df_fp = df_auditado[(df_auditado['estado_real'] == 'Aceptada') & (df_auditado['estado'] == 'Rechazada (Defecto)')]
            df_fn = df_auditado[(df_auditado['estado_real'] == 'Rechazada (Defecto)') & (df_auditado['estado'] == 'Aceptada')]

            if falsos_positivos > 0 or falsos_negativos > 0:
                st.markdown("##### 🔬 Desglose detallado de errores por tipo de pieza")

                col_graf_fn, col_graf_fp = st.columns(2)
                with col_graf_fn:
                    st.caption("Falsos Negativos: qué defecto real se le coló a la IA")
                    if not df_fn.empty:
                        conteo_fn = df_fn['clase_real'].value_counts().rename_axis('Tipo de defecto real').reset_index(name='Veces no detectado')
                        fig_fn = px.bar(conteo_fn, x='Tipo de defecto real', y='Veces no detectado',
                                        color_discrete_sequence=['#d62728'])
                        st.plotly_chart(fig_fn, use_container_width=True)
                    else:
                        st.write("Sin falsos negativos con los filtros actuales.")
                with col_graf_fp:
                    st.caption("Falsos Positivos: qué defecto 'vio' la IA sin existir")
                    if not df_fp.empty:
                        conteo_fp = df_fp['defecto'].value_counts().rename_axis('Defecto predicho por error').reset_index(name='Veces')
                        fig_fp = px.bar(conteo_fp, x='Defecto predicho por error', y='Veces',
                                        color_discrete_sequence=['#ff7f0e'])
                        st.plotly_chart(fig_fp, use_container_width=True)
                    else:
                        st.write("Sin falsos positivos con los filtros actuales.")

                tipo_error_filtro = st.radio(
                    "Ver registros individuales de:",
                    ["Ambos", "Solo Falsos Positivos", "Solo Falsos Negativos"],
                    horizontal=True
                )
                if tipo_error_filtro == "Solo Falsos Positivos":
                    df_errores = df_fp.copy()
                elif tipo_error_filtro == "Solo Falsos Negativos":
                    df_errores = df_fn.copy()
                else:
                    df_errores = pd.concat([df_fp, df_fn]).copy()

                if df_errores.empty:
                    st.success("No hay errores de este tipo con los filtros actuales.")
                else:
                    df_errores['tipo_error'] = np.where(df_errores['estado_real'] == 'Aceptada', 'Falso Positivo', 'Falso Negativo')
                    columnas_mostrar = ['timestamp', 'linea', 'modelo_ia', 'tipo_error', 'clase_real', 'defecto', 'confianza']
                    st.dataframe(df_errores[columnas_mostrar].sort_values(by='timestamp', ascending=False), use_container_width=True)

        st.divider()
        
        # 6.2. Gráficos Analíticos (Plotly)
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            # Gráfico de Torta: Distribución de tipos de defectos
            st.markdown("#### Distribución de Anomalías")
            df_defectos = df[df['defecto'] != 'Normal (Sin defectos)']
            if not df_defectos.empty:
                fig_pie = px.pie(df_defectos, names='defecto', hole=0.4, 
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.write("Aún no se han detectado defectos.")
                
        with col_graf2:
            # Gráfico de Barras: Aceptadas vs Rechazadas por Línea
            st.markdown("#### Rendimiento por Línea de Producción")
            fig_bar = px.histogram(df, x="linea", color="estado", barmode="group",
                                   category_orders={"estado": ["Aceptada", "Rechazada (Defecto)"]},
                                   color_discrete_map={"Aceptada": "green", "Rechazada (Defecto)": "red"})
            fig_bar.update_layout(xaxis_title="Número de Línea", yaxis_title="Cantidad de Piezas")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # 6.3. Tabla de Datos Crudos
        st.subheader("Registro Histórico de Auditoría")
        # Mostrar los últimos 100 registros ordenados por el más reciente
        st.dataframe(df.sort_values(by="id", ascending=False).head(100), use_container_width=True)

# ==========================================
# 7. PÁGINA 3: INSPECCIÓN MANUAL (idéntico a 05_app_despliegue_streamlit, integrado en la app)
# ==========================================
elif pagina == "Inspección Manual (Subir Imagen)":
    st.title("🔍 Inspección Manual de Piezas")
    st.markdown(f"Sube una imagen de una superficie metálica para evaluarla con el modelo **{nombre_modelo_seleccionado}** (seleccionable en la barra lateral). "
                "El sistema generará una auditoría visual explicando su decisión, igual que en 05_app_despliegue_streamlit.")

    if modelo is None:
        st.error("No se puede inspeccionar: el modelo no se cargó correctamente.")
    else:
        uploaded_file = st.file_uploader("Cargar imagen de inspección (JPG/PNG)...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            imagen_subida = Image.open(uploaded_file).convert('RGB')
            st.sidebar.image(imagen_subida, caption="Imagen de entrada original", use_container_width=True)

            img_resized = imagen_subida.resize(TARGET_SIZE)
            img_array = img_to_array(img_resized)
            img_batch = np.expand_dims(img_array, axis=0)

            with st.spinner('Analizando topología superficial...'):
                try:
                    # Inferencia pura (con la activación softmax intacta) para la confianza mostrada
                    preds_completas = modelo.predict(img_batch)
                    pred_index_softmax = int(np.argmax(preds_completas[0]))
                    confianza_softmax = float(preds_completas[0][pred_index_softmax]) * 100

                    # Grad-CAM real sobre la misma imagen
                    heatmap, _, _ = make_gradcam_heatmap(img_batch, modelo, config_modelo_activo["last_conv_layer"])
                    img_xai = create_superimposed_image(img_resized, heatmap)

                    st.subheader("Resultados de la Auditoría")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Clasificación Principal", CLASS_NAMES.get(pred_index_softmax, f"Clase_{pred_index_softmax}"))
                    with col2:
                        st.metric("Confianza del Modelo", f"{confianza_softmax:.2f} %")

                    st.markdown("### 🔎 Inspección por Rayos X (Grad-CAM)")
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.image(imagen_subida, caption="1. Superficie Original", use_container_width=True)
                    with col_img2:
                        st.image(img_xai, caption="2. Mapa de Activación (Zona de decisión)", use_container_width=True)

                    st.markdown("### 📊 Desglose de Probabilidades (Capa Softmax)")
                    probs_dict = {CLASS_NAMES[i]: float(preds_completas[0][i]) * 100 for i in range(len(CLASS_NAMES))}
                    st.bar_chart(probs_dict)
                except Exception as e:
                    st.error(f"Error durante el análisis: {e}")
        else:
            st.info("Esperando carga de imagen... Por favor, selecciona un archivo arriba.")