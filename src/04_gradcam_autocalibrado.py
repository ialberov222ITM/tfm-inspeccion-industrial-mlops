import os
import warnings

# ==========================================
# 0. BLOQUE SILENCIADOR DE CONSOLA
# ==========================================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import random
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Modo silencioso para no colapsar la memoria gráfica
matplotlib.use('Agg')

# ==========================================
# 1. FUNCIONES CORE DE GRAD-CAM Y VISUALIZACIÓN
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs], 
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy(), pred_index, preds[0]

def get_superimposed_img(img_path, heatmap, alpha=0.4):
    """Genera la imagen superpuesta en crudo sin plotearla para usarla en cuadrículas."""
    img = load_img(img_path)
    img_res = img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = matplotlib.colormaps['jet']
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_res.shape[1], img_res.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img_res
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

# ==========================================
# 2. CONFIGURACIÓN DEL ENTORNO Y MODELOS
# ==========================================
# Calculamos la ruta raíz dinámicamente multiplataforma
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Apuntamos a las carpetas relativas del repositorio
DATASET_PATH = os.path.join(BASE_DIR, "data", "sample_images")
BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "results", "reporte_gradcam")

CLASS_NAMES = {0: 'crack', 1: 'hole', 2: 'normal', 3: 'rust', 4: 'scratch'}

MODELOS_CONFIG = [
    {"nombre": "ResNet50", "ruta": os.path.join(BASE_DIR, "models", "modelo_optimo_resnet50.keras"), "capa_conv": "conv5_block3_out"},
    {"nombre": "MobileNetV2", "ruta": os.path.join(BASE_DIR, "models", "modelo_optimo_mobilenetv2.keras"), "capa_conv": "out_relu"},
    {"nombre": "EfficientNetB0", "ruta": os.path.join(BASE_DIR, "models", "modelo_optimo_efficientnetb0.keras"), "capa_conv": "top_activation"}
]

# ==========================================
# 3. SELECCIÓN DE IMÁGENES Y CARGA DE MODELOS
# ==========================================
if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    print("[INFO] Cargando los 3 modelos en memoria RAM simultáneamente...")
    modelos_cargados = []
    for config in MODELOS_CONFIG:
        if os.path.exists(config["ruta"]):
            print(f"  -> Cargando {config['nombre']}...")
            modelo = tf.keras.models.load_model(config["ruta"])
            
            # Detectar tamaño esperado por este modelo específico
            try:
                _, h, w, _ = modelo.input_shape
                target_size = (h, w) if h is not None else (224, 224)
            except:
                target_size = (224, 224)
                
            modelos_cargados.append({
                "nombre": config["nombre"],
                "modelo": modelo,
                "capa": config["capa_conv"],
                "target_size": target_size
            })
        else:
            print(f"  [ERROR] No se encontró {config['ruta']}")

    print("\n[INFO] Escaneando dataset para seleccionar las 50 imágenes maestras...")
    all_images = []
    extensions = ['*.jpg', '*.jpeg', '*.png']
    for ext in extensions:
        all_images.extend(glob.glob(os.path.join(DATASET_PATH, "**", ext), recursive=True))
        
    random.seed(42)
    all_images.sort() 
    num_samples = min(50, len(all_images))
    selected_images = random.sample(all_images, num_samples)
    print(f"[INFO] Batería de {num_samples} imágenes fijada.\n")

    # ==========================================
    # 4. PROCESAMIENTO UNIFICADO EN CUADRÍCULA (1x4)
    # ==========================================
    exitos = 0
    for i, img_path in enumerate(selected_images):
        class_real = os.path.basename(os.path.dirname(img_path))
        img_name = os.path.basename(img_path)
        
        class_output_dir = os.path.join(BASE_OUTPUT_DIR, class_real)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Crear la figura contenedora de 1 fila y 4 columnas
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # 4.1 Mostrar la imagen original en el primer panel
        img_original = load_img(img_path)
        axes[0].imshow(img_original)
        axes[0].set_title(f"Original ({class_real})", fontsize=16, fontweight='bold')
        axes[0].axis("off")
        
        # 4.2 Evaluar la imagen en los 3 modelos
        for j, config_mod in enumerate(modelos_cargados):
            modelo_actual = config_mod["modelo"]
            capa = config_mod["capa"]
            size = config_mod["target_size"]
            nombre = config_mod["nombre"]
            
            # Preparar imagen para este modelo concreto
            img_raw = load_img(img_path, target_size=size)
            img_array = np.expand_dims(img_to_array(img_raw), axis=0).copy()
            
            modelo_actual.layers[-1].activation = None
            
            try:
                heatmap, pred_index, raw_preds = make_gradcam_heatmap(img_array, modelo_actual, capa)
                modelo_actual.layers[-1].activation = tf.keras.activations.softmax
                
                pred_index_int = int(pred_index)
                class_pred_name = CLASS_NAMES.get(pred_index_int, f"Indice_{pred_index_int}")
                prob_pct = float(tf.nn.softmax(raw_preds)[pred_index_int]) * 100
                
                # Obtener la imagen coloreada y pintarla en su panel
                img_heatmap = get_superimposed_img(img_path, heatmap)
                axes[j+1].imshow(img_heatmap)
                axes[j+1].set_title(f"{nombre}\nPred: {class_pred_name} [{prob_pct:.1f}%]", fontsize=14)
                axes[j+1].axis("off")
                
            except Exception as e:
                axes[j+1].text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center')
                axes[j+1].axis("off")
                
        # Guardar la tira completa de 4 imágenes
        output_filename = f"comparativa_{i+1:02d}_{img_name}"
        target_path = os.path.join(class_output_dir, output_filename)
        
        plt.tight_layout()
        plt.savefig(target_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[{i+1}/{num_samples}] Generado panel para: {img_name}")
        exitos += 1
        
    print(f"\n[FIN] {exitos} paneles comparativos guardados organizados por defecto.")