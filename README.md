# 🏭 MLOps y Deep Learning para Inspección de Defectos Industriales

Repositorio oficial del Trabajo de Fin de Máster centrado en la detección y clasificación multiclase de defectos en superficies metálicas mediante arquitecturas CNN (*Transfer Learning*), Inteligencia Artificial Explicable (XAI) y el despliegue de un simulador MLOps en tiempo real.

---

## 📺 Demostración del Gemelo Digital (Vídeo)

Para evaluar el funcionamiento de la monitorización, la inferencia IA y la auditoría visual interactiva mediante Grad-CAM, por favor consulte la siguiente demostración:

👉 **[Ver vídeo demostrativo en YouTube](INSERTAR_ENLACE_DE_YOUTUBE)**

---

## 💾 Descarga de Modelos Entrenados (.keras)

> **Nota sobre Modelos Pesados:** Los pesos `.keras` de las redes superan el límite estándar de almacenamiento de GitHub (100 MB). Por ello, los modelos entrenados se almacenan de forma externa.

Para ejecutar la inferencia local con la red principal del estudio (**ResNet50**), descarga el archivo desde el siguiente enlace y colócalo dentro de la carpeta `models/`:

* 🔗 **[Descargar pesos de ResNet50 (resnet50_model.keras) - Google Drive / Releases] https://github.com/ialberov222ITM/tfm-inspeccion-industrial-mlops/releases/download/v1.0.0/modelo_optimo_resnet50.keras

---

## 📂 Estructura del Repositorio

    ├── data/
    │   └── sample_images/          # Muestra representativa de imágenes sintéticas para pruebas locales
    ├── models/                     # Directorio destinado a alojar los pesos .keras (descargar externamente)
    ├── notebooks/                  # Cuadernos Jupyter con EDA, Data Augmentation y entrenamiento
    │   ├── EDA_y_Augmentation.ipynb
    │   └── Entrenamiento_Comparativo.ipynb  # ResNet50, MobileNetV2, EfficientNetB0 (HTML sin warnings)
    ├── results/                    # Evidencias visuales de rendimiento (Matrices de Confusión y Grad-CAM)
    ├── src/                        # Código fuente principal de producción
    │   ├── app_planta_industrial.py         # Aplicación MLOps (Gemelo Digital) desarrollada en Streamlit
    │   └── gradcam_autocalibrado.py         # Script automatizado para auditoría visual y explicabilidad en lote
    ├── requirements.txt            # Dependencias del proyecto
    └── README.md

---

## ⚙️ Instalación y Uso Local

1. **Clona este repositorio:**

    git clone [https://github.com/ialberov222ITM/tfm-inspeccion-industrial-mlops.git](https://github.com/ialberov222ITM/tfm-inspeccion-industrial-mlops.git)
    cd tfm-inspeccion-industrial-mlops

2. **Instala las dependencias necesarias:**

    pip install -r requirements.txt

3. **Descarga y ubica los modelos:**
   * Descarga el archivo de pesos `resnet50_model.keras` desde el enlace superior y guárdalo dentro de la carpeta `models/`.

4. **Ejecuta el simulador de planta industrial:**

    streamlit run src/app_planta_industrial.py

---

## 🧠 Arquitectura de la Solución

El proyecto integra el ciclo de vida completo del modelo (**MLOps**):

* **Inferencia:** Evaluación comparativa con tres redes preentrenadas (**ResNet50**, **EfficientNetB0** y **MobileNetV2**) adaptadas a la morfología de defectos metálicos sintéticos.
* **Explicabilidad (XAI):** Mapeo de activación térmica **Grad-CAM** para auditar el campo receptivo de la red y mitigar el efecto de "caja negra".
* **Productivización:** Gemelo Digital con registro en base de datos **SQLite** y cálculo en tiempo real de KPIs industriales críticos (**Falsos Positivos / Falsos Negativos**).

---

**Desarrollado por Ismael Albero Verdú** · *Máster en Inteligencia Artificial*