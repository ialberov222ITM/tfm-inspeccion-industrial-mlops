🏭 MLOps y Deep Learning para Inspección de Defectos Industriales

Repositorio oficial del Trabajo de Fin de Máster centrado en la detección y clasificación multiclase de defectos en superficies metálicas mediante arquitecturas CNN (Transfer Learning), Inteligencia Artificial Explicable (XAI) y el despliegue de un simulador MLOps en tiempo real.

📺 Demostración del Gemelo Digital (Vídeo)

Para evaluar el funcionamiento de la monitorización, la inferencia IA y la auditoría visual interactiva mediante Grad-CAM, por favor consulte la siguiente demostración:

👉 Ver vídeo demostrativo en YouTube

📥 Nota sobre Modelos Pesados: Los pesos .keras de las redes superan el límite de GitHub. Pueden descargarse desde este Enlace a Google Drive y deben colocarse en la carpeta models/.

📂 Estructura del Repositorio

data/sample_images/: Muestra representativa de imágenes sintéticas para pruebas locales del simulador.

models/: Directorio destinado a alojar los pesos .keras de los modelos óptimos.

notebooks/: Cuadernos Jupyter con el EDA, el Data Augmentation y el entrenamiento comparativo (ResNet50, MobileNetV2, EfficientNetB0). Incluye exportaciones HTML sin warnings.

results/: Evidencias visuales de rendimiento (Matrices de Confusión) y auditoría Grad-CAM seleccionada.

## 🚀 Arquitectura y Versiones de la Aplicación

Para garantizar la reproducibilidad del proyecto en diferentes entornos sin comprometer el rendimiento, el código fuente se ha estructurado en dos versiones funcionales. Ambas ejecutan el núcleo del Gemelo Digital y los modelos de Visión Artificial, pero difieren en su gestión de datos y almacenamiento:

### 1. Versión de Evaluación / Reproducibilidad (Recomendada)
- **Archivo:** `src/app_planta_industrial.py`
- **Descripción:** Es la versión "ligera" del prototipo. Utiliza el conjunto de datos de muestra reducido (`data/sample_images/`) y **desactiva por defecto el guardado físico de imágenes en disco y base de datos**. 
- **Objetivo:** Está diseñada específicamente para que el tribunal u otros investigadores puedan clonar, instalar y ejecutar el simulador rápidamente en sus equipos sin requerir grandes capacidades de almacenamiento local.
- **Ejecución:** ```bash
  streamlit run src/app_planta_industrial.py

### 2. Versión Completa / Entorno de Producción
- **Archivo:** `src/version_completa/app_planta_industrialcompleta.py`
- **Descripción:** Es la versión íntegra del sistema utilizada para las pruebas de estrés y la grabación del vídeo demostrativo. Esta versión se conecta al repositorio masivo de imágenes locales, habilita la persistencia en la base de datos y guarda de forma permanente las evidencias de los falsos positivos y negativos generados por los modelos.
- **Objetivo:** Demostrar la viabilidad del sistema en un entorno industrial real, simulando un pipeline MLOps completo con trazabilidad absoluta de los datos.
 **Ejecución:** ```bash
  streamlit run src/version_completa/app_planta_industrialcompleta.py

gradcam_autocalibrado.py: Script automatizado para auditoría visual y explicabilidad en lote.

⚙️ Instalación y Uso Local

Clona este repositorio:

git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
cd TU_REPOSITORIO


Instala las dependencias necesarias:

pip install -r requirements.txt


Ejecuta el simulador de planta industrial:

streamlit run src/app_planta_industrial.py




🧠 Arquitectura de la Solución

El proyecto integra el ciclo de vida completo del modelo (MLOps):

Inferencia: Evaluación comparativa con tres redes preentrenadas y adaptadas a la morfología de defectos metálicos sintéticos.

Explicabilidad (XAI): Mapeo de activación térmica Grad-CAM para auditar el campo receptivo de la red y mitigar el efecto de "caja negra".

Productivización: Gemelo Digital con registro en base de datos SQLite y cálculo en tiempo real de KPIs industriales (Falsos Positivos / Falsos Negativos).

Desarrollado por Ismael Albero Verdú - Máster en Inteligencia Artificial.