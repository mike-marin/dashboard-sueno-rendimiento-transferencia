# 🧠 Dashboard Interactivo: Sueño y Rendimiento Académico

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mike-marin/dashboard-sueno-rendimiento-transferencia/main)

**Actividad de Transferencia** — *Programación para Ciencia de Datos II*  
**Fundación Universitaria Compensar** — Ingeniería en Ciencia de Datos (2026)  

- **Autor:** Michael Marín Herrera (`mmarinh@ucompensar.edu.co`)
- **Docente:** William Eduardo Clavijo Bohorquez

---

## 📁 Estructura del Repositorio

```text
sueno_rendimiento/
├── assets/
│   └── style.css                    # Estilos visuales personalizados (CSS)
├── data/
│   └── sueno_rendimiento.csv        # Dataset original (70 estudiantes)
├── .streamlit/
│   └── config.toml                  # Configuración de tema visual (Emerald / Slate)
├── .gitignore                       # Filtro de archivos locales/temporales
├── README.md                        # Documentación principal del proyecto
├── app.py                           # Punto de entrada complementario
├── dashboard_binder.ipynb           # Notebook Jupyter reproducible para Binder / Colab
├── dashboard_sueno_rendimiento.py   # Código de la aplicación interactiva Streamlit
├── enlaces_entrega.txt              # Enlaces institucionales de entrega
├── requirements.txt                 # Dependencias del proyecto
└── runtime.txt                      # Versión de entorno para despliegue
```

---

## 🚀 Cómo Ejecutar el Proyecto Localmente

Para ejecutar el dashboard en tu computadora (Windows / macOS / Linux), sigue estos sencillos pasos:

### 1. (Opcional recomendado) Crear y activar un entorno virtual:
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el dashboard:
```bash
streamlit run dashboard_sueno_rendimiento.py
```
*(O también: `streamlit run app.py`)*

El navegador se abrirá automáticamente en: **`http://localhost:8501`**

---

## 📊 Contenido y Funcionalidades del Dashboard

El dashboard cuenta con un panel lateral de filtros globales y **4 pestañas analíticas e interactivas**:

1. **Filtros Globales (Sidebar):**
   - Slider de rango de horas de sueño.
   - Checkbox interactivo para incluir o excluir al estudiante atípico (ID=9, puntaje=22).

2. **1. Resumen Exploratorio:**
   - Tarjetas de KPIs analíticos ($n$, horas de sueño promedio, puntaje promedio, correlación $r$).
   - Histograma de distribución de calificaciones con línea de referencia de la media.
   - Diagrama de dispersión dinámico coloreado por horas de estudio o nivel de estrés.
   - Tabla descriptiva completa con métricas estadísticas.

3. **2. Contraste de Hipótesis:**
   - Evaluación formal de:
     $$\begin{cases} H_0: \mu_{\text{alto}} \le \mu_{\text{bajo}} \\ H_1: \mu_{\text{alto}} > \mu_{\text{bajo}} \end{cases}$$
   - Pruebas estadísticas automáticas: **$t$ de Welch** y **Mann-Whitney $U$**.
   - Sliders en vivo para variar el percentil de corte (10% al 90%) y el nivel de significancia $\alpha$.
   - Gráficos boxplot comparativos y tarjeta de conclusión diagnóstica.

4. **3. Regresión Lineal Múltiple:**
   - Selección interactiva de variables predictoras (`horas_sueno`, `horas_estudio`, `nivel_estres`).
   - Ajuste de proporción de partición de prueba (Train/Test Split).
   - Métricas de desempeño en tiempo real: $R^2$ y RMSE.
   - Gráfico interactivo de valores reales vs. predichos.
   - **Simulador de Rendimiento Individual** en vivo.

5. **4. Regresión Logística y Riesgo:**
   - Definición de estudiantes en riesgo académico (puntaje $\le$ mediana).
   - Ajuste de umbral de probabilidad de corte y parámetro de regularización $C$.
   - Matriz de confusión visual interactiva y curva sigmoide de probabilidad.
   - **Calculadora de Diagnóstico de Riesgo Estudiantil** individual.

---

## 🌐 Despliegue en la Nube y Ejecución en Binder

- **Streamlit Community Cloud:** Despliegue automático conectado con GitHub.
- **Binder:** Abre el repositorio de manera interactiva sin instalar nada localmente haciendo clic en la insignia superior de Binder o en `dashboard_binder.ipynb`.

---
© 2026 Michael Marín Herrera — Fundación Universitaria Compensar
