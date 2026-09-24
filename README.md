# Dashboard Interactivo: Sueño y Rendimiento Académico

**Actividad de Transferencia** — Programación para Ciencia de Datos II  
**Fundación Universitaria Compensar** — Ingeniería en Ciencia de Datos (2026)  

- **Autor:** Michael Marín Herrera (`mmarinh@ucompensar.edu.co`)
- **Docente:** William Eduardo Clavijo Bohorquez

---

## 📁 Estructura del Repositorio

```text
sueno_rendimiento/
├── .streamlit/
│   └── config.toml                  # Configuración de tema visual (Emerald / Slate)
├── data/
│   └── sueno_rendimiento.csv        # Dataset con 70 estudiantes
├── sueno_rendimiento.csv            # Copia en raíz para máxima compatibilidad
├── dashboard_sueno_rendimiento.py   # Código de la aplicación interactiva Streamlit
├── app.py                           # Punto de entrada complementario
├── requirements.txt                 # Dependencias del proyecto
├── .gitignore                       # Filtro de archivos locales/temporales
└── README.md                        # Documentación
```

---

## 🚀 Cómo Ejecutar el Proyecto Localmente

Para ejecutar el dashboard en tu computadora (Windows / macOS / Linux), abre una terminal en la carpeta del proyecto y sigue estos pasos:

### 1. (Opcional recomendado) Crear y activar entorno virtual:
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

El navegador se abrirá automáticamente en: **http://localhost:8501**

---

## 📊 Contenido y Funcionalidades del Dashboard

El dashboard cuenta con un panel lateral de filtros globales y 4 secciones analíticas interactivas:

1. **Filtros Globales (Sidebar):** Rango de horas de sueño e inclusión/exclusión del estudiante atípico (ID=9, puntaje=22 pts).
2. **1. Resumen Exploratorio:** Métricas clave (n, sueño promedio, puntaje promedio, correlación r), histograma de calificaciones con línea de media, dispersograma dinámico coloreado por horas de estudio o nivel de estrés, y tabla descriptiva completa.
3. **2. Contraste de Hipótesis:** Evaluación de $H_0: \mu_{\text{alto}} \le \mu_{\text{bajo}}$ vs. $H_1: \mu_{\text{alto}} > \mu_{\text{bajo}}$ mediante prueba $t$ de Welch y Mann-Whitney $U$. Permite variar el percentil de corte (10 a 90) y el nivel de significancia $\alpha$, con boxplots comparativos y decisión automática.
4. **3. Regresión Lineal Múltiple:** Selección interactiva de variables predictoras, proporción de test, cálculo en vivo de $R^2$ y RMSE, gráfico de valores reales vs. predichos y un **Simulador de Rendimiento Individual** en tiempo real.
5. **4. Regresión Logística y Riesgo:** Detección de estudiantes en riesgo (puntaje $\le$ mediana), umbral de corte de probabilidad ajustable, regularización $C$, matriz de confusión, curva sigmoide y una **Calculadora de Diagnóstico de Riesgo Estudiantil**.
