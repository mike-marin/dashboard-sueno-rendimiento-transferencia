# 🧠 Dashboard — Sueño, Estrés y Rendimiento Académico

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-2.17.1-008DE4?logo=plotly&logoColor=white)](https://dash.plotly.com/)
[![Plotly](https://img.shields.io/badge/Plotly-5.24.1-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mike-marin/dashboard-sueno-rendimiento-transferencia/main?urlpath=lab/tree/dashboard_binder.ipynb)

**Actividad de Transferencia** — *Programación para Ciencia de Datos II*  
**Fundación Universitaria Compensar** — Ingeniería en Ciencia de Datos (2026)  

- **Autor:** Michael Marín Herrera (`mmarinh@ucompensar.edu.co`)  
- **Docente:** William Eduardo Clavijo Bohorquez  

---

## 📁 Contenido del Repositorio

```text
proyecto/
├── app.py                   # Aplicación Dash (dashboard interactivo)
├── dashboard_binder.ipynb   # Notebook para lanzar y ver el dashboard en Binder
├── requirements.txt         # Dependencias de Python (Dash, Scikit-Learn, Proxy)
├── runtime.txt              # Versión de Python para Binder (python-3.11)
├── data/
│   └── sueno_rendimiento.csv # Dataset con 70 estudiantes universitarios
├── assets/
│   └── estilos.css          # Estilos del dashboard (Dash carga esta carpeta automáticamente)
├── enlaces_entrega.txt      # Formato de entrega del proyecto
└── README.md
```

---

## 1. 💻 Ejecutar el Dashboard en su Computador

1. Abra una terminal en la carpeta del proyecto.
2. (Opcional recomendado) Cree y active un entorno virtual:
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
3. Instale las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Inicie la aplicación:
   ```bash
   python app.py
   ```
5. Abra el navegador en **http://127.0.0.1:8050**

---

## 2. 🌐 Publicar y Ver el Dashboard en Binder

1. Vaya a **[https://mybinder.org](https://mybinder.org)**
2. En **GitHub repository name or URL**, pegue la URL de su repositorio:
   `https://github.com/mike-marin/dashboard-sueno-rendimiento-transferencia`
3. Deje la rama en `main`.
4. En **Path to a notebook file (optional)** seleccione **File** y escriba `dashboard_binder.ipynb`.
5. Haga clic en **Launch** 🚀.
6. Una vez abra el cuaderno en Binder, en el menú superior haga clic en **Run → Run All Cells**. El dashboard interactivo se desplegará de inmediato dentro del cuaderno y con enlace a pantalla completa.

---

## 📊 Funcionalidades del Dashboard

El dashboard cuenta con 5 pestañas analíticas:

1. **Contexto & Resumen:** Tarjetas KPIs, resumen ejecutivo, planteamiento del problema institucional y diccionario de variables.
2. **Exploración de Datos (EDA):** Slider de rango de horas de sueño, histograma de notas con línea de media, dispersograma interactivo (coloreado por nivel de estrés o horas de estudio) y tratamiento del caso atípico (ID=9, puntaje=22).
3. **Contraste de Hipótesis:** Evaluación formal de $H_0: \mu_{\text{alto}} \le \mu_{\text{bajo}}$ vs. $H_1: \mu_{\text{alto}} > \mu_{\text{bajo}}$ con pruebas $t$ de Welch y Mann-Whitney $U$, selector de percentil de corte de sueño y nivel de significancia $\alpha$.
4. **Regresión & Modelado:** Modelos de Regresión Lineal Simple, Regresión Múltiple y Regularización Ridge ($L_2$), ajuste de proporción de Test Split, cálculo en vivo de $R^2$, RMSE, MAE y gráfico real vs. predicho.
5. **Simulador & Diagnóstico:** Calculadora interactiva individual en tiempo real (ajuste de sueño, estudio y estrés) para predecir puntaje y riesgo académico con modelo de clasificación logística y curva sigmoide.
