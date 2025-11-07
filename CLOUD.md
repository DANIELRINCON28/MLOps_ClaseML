# Proyecto: Pipeline MLOps Seguro para Detección de Fraude

Este documento describe la arquitectura tecnológica y la implementación de un pipeline CI/CD seguro, aplicando los estándares **GDPR** e **ISO 27001** para un modelo de Machine Learning enfocado en la detección de fraude financiero.

## 1. 🏗️ Arquitectura Tecnológica 

Para construir este sistema, utilizamos un stack tecnológico moderno enfocado en MLOps y la nube, donde cada pieza tiene una responsabilidad clara:



* **`GitHub`**: Es nuestro repositorio de código fuente. Sirve como la única fuente de verdad (Single Source of Truth) para todo el código de la aplicación y los scripts de entrenamiento del modelo.
* **`GitHub Actions`**: Es el orquestador de CI/CD. Actúa como el "cerebro" que automatiza cada paso del proceso, desde las pruebas hasta el despliegue, basándose en los flujos de trabajo definidos en `.github/workflows/`.
* **`SonarCloud`**: Es nuestro guardián de la calidad y seguridad del código. Se integra directamente con GitHub para ejecutar análisis estático de seguridad (SAST) y revisar vulnerabilidades, "code smells" y brechas de seguridad.
* **`Plataforma Cloud (Azure/AWS/GCP)`**: Es donde nuestro modelo vive y se ejecuta. GitHub Actions se encarga de desplegar el modelo entrenado como un *endpoint* (API) en la nube para que pueda hacer predicciones en tiempo real.
* **El Desafío (El "Por Qué")**: El modelo predice **fraude financiero**. Esto significa que, por definición, estamos tratando con **Información Personal Identificable (PII)** extremadamente sensible. No es un proyecto de "predecir flores"; es un sistema que maneja datos críticos.

---

## 2. 🛡️ Pipeline Seguro: GDPR e ISO 27001 en Acción (El "Cómo")

El verdadero reto no es solo desplegar un modelo, es hacerlo de forma segura y cumpliendo la ley. Así es como integramos GDPR e ISO 27001 en nuestro pipeline de GitHub Actions.

### El Problema: Datos Sensibles
En un pipeline de CI/CD, es fácil filtrar datos. Un `print()` accidental, un log de error, o un set de datos de prueba incorrecto pueden exponer PII y violar GDPR, resultando en multas millonarias.

### A. Cumpliendo con GDPR (Protección del Dato)

GDPR se centra en la **protección de los datos personales**. En nuestro pipeline, lo aplicamos así:

* **🔒 Segregación de Entornos:** El pipeline de GitHub Actions **NUNCA** toca datos de producción.
    * **Entrenamiento y Pruebas:** Los flujos de trabajo (`workflows`) solo tienen acceso a una base de datos de "testing" que contiene datos **anonimizados** o **pseudonimizados**.
    * **Producción:** El re-entrenamiento con datos reales solo ocurre en un entorno de producción seguro y aislado, fuera del alcance del pipeline de CI/CD de desarrollo.

* **📜 Minimización de Datos en Logs:**
    * Auditamos todos nuestros scripts (`pytest`, `train.py`) para asegurarnos de que **ningún log o `print()`** escriba información sensible (como IDs de usuario, números de cuenta, etc.) en la consola.
    * Los logs de GitHub Actions son públicos para el equipo, por lo que tratarlos como PII es fundamental.

### B. Implementando Controles ISO 27001 (Protección del Proceso)

ISO 27001 se centra en el **proceso** y los **controles** (un Sistema de Gestión de Seguridad de la Información).

* **🔑 Control A.9: Control de Acceso**
    * **GitHub Secrets:** Todas las credenciales (tokens de la nube, contraseñas de BD) se almacenan como `secrets` encriptados en GitHub. El código **nunca** contiene contraseñas.
    * **Protección de Ramas (`Branch Protection`):** La rama `main` está protegida. No se puede hacer `push` directo. Todo cambio debe pasar por un **Pull Request (PR)**.
    * **Entornos (`Environments`):** Usamos "Environments" de GitHub para `produccion`. Un despliegue a este entorno requiere una **aprobación manual** de un administrador del proyecto, creando un punto de control humano.

* **🕵️‍♂️ Control A.14: Seguridad en el Desarrollo (¡Aquí brilla SonarCloud!)**
    * **Análisis SAST Automatizado:** En cada PR, GitHub Actions ejecuta el análisis de SonarCloud.
    * **SonarCloud** revisa el código en busca de vulnerabilidades comunes (OWASP Top 10), como "secretos" hardcodeados, inyecciones, o librerías desactualizadas.
    * **El "Quality Gate" (La Barrera):** Configuramos un "Quality Gate" en SonarCloud. Si el código nuevo introduce una vulnerabilidad (ej. `CRITICAL` o `BLOCKER`), SonarCloud **falla la revisión y bloquea el PR**.
    * *Resultado:* Es técnicamente imposible fusionar código inseguro a `main`.

* **📜 Control A.12: Auditoría y Registros**
    * Cada ejecución de GitHub Actions es un **registro de auditoría inmutable**.
    * Podemos ver *quién* solicitó un despliegue, *quién* lo aprobó, *qué* commit exacto se desplegó, y si pasó todas las pruebas y los análisis de SonarCloud. Esto es crucial para la trazabilidad que exige ISO 27001.

---

## Resumen

Este pipeline no solo automatiza el MLOps (CI/CD), sino que implementa **DevSecOps** al integrar la seguridad como un paso fundamental e ineludible, usando SonarCloud como nuestro guardia de seguridad automatizado y los controles de GitHub para cumplir con las normativas GDPR e ISO 27001.