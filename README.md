# Sistema de Estimación Postural con MediaPipe
Sistema de estimación postural cervical en tiempo real con análisis estadístico utilizando Python y MediaPipe.

## Descripción

Este proyecto implementa un sistema de estimación postural utilizando visión por computadora y la librería MediaPipe. 
El programa calcula el ángulo corporal en tiempo real a partir de puntos anatómicos detectados por la cámara, 
clasificando la postura en diferentes categorías (perfecta, buena, regular, mala, pesima y critica). 
Además, permite almacenar los datos y generar gráficas para analizar la evolución postural.

## Objetivo

Desarrollar un sistema capaz de evaluar la postura corporal mediante el cálculo de ángulos articulares, 
permitiendo su clasificación y análisis estadístico.

## Requisitos

- Python 3.10.0 (recomendado)
- Cámara web
- Librerías incluidas en requirements.txt
  
Nota:
Se recomienda utilizar Python 3.10.0 debido a posibles incompatibilidades 
de MediaPipe con versiones superiores.

## Instalación

1. Instalar Python 3.10.0.
2. Abrir el Símbolo del sistema (CMD).
3. Crear un entorno virtual:

   python -m venv venv

4. Activar el entorno virtual:

   venv\Scripts\activate

5. Instalar las dependencias:

   pip install -r requirements.txt

## Uso

Ejecutar el archivo principal desde la terminal:

python sistema_postura.py

Al iniciar el programa se desplegará un menú con las siguientes opciones:

1. Capturar postura: activa la cámara y calcula el ángulo en tiempo real.
2. Graficar resultados: permite visualizar la evolución del ángulo almacenado en un archivo CSV.
3. Análisis estadístico individual: realiza ANOVA, cálculo de d de Cohen y prueba de Tukey para las actividades registradas.

Seleccionar la opción deseada ingresando el número correspondiente.
