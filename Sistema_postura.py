import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CARPETA = "datos_csv"
os.makedirs(CARPETA, exist_ok=True)

# ============================================================
# -------------------- OPCIÓN 1 ------------------------------
# CAPTURA DE POSTURA
# ============================================================

def captura_postura():
    import cv2
    import mediapipe as mp
    import math
    import time

    nombre = input("Nombre del participante: ")
    actividad = input("Actividad (a, b o c): ").lower()

    archivo_csv = f"{CARPETA}/{nombre}_{actividad}.csv"

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    print("Presiona 'q' para detener.\n")

    tiempo_inicial = time.time()
    datos = []

    def calcular_angulo(a, b, c):
        angulo = math.degrees(
            math.atan2(c[1]-b[1], c[0]-b[0]) -
            math.atan2(a[1]-b[1], a[0]-b[0])
        )
        angulo = abs(angulo)
        if angulo > 180:
            angulo = 360 - angulo
        return angulo

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            h, w, _ = frame.shape

            cadera = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            hombro = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            oreja = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]

            c = (int(cadera.x*w), int(cadera.y*h))
            hmb = (int(hombro.x*w), int(hombro.y*h))
            o = (int(oreja.x*w), int(oreja.y*h))

            angulo = calcular_angulo(c, hmb, o)

            tiempo_rel = round(time.time()-tiempo_inicial, 2)
            datos.append([tiempo_rel, angulo])

            # Clasificación visual
            if angulo >= 170:
                estado = "Postura perfecta"
                color = (0,255,0)
            elif angulo >= 160:
                estado = "Buena postura"
                color = (50,205,50)
            elif angulo >= 150:
                estado = "Postura regular"
                color = (0,255,255)
            elif angulo >= 140:
                estado = "Mala postura"
                color = (0,165,255)
            elif angulo >= 130:
                estado = "Postura pésima"
                color = (0,0,255)
            else:
                estado = "Postura crítica"
                color = (128,0,128)

            mp_drawing.draw_landmarks(frame, result.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame, f"Angulo: {int(angulo)}°",
                        (30,40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

            cv2.putText(frame, estado,
                        (30,80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        cv2.imshow("Captura Postura", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(datos, columns=["Tiempo", "Ángulo"])
    df.to_csv(archivo_csv, index=False)
    print(f"\nDatos guardados en: {archivo_csv}")


# ============================================================
# -------------------- OPCIÓN 2 ------------------------------
# GRAFICAR RESULTADOS
# ============================================================

def graficar_resultados():
    archivo = input("Nombre del archivo (ej. Juan_a.csv): ")
    path = f"{CARPETA}/{archivo}"

    if not os.path.exists(path):
        print("Archivo no encontrado.")
        return

    df = pd.read_csv(path)

    plt.figure(figsize=(10,6))
    plt.plot(df["Tiempo"], df["Ángulo"], linewidth=2)

    plt.axhspan(170,180,alpha=0.2,label="Perfecta")
    plt.axhspan(160,170,alpha=0.2,label="Buena")
    plt.axhspan(150,160,alpha=0.2,label="Regular")
    plt.axhspan(140,150,alpha=0.2,label="Mala")
    plt.axhspan(130,140,alpha=0.2,label="Pésima")
    plt.axhspan(0,130,alpha=0.2,label="Postura crítica")

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Ángulo (°)")
    plt.title("Evolución Postural")
    plt.legend()
    plt.ylim(0,185)
    plt.show()


# ============================================================
# -------------------- OPCIÓN 3 ------------------------------
# ANÁLISIS ESTADÍSTICO INDIVIDUAL
# ============================================================

def analisis_estadistico():

    from scipy.stats import f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import itertools

    MAPA_ACTIVIDADES = {"a": "Actividad_A",
                        "b": "Actividad_B",
                        "c": "Actividad_C"}

    def leer_angulos(path):
        df = pd.read_csv(path)
        for col in df.columns:
            if col.lower() in ['angulo', 'ángulo']:
                return df[col].dropna()
        raise ValueError(f"No se encontró columna de ángulo en {path}")

    def cohen_d(grupo1, grupo2):
        n1, n2 = len(grupo1), len(grupo2)
        s1, s2 = np.std(grupo1, ddof=1), np.std(grupo2, ddof=1)
        sd_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        return (np.mean(grupo1) - np.mean(grupo2)) / sd_pooled

    participantes = {}

    for archivo in os.listdir(CARPETA):
        if archivo.endswith(".csv"):
            nombre = archivo.replace(".csv", "")
            try:
                persona, actividad = nombre.split("_")
            except:
                continue

            try:
                angulos = leer_angulos(os.path.join(CARPETA, archivo))
                if persona not in participantes:
                    participantes[persona] = {}
                participantes[persona][MAPA_ACTIVIDADES.get(actividad, actividad)] = angulos
            except:
                continue

    print("\nPersonas disponibles:")
    for p in participantes.keys():
        print(f" - {p}")

    persona_elegida = input("\nEscribe el nombre EXACTO de la persona: ")

    if persona_elegida not in participantes:
        print("Persona no encontrada.")
        return

    actividades = participantes[persona_elegida]

    if len(actividades) < 2:
        print("Menos de 2 actividades disponibles.")
        return

    datos = [v for v in actividades.values()]
    resultado = f_oneway(*datos)

    print(f"\nANOVA: F = {resultado.statistic:.3f}, p = {resultado.pvalue:.5f}")
    print("Significativo" if resultado.pvalue < 0.05 else "No significativo")

    print("\n--- d de Cohen ---")

    pares = itertools.combinations(actividades.items(), 2)

    for (act1, g1), (act2, g2) in pares:
        d = cohen_d(g1, g2)
        interpretacion = (
            "pequeño" if abs(d) < 0.5 else
            "mediano" if abs(d) < 0.8 else
            "grande"
        )
        print(f"{act1} vs {act2}: d = {d:.3f} ({interpretacion})")

    if resultado.pvalue < 0.05:
        print("\n--- Tukey HSD ---")

        df_tukey = pd.DataFrame({
            "Ángulo": sum([list(v) for v in actividades.values()], []),
            "Actividad": sum([[k]*len(v) for k,v in actividades.items()], [])
        })

        tukey = pairwise_tukeyhsd(
            endog=df_tukey["Ángulo"],
            groups=df_tukey["Actividad"],
            alpha=0.05
        )

        print(tukey)

    print("\nAnálisis completado.")


# ============================================================
# -------------------- MENÚ PRINCIPAL ------------------------
# ============================================================

def menu():
    while True:
        print("\n==============================")
        print("   SISTEMA DE POSTURA")
        print("==============================")
        print("1) Capturar postura")
        print("2) Graficar resultados")
        print("3) Análisis estadístico individual")
        print("4) Salir")

        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            captura_postura()
        elif opcion == "2":
            graficar_resultados()
        elif opcion == "3":
            analisis_estadistico()
        elif opcion == "4":
            break
        else:
            print("Opción inválida.")


if __name__ == "__main__":
    menu()