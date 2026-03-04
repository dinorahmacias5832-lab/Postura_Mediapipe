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


CARPETA = r"C:\Users\Dinorah\OneDrive\Escritorio\Proyecto final\datos_csv"

def graficar_resultados():
    archivos = []
    
    print("Ingresa los 3 archivos a comparar (ej. Juan_A.csv)")
    
    for i in range(3):
        archivo = input(f"Nombre del archivo {i+1}: ")
        path = os.path.join(CARPETA, archivo)
        
        if not os.path.exists(path):
            print(f"❌ {archivo} no encontrado en la carpeta.")
            return
        
        archivos.append((archivo, path))
    
    plt.figure(figsize=(12,7))
    colores_linea = ["blue", "black", "magenta"]
    
    # =========================
    # GRAFICAR LÍNEAS
    # =========================
    for i, (archivo, path) in enumerate(archivos):
        df = pd.read_csv(path)
        
        nombre = archivo.upper()
        
        if "_A" in nombre:
            actividad = "Actividad A - Lectura"
        elif "_B" in nombre:
            actividad = "Actividad B - Escritura"
        elif "_C" in nombre:
            actividad = "Actividad C - Uso libre"
        else:
            actividad = archivo
        
        plt.plot(
            df["Tiempo"], 
            df["Ángulo"],
            linewidth=2.5,
            color=colores_linea[i],
            label=actividad,
            zorder=3   # Las líneas quedan encima
        )
    
    # =========================
    # ZONAS POSTURALES (FONDO)
    # =========================
    plt.axhspan(170,180,facecolor='green',alpha=0.12,label="Perfecta", zorder=1)
    plt.axhspan(160,170,facecolor='lightgreen',alpha=0.12,label="Buena", zorder=1)
    plt.axhspan(150,160,facecolor='yellow',alpha=0.12,label="Regular", zorder=1)
    plt.axhspan(140,150,facecolor='orange',alpha=0.12,label="Mala", zorder=1)
    plt.axhspan(130,140,facecolor='red',alpha=0.12,label="Pésima", zorder=1)
    plt.axhspan(0,130,facecolor='purple',alpha=0.12,label="Postura crítica", zorder=1)
    
    # =========================
    # CONFIGURACIÓN FINAL
    # =========================
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Ángulo (°)")
    plt.title("Comparación Postural entre Actividades A, B y C")
    
    
    plt.ylim(0,185)
    plt.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.show()


# ============================================================
# -------------------- OPCIÓN 3 ------------------------------
# ANALISIS ESTADISTICO INDIVIDUAL (FRIEDMAN POR PERSONA)
# ============================================================

def friedman_por_persona():

    from scipy.stats import friedmanchisquare
    import numpy as np
    import pandas as pd
    import os

    MAPA = {"a": "Actividad_A",
            "b": "Actividad_B",
            "c": "Actividad_C"}

    # -------- DETECTAR PERSONAS DISPONIBLES --------
    personas = set()

    for archivo in os.listdir(CARPETA):
        if archivo.endswith(".csv"):
            try:
                persona, _ = archivo.replace(".csv", "").split("_")
                personas.add(persona)
            except:
                continue

    personas = sorted(list(personas))

    if not personas:
        print("No se encontraron participantes.")
        return

    print("\nParticipantes disponibles:")
    for i, p in enumerate(personas):
        print(f"{i+1}. {p}")

    opcion = input("\nSelecciona el número de la persona: ")

    try:
        persona_objetivo = personas[int(opcion) - 1]
    except:
        print("Selección inválida.")
        return

    datos = {}

    # -------- CARGAR SUS 3 ACTIVIDADES --------
    for archivo in os.listdir(CARPETA):

        if archivo.startswith(persona_objetivo + "_") and archivo.endswith(".csv"):

            actividad = archivo.replace(".csv", "").split("_")[1].lower()

            if actividad in MAPA:

                df = pd.read_csv(os.path.join(CARPETA, archivo))

                # Buscar columna ángulo
                col = None
                for c in df.columns:
                    if c.lower() in ["angulo", "ángulo"]:
                        col = c
                        break

                if col is None:
                    continue

                serie = df[col].dropna().reset_index(drop=True)

                # -------- RECORTE PROPORCIONAL --------
                prop = 25 / 300
                n = len(serie)
                corte = int(n * prop)

                if n < 50:
                    print("Muy pocos datos para analizar.")
                    return

                serie = serie.iloc[corte: n - corte]

                datos[MAPA[actividad]] = serie

    if len(datos) != 3:
        print("No se encontraron las 3 actividades completas.")
        return

    # -------- DIVIDIR EN 10 BLOQUES --------
    bloques = 10
    listas = {}

    for act, serie in datos.items():

        tamaño_bloque = len(serie) // bloques

        medias = []

        for i in range(bloques):
            inicio = i * tamaño_bloque
            fin = inicio + tamaño_bloque
            bloque = serie.iloc[inicio:fin]
            medias.append(bloque.mean())

        listas[act] = medias

       # -------- FRIEDMAN --------
    resultado = friedmanchisquare(
        listas["Actividad_A"],
        listas["Actividad_B"],
        listas["Actividad_C"]
    )

    print("\n======================================")
    print(f"FRIEDMAN INDIVIDUAL - {persona_objetivo}")
    print("======================================")
    print(f"X² = {resultado.statistic:.3f}")
    print(f"p  = {resultado.pvalue:.5f}")

    if resultado.pvalue < 0.05:

        print("Resultado: Diferencias significativas")
        print("\n--- Post-hoc Wilcoxon (Bonferroni α=0.0167) ---")

        from scipy.stats import wilcoxon
        import numpy as np

        comparaciones = [
            ("A vs B", listas["Actividad_A"], listas["Actividad_B"]),
            ("A vs C", listas["Actividad_A"], listas["Actividad_C"]),
            ("B vs C", listas["Actividad_B"], listas["Actividad_C"])
        ]

        for nombre, g1, g2 in comparaciones:

            stat, p = wilcoxon(g1, g2)

            n = len(g1)

        # cálculo correcto de Z
            mean_W = n * (n + 1) / 4
            sd_W = np.sqrt(n * (n + 1) * (2*n + 1) / 24)
            z = (stat - mean_W) / sd_W

            r = abs(z) / np.sqrt(n)

            if p < 0.0167:
                resultado_texto = "Significativo"
            else:
                resultado_texto = "No significativo"

            print(f"{nombre}: p = {p:.5f} → {resultado_texto}")
            print(f"    Tamaño del efecto r = {r:.3f}")
 
    else:
        print("Resultado: No significativo")


# ============================================================
# -------------------- OPCIÓN 4 ------------------------------
# ANALISIS ESTADISTICO INDIVIDUAL (FRIEDMAN POR PERSONA)
# ============================================================

def analisis_estadistico():
    import os
    import pandas as pd
    
    print("\n--- ANÁLISIS ESTADÍSTICO INDIVIDUAL ---\n")
    
    archivos = [f for f in os.listdir(CARPETA) if f.endswith(".csv")]
    
    if not archivos:
        print("No hay archivos disponibles.")
        return
    
    resultados = []
    
    for archivo in archivos:
        ruta = os.path.join(CARPETA, archivo)
        df = pd.read_csv(ruta)
        
        nombre, actividad = archivo.replace(".csv", "").split("_")
        
        desc = df["Ángulo"].describe()
        
        media = desc["mean"]
        mediana = desc["50%"]
        std = desc["std"]
        
        # Mala postura = Ángulo < 150°
        mala = df[df["Ángulo"] < 150]
        
        tiempo_mala = 0
        if not mala.empty:
            tiempo_mala = mala["Tiempo"].diff().fillna(0).sum()
        
        resultados.append([
            nombre,
            actividad,
            round(media,2),
            round(mediana,2),
            round(std,2),
            round(tiempo_mala,2)
        ])
    
    tabla = pd.DataFrame(resultados,
                         columns=["Sujeto",
                                  "Actividad",
                                  "Media (°)",
                                  "Mediana (°)",
                                  "Std (°)",
                                  "Tiempo mala postura (s)"])
    
    print(tabla)

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
        print("3) Test Friedman")
        print("4) Estadistica descriptiva")
        print("5) Salir")

        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            captura_postura()
        elif opcion == "2":
            graficar_resultados()
        elif opcion == "3":
            friedman_por_persona()
        elif opcion == "4":
            analisis_estadistico()    
        elif opcion == "5":
            break
        else:
            print("Opción inválida.")


if __name__ == "__main__":
    menu()