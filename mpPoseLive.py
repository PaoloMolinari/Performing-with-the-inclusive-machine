
import cv2 as cv
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pythonosc import udp_client

# OSC Configuration
OSC_IP = "127.0.0.1"
OSC_PORT = 57120
osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

model_path = '/home/pemb/python/mediapipe/pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    #print('pose landmarker result: {}'.format(result))
    if result and result.pose_landmarks:
        # Creamos una lista llamada landmarks-data
        landmarks_data = []
        # Para cada grupo de landmarks de cada mano
        for hand_idx, landmarkResult in enumerate(result.pose_landmarks):
            # Para cada landmark de cada grupo de landmarks
            for i, land in enumerate(landmarkResult):
                #print('pose landmarker result: {}'.format(landmarkResult))
                # populamos el array o lista con el indice y las coordenadas
                # x, y, z de cada landmark.
                # landmarks_data.extend([i, land.x, land.y, land.z])
                landmarks_data.extend([land.x, land.y, land.z])
            osc_client.send_message(f"/landmarks/{hand_idx}", landmarks_data)
            landmarks_data.clear()

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_poses=1,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

with PoseLandmarker.create_from_options(options) as landmarker:
 # bucle infinito 
    while True:
        # captura cuadro por cuadro
        ret, frame = cap.read()
        # Si el cuadro no es recivio correctamente
        # es decir ret no es true
        # enviamos un mensaje y salimos del bucle
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Si ret es true, es decir recibimos un cuadro
        # convertimos la imagen del cuadro 
        # a un tipo de imagen mediaPipe SRGB
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)

        # Ingresamos la imagen como parametro
        # al metodo recognize_async
        landmarker.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Mostramos la imagen de la camara
        cv.imshow('frame', frame)

        # Si pulsamos la tecla "q" salimos del bucle
        if cv.waitKey(1) == ord('q'):
            break
# Cerramos la captura
# y destrumimos todas las ventanas abiertas        
cap.release()
cv.destroyAllWindows()
