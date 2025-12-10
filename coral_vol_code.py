import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
from collections import deque
import time
from threading import Thread, Lock
from flask import Flask, Response

# --- LCD BIBLIOTHEEK ---
from RPLCD.i2c import CharLCD
# -----------------------

# --- INSTELLINGEN ---
MODEL_PATH = 'nabirds_strict_quant_edgetpu.tflite'
LABEL_PATH = 'labels.txt'
THRESHOLD = 0.6
SMOOTHING_WINDOW = 10
FLASK_PORT = 5000 # Poort waarop de stream wordt gehost
CAMERA_INDEX = 1 # Pas aan indien nodig (0 of 1)

# --- LCD INSTELLINGEN ---
I2C_ADDRESS = 0x27  # Controleer of dit het juiste adres is voor jouw LCD
# -----------------------

# Globale variabele voor het meest recente JPEG-frame en een slot
latest_frame = None
frame_lock = Lock()

# Globale LCD-instantie
lcd = None

app = Flask(__name__)

# --- MODEL & UTILITY FUNCTIES ---
def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def initialize_lcd():
    """Initialiseert het LCD-scherm."""
    global lcd
    try:
        lcd = CharLCD(
            i2c_expander='PCF8574',
            address=I2C_ADDRESS,
            port=1,
            cols=16,
            rows=2,
            dotsize=8
        )
        print("=== LCD Succesvol Geïnitialiseerd ===")
        lcd.write_string("Initialisatie...")
    except Exception as e:
        print(f"Fout bij LCD-initialisatie (Controleer I2C): {e}")
        # Stel lcd in op None zodat de code niet crasht als het niet werkt
        lcd = None

def display_result_on_lcd(species_name, score):
    """
    Toont de AI voorspelling op het LCD.
    """
    if lcd is None:
        return # Doe niets als LCD niet beschikbaar is

    # Formatteer de score als percentage
    accuracy_percent = score * 100

    # Verkort de soortnaam indien nodig (16 karakters is de limiet)
    line1 = f"Vogel: {species_name}"
    line2 = f"Acc: {accuracy_percent:.1f}%"
    # Zorg dat de lijnen 16 karakters niet overschrijden
    line1 = line1[:16]
    line2 = line2[:16]

    try:
        lcd.clear()
        lcd.write_string(line1)
        lcd.crlf() # Nieuwe regel
        lcd.write_string(line2)
        # Debug-output in terminal
        print(f"[LCD Update] {line1} | {line2}")
    except Exception as e:
        print(f"Fout bij schrijven naar LCD: {e}")
        # Zet LCD op None om verdere pogingen te vermijden als het defect is
        # lcd = None

# --- HOOFDLOGICA VOOR INFERENCE EN FRAME GENERATIE ---
def inference_loop():
    global latest_frame

    # Model Setup (Zelfde als uw originele main functie)
    try:
        interpreter = Interpreter(
            model_path=MODEL_PATH,
            experimental_delegates=[load_delegate('libedgetpu.so.1')]
        )
    except Exception as e:
        print(f"Fout bij laden model/Edge TPU: {e}")
        # Update LCD dat er een fout is
        if lcd is not None:
            lcd.clear()
            lcd.write_string("MODEL FOUT!")
        return

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_type = input_details[0]['dtype']
    is_quantized_input = input_type == np.uint8
    output_type = output_details[0]['dtype']
    is_quantized_output = output_type == np.uint8
    scale, zero_point = 0.0, 0
    if is_quantized_output:
        scale, zero_point = output_details[0]['quantization']
        if scale == 0: scale = 1.0 / 255.0

    labels = load_labels(LABEL_PATH)
    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    print("Inference loop gestart...")
    if lcd is not None:
        lcd.clear()
        lcd.write_string("Zoeken naar")
        lcd.crlf()
        lcd.write_string("Vogels...")
    last_lcd_update = time.time()
    LCD_UPDATE_INTERVAL = 1.0 # Update het LCD maximaal 1 keer per seconde

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            if lcd is not None:
                lcd.clear()
                lcd.write_string("Camerafout!")
            break

        # 1. PREPROCESSING (Zelfde als in uw code)
        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        crop_img = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        rgb_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (224, 224))
        input_data = np.expand_dims(resized_img, axis=0)

        if is_quantized_input:
            input_data = input_data.astype(np.uint8)
        # else: ... (float path weggelaten omdat Edge TPU uint8 vereist)

        # 2. INFERENCE & 3. DE-QUANTIZATION
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details[0]['index'])[0]

        if is_quantized_output:
            probabilities = (raw_output.astype(np.float32) - zero_point) * scale
        else:
            probabilities = raw_output
        # 4. & 5. BUFFER & GEMIDDELDE
        prediction_buffer.append(probabilities)
        avg_prediction = np.mean(prediction_buffer, axis=0)

        # 6. RESULTAAT BEPALEN
        class_id = np.argmax(avg_prediction)
        score = avg_prediction[class_id]
        prediction_name = labels[class_id]

        # 7. VISUALISATIE (Tekenen op het frame)
        color = (0, 255, 0) if score > THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (start_x, start_y), (start_x+min_dim, start_y+min_dim), color, 2)

        if score > THRESHOLD:
            text = f"{prediction_name}: {score:.2f}"
            # --- UPDATE LCD HIER ---
            if time.time() - last_lcd_update > LCD_UPDATE_INTERVAL:
                 display_result_on_lcd(prediction_name, score)
                 last_lcd_update = time.time()
            # ------------------------
        else:
            text = f"Zoeken... ({score:.2f})"
            # --- Update LCD om 'Zoeken...' weer te geven als de drempel niet wordt bereikt
            if time.time() - last_lcd_update > LCD_UPDATE_INTERVAL * 2: # Minder vaak updaten als er niets is
                if lcd is not None:
                    try:
                        lcd.clear()
                        lcd.write_string("Zoeken naar")
                        lcd.crlf()
                        lcd.write_string("Vogels...")
                    except: # Negeer fouten als LCD is losgekoppeld
                        pass
                last_lcd_update = time.time()
            # -------------------------------------------------------------


        h_frame, w_frame, _ = frame.shape
        fps = 1.0 / (time.time() - start_time)

        # Tekst en FPS weergave
        cv2.rectangle(frame, (10, 10), (w_frame - 10, 60), (0,0,0), -1)
        cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w_frame-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- CONVERSIE NAAR JPEG VOOR WEBSERVER ---
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Update de globale frame-variabele
        with frame_lock:
            latest_frame = frame_bytes

    cap.release()
    # Wis LCD bij beëindiging
    if lcd is not None:
        lcd.clear()
        lcd.write_string("KLAAR!")

# --- FLASK STREAMING FUNCTIES (ongewijzigd) ---

def generate_frames():
    """Generatort voor het MJPEG stream formaat."""
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.1)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Het HTTP-endpoint voor de stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Simpele HTML-pagina om de stream in te bedden."""
    return (
        "<html><body>"
        "<h1>Vogelherkenning Stream</h1>"
        f'<img src="/video_feed" width="640" height="480">'
        "</body></html>"
    )

if __name__ == '__main__':
    # Initialiseer eerst het LCD
    initialize_lcd()

    # Start de inference- en visualisatie-lus in een aparte thread
    t = Thread(target=inference_loop)
    t.daemon = True # Zorgt ervoor dat de thread stopt wanneer Flask stopt
    t.start()

    # Start de Flask webserver
    print(f"Stream is beschikbaar op http://<IP_ADRES_CORAL>:{FLASK_PORT}")
    try:
        app.run(host='0.0.0.0', port=FLASK_PORT, threaded=True)
    except KeyboardInterrupt:
        # Wis LCD als de Flask-server wordt gestopt via Ctrl+C
        print("\nProgramma gestopt door gebruiker.")
        if lcd is not None:
            lcd.clear()
            lcd.write_string("Gestopt!")
