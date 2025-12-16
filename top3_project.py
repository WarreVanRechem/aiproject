import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
from collections import deque, Counter
import time
from threading import Thread, Lock
from flask import Flask, Response, jsonify

# --- LCD BIBLIOTHEEK ---
from RPLCD.i2c import CharLCD
# -----------------------

# --- INSTELLINGEN ---
MODEL_PATH = 'nabirds_strict_quant_edgetpu.tflite'
LABEL_PATH = 'labels.txt'
THRESHOLD = 0.6
SMOOTHING_WINDOW = 10
FLASK_PORT = 5000
CAMERA_INDEX = 1
COUNT_COOLDOWN = 5.0  # ### NIEUW ### Aantal seconden wachten voordat we dezelfde vogel opnieuw tellen

# --- LCD INSTELLINGEN ---
I2C_ADDRESS = 0x27

# Globale variabele voor frames en slot
latest_frame = None
frame_lock = Lock()

# ### NIEUW ### Globale variabelen voor statistieken
bird_counts = Counter()  # Houdt de telling bij: {'Mus': 5, 'Merel': 2}
stats_lock = Lock()      # Zorgt dat Flask en de AI niet tegelijk de lijst aanpassen
last_detection_time = 0  # Tijdstip van laatste telling
last_detected_bird = None # Welke vogel als laatste geteld is

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
        lcd = None

def display_result_on_lcd(species_name, score):
    if lcd is None: return
    accuracy_percent = score * 100
    line1 = f"Vogel: {species_name}"[:16]
    line2 = f"Acc: {accuracy_percent:.1f}%"[:16]
    try:
        lcd.clear()
        lcd.write_string(line1)
        lcd.crlf()
        lcd.write_string(line2)
    except Exception as e:
        print(f"Fout bij LCD: {e}")

# --- HOOFDLOGICA VOOR INFERENCE ---
def inference_loop():
    global latest_frame, last_detection_time, last_detected_bird

    # Model laden
    try:
        interpreter = Interpreter(
            model_path=MODEL_PATH,
            experimental_delegates=[load_delegate('libedgetpu.so.1')]
        )
    except Exception as e:
        print(f"Fout bij laden model: {e}")
        return

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_type = input_details[0]['dtype']
    is_quantized_input = input_type == np.uint8
    output_type = output_details[0]['dtype']
    is_quantized_output = output_type == np.uint8
    scale, zero_point = output_details[0]['quantization']
    if scale == 0: scale = 1.0 / 255.0

    labels = load_labels(LABEL_PATH)
    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    print("Inference loop gestart...")
    last_lcd_update = time.time()
    LCD_UPDATE_INTERVAL = 1.0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        # 1. PREPROCESSING
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

        # 2. INFERENCE
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details[0]['index'])[0]

        if is_quantized_output:
            probabilities = (raw_output.astype(np.float32) - zero_point) * scale
        else:
            probabilities = raw_output

        prediction_buffer.append(probabilities)
        avg_prediction = np.mean(prediction_buffer, axis=0)

        class_id = np.argmax(avg_prediction)
        score = avg_prediction[class_id]
        prediction_name = labels[class_id]

        # 3. VISUALISATIE & TELLEN
        color = (0, 255, 0) if score > THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (start_x, start_y), (start_x+min_dim, start_y+min_dim), color, 2)

        if score > THRESHOLD:
            # --- LCD UPDATE ---
            if time.time() - last_lcd_update > LCD_UPDATE_INTERVAL:
                 display_result_on_lcd(prediction_name, score)
                 last_lcd_update = time.time()

            # ### NIEUW ### --- TELLEN MET COOLDOWN ---
            # We tellen alleen als:
            # A. Het een andere vogel is dan de vorige keer
            # OF
            # B. Het dezelfde vogel is, maar de cooldown tijd is voorbij (bv. vogel vloog weg en kwam terug)
            current_time = time.time()
            if (prediction_name != last_detected_bird) or (current_time - last_detection_time > COUNT_COOLDOWN):
                with stats_lock:
                    bird_counts[prediction_name] += 1
                    print(f"Nieuwe telling: {prediction_name} (Totaal: {bird_counts[prediction_name]})")
                last_detected_bird = prediction_name
                last_detection_time = current_time
            # ----------------------------------------
        else:
            text = f"Zoeken... ({score:.2f})"
            if time.time() - last_lcd_update > LCD_UPDATE_INTERVAL * 2:
                if lcd is not None:
                    try:
                        lcd.clear()
                        lcd.write_string("Zoeken naar...")
                    except: pass
                last_lcd_update = time.time()

        # FPS en overlay
        fps = 1.0 / (time.time() - start_time)
        cv2.rectangle(frame, (10, 10), (w-10, 60), (0,0,0), -1)
        cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        with frame_lock:
            latest_frame = frame_bytes

    cap.release()
    if lcd is not None: lcd.clear()

# --- FLASK STREAMING & API ---

def generate_frames():
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
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ### NIEUW ### API endpoint voor statistieken
@app.route('/stats')
def get_stats():
    """Geeft de top 3 meest gespotte vogels terug als JSON."""
    with stats_lock:
        # most_common(3) geeft een lijst: [('Mus', 10), ('Merel', 5), ...]
        top_birds = bird_counts.most_common(3)
    # Omzetten naar een mooi JSON formaat
    data = [{'name': bird, 'count': count} for bird, count in top_birds]
    return jsonify(data)

@app.route('/')
def index():
    # ### NIEUW ### Uitgebreide HTML met JavaScript om de lijst live te updaten
    return """
    <html>
    <head>
        <title>Vogelspotter AI</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f0f0; }
            .container { display: flex; flex-direction: column; align-items: center; margin-top: 20px; }
            .video-box { border: 5px solid #333; border-radius: 10px; }
            .stats-box {
                margin-top: 20px;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                width: 300px;
            }
            h2 { color: #2c3e50; }
            ul { list-style-type: none; padding: 0; }
            li {
                background: #ecf0f1;
                margin: 5px 0;
                padding: 10px;
                border-radius: 5px;
                display: flex;
                justify-content: space-between;
                font-weight: bold;
            }
            .count { color: #e74c3c; }
        </style>
    </head>
    <body>
        <h1>AI Vogelspotter Stream</h1>
        <div class="container">
            <img class="video-box" src="/video_feed" width="640" height="480">
            <div class="stats-box">
                <h2>Top 3 meest gespot</h2>
                <ul id="bird-list">
                    <li>Laden...</li>
                </ul>
            </div>
        </div>

        <script>
            // Functie om de stats op te halen van de Python server
            function updateStats() {
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        const list = document.getElementById('bird-list');
                        list.innerHTML = ''; // Maak lijst leeg
                        if (data.length === 0) {
                            list.innerHTML = '<li>Nog geen vogels gespot</li>';
                            return;
                        }

                        // Loop door de top 3 en maak HTML items
                        data.forEach((bird, index) => {
                            const li = document.createElement('li');
                            // Voeg medaille emoji toe voor 1, 2 en 3
                            let prefix = '';
                            if (index === 0) prefix = '🥇 ';
                            if (index === 1) prefix = '🥈 ';
                            if (index === 2) prefix = '🥉 ';
                            li.innerHTML = `<span>${prefix}${bird.name}</span> <span class="count">${bird.count}x</span>`;
                            list.appendChild(li);
                        });
                    })
                    .catch(error => console.error('Fout bij ophalen stats:', error));
            }

            // Update elke 2 seconden
            setInterval(updateStats, 2000);
            updateStats(); // Directe aanroep bij laden
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    initialize_lcd()
    t = Thread(target=inference_loop)
    t.daemon = True
    t.start()

    print(f"Stream is beschikbaar op http://<IP_ADRES_CORAL>:{FLASK_PORT}")
    try:
        app.run(host='0.0.0.0', port=FLASK_PORT, threaded=True)
    except KeyboardInterrupt:
        if lcd is not None:
            lcd.clear()
            lcd.write_string("Gestopt!")
