import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time

# --- INSTELLINGEN ---
MODEL_PATH = 'nabirds_strict_quant.tflite' # Check je bestandsnaam!
LABEL_PATH = 'labels.txt'
THRESHOLD = 0.6       # Pas tonen als gemiddelde zekerheid hoog is
SMOOTHING_WINDOW = 10   # Aantal frames om het gemiddelde van te nemen (5-15 is meestal goed)

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def main():
    print("Model laden...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check input type
    input_type = input_details[0]['dtype']
    is_quantized_input = input_type == np.uint8
    
    # Check output type (voor de-quantization)
    output_type = output_details[0]['dtype']
    is_quantized_output = output_type == np.uint8
    
    # Haal schaling parameters op als het model quantized is
    scale, zero_point = 0.0, 0
    if is_quantized_output:
        scale, zero_point = output_details[0]['quantization']
        if scale == 0: scale = 1.0 / 255.0

    labels = load_labels(LABEL_PATH)
    print(f"Labels geladen: {len(labels)}")

    # --- DE BUFFER (Hier slaan we de resultaten in op) ---
    # Een deque gooit automatisch de oudste weg als hij vol is
    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

    cap = cv2.VideoCapture(1)

    print(f"Start webcam... (Middelen over {SMOOTHING_WINDOW} frames)")
    print("Druk op 'q' om te stoppen.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        # 1. PREPROCESSING (Hetzelfde als altijd)
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
        else:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        # 2. INFERENCE (Denken)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details[0]['index'])[0]

        # 3. DE-QUANTIZATION (Naar procenten rekenen)
        # We zetten het direct om naar floats zodat we kunnen middelen
        if is_quantized_output:
            probabilities = (raw_output.astype(np.float32) - zero_point) * scale
        else:
            probabilities = raw_output

        # 4. TOEVOEGEN AAN BUFFER
        prediction_buffer.append(probabilities)

        # 5. GEMIDDELDE BEREKENEN
        # np.mean pakt het gemiddelde van alle lijsten in de buffer
        avg_prediction = np.mean(prediction_buffer, axis=0)

        # 6. RESULTAAT BEPALEN (Op basis van gemiddelde!)
        class_id = np.argmax(avg_prediction)
        score = avg_prediction[class_id]
        prediction_name = labels[class_id]

        # 7. VISUALISATIE
        # Teken vierkant
        color = (0, 255, 0) if score > THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (start_x, start_y), (start_x+min_dim, start_y+min_dim), color, 2)

        # Tekst balk
        # We tonen nu: "NAAM (GEMIDDELDE SCORE)"
        if score > THRESHOLD:
            text = f"{prediction_name}: {score:.2f}"
        else:
            text = f"Zoeken... ({score:.2f})"

        # Zwarte achtergrond voor tekst (zodat het leesbaar is)
        cv2.rectangle(frame, (10, 10), (300, 60), (0,0,0), -1)
        cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # FPS teller
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Vogel Herkenning (Smoothed)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
