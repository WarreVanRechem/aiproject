# test_camera_fast.py  ← nieuwe naam zodat je de oude houdt
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path

# Model laden
model = tf.keras.models.load_model("aiproject/res/cub200_mobilenetv2_alpha035.keras")
print("Model geladen ✓")

# Class names (mooi leesbaar)
data_dir = Path("CUB_200_2011/images")
class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
pretty_names = [name.split(".", 1)[1].replace("_", " ") if "." in name else name.replace("_", " ")
                for name in class_names]

# Webcam met betere instellingen (geen overbelichting + sneller)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# BELANGRIJK: automatische belichting en witbalans uitzetten!
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual mode, 0.75 = auto
cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # lagere waarde = donkerder (pas aan: -4 tot -10)
cap.set(cv2.CAP_PROP_GAIN, 0)              # minder ruis
cap.set(cv2.CAP_PROP_AUTO_WB, 0)           # witbalans vastzetten
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5000)

print("Camera gestart – belichting handmatig ingesteld – geen lag meer")
print("Druk 'q' om te stoppen, 's' voor screenshot\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Neem alleen het midden van het beeld (meest relevante deel + supersnel)
    h, w = frame.shape[:2]
    size = 300
    x = (w - size) // 2
    y = (h - size) // 2
    crop = frame[y:y+size, x:x+size]

    # Resize naar 224x224 + preprocess
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict (heel snel!)
    pred = model.predict(img_array, verbose=0)[0]
    idx = np.argmax(pred)
    conf = pred[idx] * 100
    name = pretty_names[idx]

    # Kader + tekst tekenen op origineel frame
    color = (0, 255, 0) if conf > 50 else (0, 255, 255)
    cv2.rectangle(frame, (x, y), (x+size, y+size), color, 4)
    label = f"{name} {conf:.1f}%"
    cv2.putText(frame, label, (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 3)

    # Top-3 onderaan
    top3 = np.argsort(pred)[-3:][::-1]
    for i, idx in enumerate(top3):
        txt = f"{i+1}. {pretty_names[idx]:35} {pred[idx]*100:5.1f}%"
        cv2.putText(frame, txt, (10, h - 70 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Vogelherkenning – snel & goed beeld – richt vogel in het midden", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fn = f"vogel_{name.replace(' ', '_')[:20]}_{conf:.0f}.jpg"
        cv2.imwrite(fn, frame)
        print(f"Foto opgeslagen → {fn}")

cap.release()
cv2.destroyAllWindows()