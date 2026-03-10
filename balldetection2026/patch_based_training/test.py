import cv2
import json
import re
import numpy as np
import tensorflow as tf
from pathlib import Path

CAMERA = "BOTTOM"
MODEL_PATH = f"data/{CAMERA}/mein_ball_modell.keras"
PATCH_DIR = Path(f"data/{CAMERA}/patches")
INVENTORY_LIST = Path(f"data/{CAMERA}/val_images.json")


if not INVENTORY_LIST.exists():
    print("Fehler: Keine val_images.json gefunden")
    exit()

with open(INVENTORY_LIST, "r") as f:
    allowed_images = set(json.load(f))

def hole_original_namen(patch_name):
    base_stem = re.sub(r'_(no)?ball_\d+$', '', patch_name)
    return f"{base_stem}.png"

test_bilder = []
wahre_etiketten = []

kategorien = ["ball", "noball"]

for klassen_id, klassen_name in enumerate(kategorien):
    ordner_pfad = PATCH_DIR / klassen_name
    
    if not ordner_pfad.exists():
        continue
        
    for patch_pfad in ordner_pfad.glob("*.png"):
        original_bild = hole_original_namen(patch_pfad.stem)
        
        
        if original_bild in allowed_images:
            img = cv2.imread(str(patch_pfad))
            if img is not None:
                
                img = cv2.resize(img, (16, 16))
                test_bilder.append(img)
                wahre_etiketten.append(klassen_id)

if len(test_bilder) == 0:
    print("Keine passenden Patches gefunden. Breche ab.")
    exit()

X_test = np.array(test_bilder)
y_true = np.array(wahre_etiketten)

print("Lade Modell...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Der Test beginnt...\n")
vorhersagen = model.predict(X_test, batch_size=32, verbose=0)

y_pred = (vorhersagen > 0.5).astype(int).flatten()

echte_baelle = 0
baelle_erkannt = 0
baelle_verpasst = 0

echter_hintergrund = 0
hintergrund_erkannt = 0
falscher_alarm = 0

for wahr, vorhergesagt in zip(y_true, y_pred):
    if wahr == 0: 
        echte_baelle += 1
        if vorhergesagt == wahr:
            baelle_erkannt += 1
        else:
            baelle_verpasst += 1
            
    else: 
        echter_hintergrund += 1
        if vorhergesagt == wahr:
            hintergrund_erkannt += 1
        else:
            falscher_alarm += 1

print(f"Kategorie: BÄLLE (Gesamt: {echte_baelle})")
print(f" Richtig als Ball erkannt: {baelle_erkannt}")
print(f" Übersehen (als Rasen abgetan): {baelle_verpasst}")
if echte_baelle > 0:
    print(f" -> Erkennungsrate Bälle: {(baelle_erkannt/echte_baelle)*100:.1f}%\n")

print(f"Kategorie: HINTERGRUND (Gesamt: {echter_hintergrund})")
print(f" Richtig als Rasen erkannt: {hintergrund_erkannt}")
print(f" Falscher Alarm (für Ball gehalten): {falscher_alarm}")
if echter_hintergrund > 0:
    print(f" -> Erkennungsrate Hintergrund: {(hintergrund_erkannt/echter_hintergrund)*100:.1f}%")
print("="*45)