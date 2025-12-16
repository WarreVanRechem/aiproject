import os
import shutil

# --- CONFIGURATIE ---
# Jouw gekozen ID's
TARGET_IDS = [991, 904, 821, 33, 686, 507, 108,159]
# Paden (we gaan er vanuit dat dit script in de nabird map staat)
BASE_DIR = os.getcwd()
IMAGES_DIR = os.path.join(BASE_DIR, 'nabird/images')
DEST_DIR = os.path.join(BASE_DIR, 'coral_dataset')

# Bestandsnamen van NABirds
FILE_CLASSES = 'nabird/classes.txt'
FILE_IMAGES = 'nabird/images.txt'
FILE_LABELS = 'nabird/image_class_labels.txt'
FILE_SPLIT = 'nabird/train_test_split.txt'

def load_mapping(filename, key_type=str, val_type=str):
    """Leest een txt bestand en maakt een dictionary."""
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                k, v = parts
                mapping[key_type(k)] = val_type(v)
    return mapping

def main():
    print("--- Start Data Voorbereiding ---")

    # 1. Namen van vogels laden
    print("Klassen namen laden...")
    class_names = load_mapping(FILE_CLASSES, key_type=int, val_type=str)

    # Check of jouw ID's bestaan en print de namen
    target_names = {}
    for tid in TARGET_IDS:
        if tid in class_names:
            # Spaties vervangen door underscores voor mapnamen
            safe_name = class_names[tid].replace(' ', '_').replace('/', '-')
            target_names[tid] = safe_name
            print(f"ID {tid} gevonden: {safe_name}")
        else:
            print(f"WAARSCHUWING: ID {tid} niet gevonden in classes.txt!")

    # 2. Metadata laden
    print("Metadata laden (dit kan even duren)...")
    image_paths = load_mapping(FILE_IMAGES, key_type=str, val_type=str) # image_id -> path
    image_labels = load_mapping(FILE_LABELS, key_type=str, val_type=int) # image_id -> class_id
    train_split = load_mapping(FILE_SPLIT, key_type=str, val_type=int) # image_id -> is_train (1 or 0)

    # 3. Mappenstructuur aanmaken
    if os.path.exists(DEST_DIR):
        print(f"Oude map {DEST_DIR} verwijderen...")
        shutil.rmtree(DEST_DIR)

    for split in ['train', 'val']:
        for name in target_names.values():
            os.makedirs(os.path.join(DEST_DIR, split, name), exist_ok=True)

    # 4. Bestanden kopiëren
    print("Bestanden kopiëren naar nieuwe structuur...")
    count = 0

    for img_id, class_id in image_labels.items():
        # Check of deze foto bij onze 8 vogels hoort
        if class_id in TARGET_IDS:

            # Bepaal of het train of val is (NABirds gebruikt 1 voor train, 0 voor test)
            is_train = train_split.get(img_id, 0)
            split_dir = 'train' if is_train == 1 else 'val'

            bird_name = target_names[class_id]
            src_filename = image_paths[img_id]

            # Volledig pad construeren
            src_path = os.path.join(IMAGES_DIR, src_filename)
            dst_path = os.path.join(DEST_DIR, split_dir, bird_name, os.path.basename(src_filename))

            try:
                shutil.copy2(src_path, dst_path)
                count += 1
                if count % 100 == 0:
                    print(f"{count} fotos verwerkt...", end='\r')
            except FileNotFoundError:
                print(f"Kon bestand niet vinden: {src_path}")

    print(f"\nKlaar! {count} fotos gekopieerd naar '{DEST_DIR}'.")
    print("Je kunt nu verder met het trainen van het model.")

if __name__ == "__main__":
    main()
