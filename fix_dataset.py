import os
import shutil

# --- CONFIGURATIE ---
TARGET_IDS = [991, 904, 821, 33, 686, 507, 108,159]
BASE_DIR = os.getcwd()
IMAGES_DIR = os.path.join(BASE_DIR, 'nabird/images')
DEST_DIR = os.path.join(BASE_DIR, 'coral_dataset')

FILE_IMAGES = 'nabird/images.txt'
FILE_LABELS = 'nabird/image_class_labels.txt'
FILE_CLASSES = 'nabird/classes.txt'
FILE_SPLIT = 'nabird/train_test_split.txt'
FILE_HIERARCHY = 'nabird/hierarchy.txt' # DEZE IS NIEUW EN CRUCIAAL

def load_mapping(filename, key_type, val_type):
    mapping = {}
    if not os.path.exists(filename):
        print(f"⚠️ Bestand niet gevonden: {filename}")
        return {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                mapping[key_type(parts[0])] = val_type(parts[1])
    return mapping

def get_all_children(parent_id, hierarchy):
    """Zoekt recursief alle sub-IDs die bij deze vogel horen."""
    children = [parent_id]
    # Zoek directe kinderen
    direct_kids = [child for child, parent in hierarchy.items() if parent == parent_id]

    for kid in direct_kids:
        children.extend(get_all_children(kid, hierarchy))

    return list(set(children)) # Verwijder dubbelen

def main():
    print("--- START SLIMME DATA REPARATIE (MET HIERARCHIE) ---")

    if os.path.exists(DEST_DIR):
        print("Oude map opschonen...")
        shutil.rmtree(DEST_DIR)

    # 1. Alles inladen
    print("Bestanden inlezen...")
    class_names = load_mapping(FILE_CLASSES, int, str)
    image_paths = load_mapping(FILE_IMAGES, str, str)
    image_labels = load_mapping(FILE_LABELS, str, int)
    train_split = load_mapping(FILE_SPLIT, str, int)

    # Hierarchie inladen (Child -> Parent)
    hierarchy_raw = load_mapping(FILE_HIERARCHY, int, int)

    if not class_names:
        print("❌ FOUT: classes.txt niet kunnen lezen. Stop.")
        return

    # 2. Verwerken
    total_copied = 0

    for target_id in TARGET_IDS:
        if target_id not in class_names:
            print(f"⚠️ ID {target_id} niet gevonden in classes.txt")
            continue

        # Naam ophalen
        bird_name = class_names[target_id].replace(' ', '_').replace('/', '-')

        # ALLE IDs verzamelen (De vogel zelf + al zijn kinderen/ondersoorten)
        related_ids = get_all_children(target_id, hierarchy_raw)

        print(f"\nVerwerken: {bird_name} (ID: {target_id})")
        if len(related_ids) > 1:
            print(f"   ↳ Inclusief {len(related_ids)-1} ondersoorten (IDs: {related_ids})")

        # Mappen maken
        os.makedirs(os.path.join(DEST_DIR, 'train', bird_name), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, 'val', bird_name), exist_ok=True)

        count = 0
        for img_id, cls_id in image_labels.items():
            # Check of deze foto bij een van de gerelateerde IDs hoort
            if cls_id in related_ids:
                if img_id in image_paths:
                    filename = image_paths[img_id]
                    src = os.path.join(IMAGES_DIR, filename)

                    if os.path.exists(src):
                        is_train = train_split.get(img_id, 1)
                        folder = 'train' if is_train == 1 else 'val'
                        dst = os.path.join(DEST_DIR, folder, bird_name, os.path.basename(filename))
                        shutil.copy2(src, dst)
                        count += 1
                        total_copied += 1

        if count == 0:
            print(f"   ❌ NOG STEEDS GEEN FOTO'S. Check of ID {target_id} wel klopt.")
        else:
            print(f"   ✅ {count} fotos gekopieerd!")

    print(f"\n--- KLAAR ---")
    print(f"Totaal: {total_copied} fotos.")
    print("Tip: Check nu nog eens hoeveel fotos je hebt. Je kunt nu veilig opnieuw trainen.")

if __name__ == "__main__":
    main()
