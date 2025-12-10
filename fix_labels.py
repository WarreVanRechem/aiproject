import os

DATA_DIR = 'coral_dataset/train'
OUTPUT_FILE = 'labels.txt'

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Kan map {DATA_DIR} niet vinden!")
        return

    # 1. Haal alle mapnamen op
    # Filter verborgen bestanden (die met een punt beginnen) eruit
    folder_names = [d for d in os.listdir(DATA_DIR)
                    if os.path.isdir(os.path.join(DATA_DIR, d))
                    and not d.startswith('.')]

    # 2. SORTEER ZE ALFABETISCH (Cruciaal!)
    sorted_names = sorted(folder_names)

    print(f"Gevonden klassen ({len(sorted_names)}):")
    for i, name in enumerate(sorted_names):
        print(f"  {i}: {name}")

    # 3. Schrijf naar labels.txt
    with open(OUTPUT_FILE, 'w') as f:
        for name in sorted_names:
            f.write(name + '\n')

    print(f"\nSucces! Nieuwe volgorde opgeslagen in '{OUTPUT_FILE}'.")
    print("Draai nu je debug_webcam.py opnieuw.")

if __name__ == "__main__":
    main()
