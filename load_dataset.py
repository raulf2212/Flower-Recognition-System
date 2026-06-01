import os
import zipfile

zip_file_name = 'dataset.zip'
base_dir = 'dataset/'

if os.path.exists(zip_file_name):
    if not os.path.exists(base_dir):
        print("Extracting necessary dataset files...")
        os.makedirs(base_dir, exist_ok=True)

        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if 'train/' in file or 'valid/' in file:
                    zip_ref.extract(file, './')
                elif 'cat_to_name.json' in file:
                    json_data = zip_ref.read(file)
                    target_json_path = os.path.join(base_dir, 'cat_to_name.json')
                    with open(target_json_path, 'wb') as f:
                        f.write(json_data)

        print("Extracting complete.")
    else:
        print("Dataset folder already exists. Skipping extraction.")
else:
    print(f"ERROR: Please ensure '{zip_file_name}' is placed in your main directory.")