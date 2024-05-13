import os
import json
import argparse
import glob
import shutil
import numpy as np
import yaml

# Process files in the directory
def process_files(directory):
    files = [file for file in os.listdir(directory)]
    deleted_files_count = 0
    for file in files:
        if file.endswith('.json'):
            try:
                with open(os.path.join(directory, file), 'r') as f:
                    json.load(f)
            except json.JSONDecodeError:
                os.remove(os.path.join(directory, file))
                deleted_files_count += 1
        elif file.endswith('.jpg'):
            json_file = os.path.splitext(file)[0] + '.json'
            if json_file not in files:
                os.remove(os.path.join(directory, file))
                deleted_files_count += 1
    print(f"Deleted {deleted_files_count} invalid files.")

# Read JSON files in the directory
def read_json_files(directory):
    class_list = []
    json_files = glob.glob(os.path.join(directory, "*.json"))
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        shapes = data["shapes"]
        for shape in shapes:
            class_list.append(shape["label"])
    return class_list

# Convert JSON files to YOLO format
def json_to_yolo_seg(directory, class_list):
    json_files = [pos_json for pos_json in os.listdir(directory) if pos_json.endswith('.json')]
    for json_file in json_files:
        with open(os.path.join(directory, json_file)) as f:
            data = json.load(f)
        width = data["imageWidth"]
        height = data["imageHeight"]
        shapes = data["shapes"]
        text_file_name = os.path.join(directory, json_file.replace("json","txt"))
        with open(text_file_name, 'w') as text_file:
            for shape in shapes:
                class_id = class_list.index(shape["label"])
                points = shape["points"]
                normalize_point_list = [class_id]
                for point in points:
                    normalize_x = min(max(point[0]/width, 0.0), 1.0)
                    normalize_y = min(max(point[1]/height, 0.0), 1.0)
                    normalize_point_list.extend([normalize_x, normalize_y])
                text_file.write(' '.join(map(str, normalize_point_list)) + "\n")

# Delete JSON files in the directory
def delete_json_files(directory):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    for json_file in json_files:
        os.remove(json_file)

# Count the number of txt and json files in the directory
def count_files(directory):
    jpg_files = len(glob.glob(os.path.join(directory, "*.jpg")))
    txt_files = len(glob.glob(os.path.join(directory, "*.txt")))
    print(f'Number of image files {jpg_files}, text files {txt_files}')

# Split data into training and validation sets
def split_data(directory, train_ratio):

    train_dir = os.path.join(directory, 'train')
    val_dir = os.path.join(directory, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    jpg_files = sorted(glob.glob(os.path.join(directory, "*.jpg")))
    txt_files = sorted(glob.glob(os.path.join(directory, "*.txt")))
    paired_files = list(zip(jpg_files, txt_files))

    np.random.shuffle(paired_files)
    train_files = paired_files[:int(len(paired_files) * train_ratio)]
    val_files = paired_files[int(len(paired_files) * train_ratio):]

    for jpg_file, txt_file in train_files:
        shutil.move(jpg_file, os.path.join(train_dir, os.path.basename(jpg_file)))
        shutil.move(txt_file, os.path.join(train_dir, os.path.basename(txt_file)))
    for jpg_file, txt_file in val_files:
        shutil.move(jpg_file, os.path.join(val_dir, os.path.basename(jpg_file)))
        shutil.move(txt_file, os.path.join(val_dir, os.path.basename(txt_file)))

# Create a configuration file with labels
def create_config_with_labels(class_list):
    train_dir = '../train'
    val_dir = '../val'
    labels = list(set(class_list))

    config = {
        'train': train_dir,
        'val': val_dir,
        'nc': len(labels),
        'names': labels
    }

    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)

    print("Created config.yaml file.")

# Main function
def main(directory):
    process_files(directory)
    class_list = read_json_files(directory)
    json_to_yolo_seg(directory, class_list)
    delete_json_files(directory)
    count_files(directory)
    split_data(directory, train_ratio=0.8)
    create_config_with_labels(directory, class_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, required=True, help='Path to the directory')
    args = parser.parse_args()
    main(args.directory)
