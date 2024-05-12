# YOLO Data Preparation Tool

This repository contains a simple and efficient tool for preparing your data for training with YOLO. It is especially useful for beginners who are not familiar with the process of converting data from JSON to TXT format.

## Overview

The provided Python script performs several tasks:

1. **Processing Files**: It checks the validity of the JSON files in a given directory and removes any invalid ones. It also removes any JPG files that do not have a corresponding JSON file.

2. **Reading JSON Files**: It reads all the JSON files in the directory and extracts the class labels.

3. **Converting Labelme JSON to YOLO Format**: It converts the Labelme JSON files to YOLO format, which involves normalizing the bounding box coordinates and saving them in a TXT file.

4. **Deleting JSON Files**: After the conversion, it deletes the original JSON files to save space.

5. **Counting Files**: It counts the number of image and text files in the directory.

6. **Splitting Data**: It splits the data into training and validation sets based on a specified ratio.

7. **Creating Configuration File**: Finally, it creates a YAML configuration file with the class labels.

## Usage

To use this tool, simply run the main function with the directory containing your data as an argument:

```python
python main.py --directory /path/to/your/data
