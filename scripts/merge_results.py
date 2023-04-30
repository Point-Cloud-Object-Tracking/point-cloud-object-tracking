#!/usr/bin/env python3

import json
import os

classes = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck']
output_file = 'merged_output.json'

# Initialize an empty dictionary for the merged results
merged_results = {}
merged_meta = {}

# Loop through all classes
for class_name in classes:
    print(f"Merging class: {class_name}")
    file_path = os.path.join(class_name, 'results.json')

    # Load the content of the current JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Merge the "results" objects
    for key, value in data['results'].items():
        if key not in merged_results:
            merged_results[key] = []
        merged_results[key].extend(value)

    # Get the "meta" object from the first file
    if not merged_meta:
        merged_meta = data['meta']

# Create the output JSON object
output = {
    'meta': merged_meta,
    'results': merged_results
}

# Write the output JSON object to the output file
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

