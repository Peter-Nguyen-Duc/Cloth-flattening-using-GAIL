#!/bin/bash

# Set your line numbers here (space-separated)
line_numbers=(29 36 43 50 57 64 71 78 85)

xml_file="cloth_sim_test.xml"  # Change this to your XML file

# Sort line numbers in descending order to avoid shifting issues
IFS=$'\n' sorted=($(sort -nr <<<"${line_numbers[*]}"))
unset IFS

# Create a temporary file
temp_file=$(mktemp)

# Initialize counter for joint names
counter=9

# Process each line number
for line in "${sorted[@]}"; do
    # Check if line number is valid
    if ! [[ "$line" =~ ^[0-9]+$ ]]; then
        echo "Invalid line number: $line"
        continue
    fi

    # Remove the line and the next two lines, then insert the new joint element
    sed -i "${line}s/.*/      <joint name=\"c1_free_joint_${counter}\" type=\"free\"\/>/" "$xml_file"
    sed -i "$((line+1))d" "$xml_file"
    sed -i "$((line+1))d" "$xml_file"


    ((counter--))
done

echo "XML file updated."