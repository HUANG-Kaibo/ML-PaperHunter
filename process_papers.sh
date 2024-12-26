#!/bin/bash

# Check if input file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_jsonl_file>"
    echo "Example: $0 data/iclr2024.jsonl"
    exit 1
fi

input_file="$1"

# Extract base name without extension
base_name="${input_file%.*}"

# Define output files
relevant_file="${base_name}_relevant.jsonl"
translated_file="${base_name}_translated.jsonl"

# Process papers to find relevant ones
echo "Finding relevant papers..."
python get_relatex_paper.py --input "$input_file" --output "$relevant_file"

# Translate relevant papers
echo "Translating relevant papers..."
python translate_papers.py --input "$relevant_file" --output "$translated_file"

echo "Processing complete!"
echo "Relevant papers saved to: $relevant_file"
echo "Translated papers saved to: $translated_file" 
