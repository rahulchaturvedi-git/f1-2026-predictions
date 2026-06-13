#!/bin/bash

# This script creates aliases for the prediction pipeline.
# Run it once in your terminal: source setup_shortcuts.sh

gps=("australia" "china" "miami" "japan" "canada" "monaco")

for gp in "${gps[@]}"; do
    alias_name="predict_${gp}_gp"
    script_path="$(pwd)/src/predict_pipeline.py"
    
    echo "Creating alias: $alias_name"
    alias "$alias_name"="python3 $script_path $gp"
done

echo ""
echo "✅ Shortcuts created! You can now use:"
for gp in "${gps[@]}"; do
    echo "  predict_${gp}_gp"
done
