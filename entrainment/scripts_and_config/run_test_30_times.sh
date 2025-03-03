#!/bin/bash

# Pass test.py to variable
SCRIPT_DIR="./Asist3_data_management"
PYTHON_SCRIPT="asist3_test.py"
# Output file
OUTPUT_FILE="accuracy_multicat_no-addressee_30_runs.txt"

# Ensure the output file is empty
> "$OUTPUT_FILE"

# Change to the script directory
cd "$SCRIPT_DIR" || { echo "Failed to enter directory $SCRIPT_DIR"; exit 1; }

# Parse additional arguments

# Run the script 30 times
for i in {1..30}; do
    # Run the script with optional arguments and capture output and error
    OUTPUT=$(python "$PYTHON_SCRIPT" "$@" 2>&1)
    EXIT_STATUS=$?
    
    # Check if the Python script encountered an error
    if [ $EXIT_STATUS -ne 0 ]; then
        echo "Error encountered in run $i: $OUTPUT"
        echo "Terminating script."
        exit 1
    fi
    
    # Capture only the last line of output
    LAST_LINE=$(echo "$OUTPUT" | tail -n 1)

    # Append the last line to the output file
    echo "$LAST_LINE" >> "./$OUTPUT_FILE"
    echo "Run $i completed."
done

# Return to the original directory
cd - > /dev/null
echo "All runs completed. Results saved in $OUTPUT_FILE"
