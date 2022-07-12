#!/bin/bash
conda_env=$(echo ${CONDA_DEFAULT_ENV})

# Check that our current environment has the command-line parameter as a substring
if [[ "$conda_env" == *"$1"* ]]; then
    echo "Passed conda environment check. Continuing...";
else
    echo "Need to be in a $1 conda environment (or similar)";
    exit 1;
fi
