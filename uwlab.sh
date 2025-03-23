#!/usr/bin/env bash
# Exit immediately if a command fails
set -e

# Define a function to print help
print_help() {
    echo "Usage: $(basename "$0") [option]"
    echo -e "\t-h, --help          Display this help message."
    echo -e "\t-i, --install [LIB]  Install the uwlab packages. Optionally specify a LIB."
    echo -e "\t-f, --format         Format the uwlab code."
    echo -e "\t-t, --test           Run tests for uwlab."
}

# If no arguments are provided, show the help and exit.
if [ $# -eq 0 ]; then
    print_help
    exit 0
fi

# Base path to the packages (adjust as needed)
BASE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/source"

# Process the command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            print_help
            exit 0
            ;;
        -i|--install)
            echo "[INFO] Installing uwlab packages..."
            echo "Installing uwlab..."
            pip install -e "${BASE_PATH}/uwlab"
            echo "Installing uwlab_assets..."
            pip install -e "${BASE_PATH}/uwlab_assets"
            echo "Installing uwlab_tasks..."
            pip install -e "${BASE_PATH}/uwlab_tasks"
            pip install -e "${BASE_PATH}/uwlab_rl"
            echo "[INFO] All packages have been installed in editable mode."
            ;;
        -f|--format)
            echo "[INFO] Formatting uwlab code..."
            # Reset the PYTHONPATH if using a conda environment to avoid conflicts with pre-commit
            if [ -n "${CONDA_DEFAULT_ENV}" ]; then
                cache_pythonpath=${PYTHONPATH}
                export PYTHONPATH=""
            fi

            # Ensure pre-commit is installed
            if ! command -v pre-commit &>/dev/null; then
                echo "[INFO] Installing pre-commit..."
                pip install pre-commit
            fi

            echo "[INFO] Formatting the repository..."
            # Determine the repository root directory (assumes this script is at the repo root)
            UWLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
            cd ${UWLAB_PATH}
            pre-commit run --all-files
            cd - > /dev/null

            # Restore the PYTHONPATH if it was modified
            if [ -n "${CONDA_DEFAULT_ENV}" ]; then
                export PYTHONPATH=${cache_pythonpath}
            fi
            shift
            break
            ;;
        -t|--test)
            echo "[INFO] Running tests..."
            # Insert your test command here (for example, pytest or another test runner)
            # pytest
            ;;
        *)
            echo "[ERROR] Unknown option: $key"
            print_help
            exit 1
            ;;
    esac
    shift
done
