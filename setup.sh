#!/usr/bin/env bash
set -e

# Colors
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
NC="\033[0m" # No Color

PYTHON_VERSION="3.14.0"
VENV_DIR="venv"

echo -e "${BLUE}=== Checking for Python >= $PYTHON_VERSION ===${NC}"

# Detect OS
OS="$(uname -s)"
echo -e "${BLUE}Detected OS: $OS${NC}"

PYTHON_CMD=""

# Function to compare versions
version_ge() {
    # returns 0 if $1 >= $2
    IFS='.' read -r i1 i2 i3 <<< "$1"
    IFS='.' read -r j1 j2 j3 <<< "$2"

    i1=${i1:-0}; i2=${i2:-0}; i3=${i3:-0}
    j1=${j1:-0}; j2=${j2:-0}; j3=${j3:-0}

    if (( i1 > j1 )); then return 0; fi
    if (( i1 < j1 )); then return 1; fi
    if (( i2 > j2 )); then return 0; fi
    if (( i2 < j2 )); then return 1; fi
    if (( i3 >= j3 )); then return 0; else return 1; fi
}

# Detect python
detect_python() {
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        PYTHON_CMD=""
    fi

    if [ -n "$PYTHON_CMD" ]; then
        VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
        if version_ge "$VERSION" "$PYTHON_VERSION"; then
            echo -e "${GREEN}Python $VERSION found (>= $PYTHON_VERSION)${NC}"
            return 0
        else
            echo -e "${YELLOW}Python found ($VERSION), but < $PYTHON_VERSION${NC}"
            return 1
        fi
    else
        echo -e "${RED}Python not found${NC}"
        return 1
    fi
}

# Install Python (if needed)
install_python() {
    echo -e "${BLUE}Installing Python >= $PYTHON_VERSION ...${NC}"
    if [[ "$OS" == "Darwin" ]]; then
        if ! command -v brew >/dev/null 2>&1; then
            echo -e "${YELLOW}Homebrew not found. Installing Homebrew...${NC}"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install python || echo -e "${YELLOW}Python might already exist in brew${NC}"
    elif [[ "$OS" == "Linux" ]]; then
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update
            sudo apt-get install -y python3 python3-venv python3-pip
        elif command -v yum >/dev/null 2>&1; then
            sudo yum install -y python3 python3-venv python3-pip
        else
            echo -e "${RED}Please install Python >= $PYTHON_VERSION manually.${NC}"
            exit 1
        fi
    elif [[ "$OS" == MINGW* || "$OS" == CYGWIN* || "$OS" == MSYS* ]]; then
        echo -e "${RED}Windows detected. Please install Python >= $PYTHON_VERSION manually from https://www.python.org/downloads/windows/${NC}"
        exit 1
    else
        echo -e "${RED}Unsupported OS.${NC}"
        exit 1
    fi

    detect_python || { echo -e "${RED}Python installation failed or version mismatch.${NC}"; exit 1; }
}

# Check Python
if ! detect_python; then
    install_python
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${BLUE}Creating virtual environment in $VENV_DIR ...${NC}"
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment ...${NC}"
if [[ "$OS" == MINGW* || "$OS" == MSYS* || "$OS" == CYGWIN* ]]; then
    source "$VENV_DIR/Scripts/activate"
else
    source "$VENV_DIR/bin/activate"
fi

# Upgrade pip
echo -e "${BLUE}Upgrading pip ...${NC}"
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo -e "${BLUE}Installing requirements from requirements.txt ...${NC}"
    pip install -r requirements.txt
else
    echo -e "${YELLOW}requirements.txt not found!${NC}"
fi

echo -e "${GREEN}=== Setup complete! Virtual environment activated ===${NC}"

source "$VENV_DIR/bin/activate"

echo -e "${BLUE}DONT FORGET TO ACTIVATE ENV MANUALLY, run:${NC}"
if [[ "$OS" == MINGW* || "$OS" == MSYS* || "$OS" == CYGWIN* ]]; then
    echo -e "  ${RED}$VENV_DIR\\Scripts\\activate${NC}"
else
    echo -e "  ${RED}source $VENV_DIR/bin/activate${NC}"
fi

# to run the project use: steamlit run /src/dashboard.py
echo -e "${BLUE}To run the dashboard, use:${NC}"
echo -e "  ${RED}streamlit run src/dashboard.py${NC}"
