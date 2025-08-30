#!/bin/bash

# Bootstrap Environment Script
# Sets up virtual environment, installs dependencies, initializes logs

set -e

echo "Bootstrapping PSE-LLM environment..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -e .

# Install dev dependencies
pip install -e .[dev]

# Initialize logs directory
if [ ! -d "logs" ]; then
    mkdir logs
    echo "Logs directory created."
else
    echo "Logs directory already exists."
fi

# Check tools
echo "Checking tools..."
if command -v ruff &> /dev/null; then
    echo "ruff available"
else
    echo "Warning: ruff not found"
fi

if command -v mypy &> /dev/null; then
    echo "mypy available"
else
    echo "Warning: mypy not found"
fi

if command -v pytest &> /dev/null; then
    echo "pytest available"
else
    echo "Warning: pytest not found"
fi

echo "Bootstrap complete. Activate venv with: source venv/bin/activate"