#!/bin/bash
set -e

# Force Python 3.12
export PYTHON_VERSION=3.12.0
export PIP_NO_BUILD_ISOLATION=1

# Uninstall Python 3.14 if it exists
which python3.14 && rm -rf /opt/pyenv/versions/3.14.3 || true

# Install Python 3.12 explicitly
python3.12 -m pip install --upgrade pip

# Install ONLY pre-built wheels, NO source compilation
python3.12 -m pip install --only-binary :all: \
  fastapi==0.104.1 \
  uvicorn==0.24.0 \
  pydantic==2.4.2 \
  numpy==2.1.0 \
  joblib==1.4.2 \
  pytest==7.4.3 \
  httpx==0.25.2

echo "Build completed successfully!"
