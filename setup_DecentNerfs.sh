#!/bin/bash
# ----------------------------------------------------------
# Setup script for DecentNerfs project
# ----------------------------------------------------------
# Run this script from:
# /Users/krish/Workbook/SBU/sem1/computational-photography/DecentNerfs
# ----------------------------------------------------------

BASE_DIR="$(pwd)"
echo "📁 Setting up DecentNerfs project in: $BASE_DIR"

# 1️⃣ Create required folders (leave your data/ intact)
mkdir -p utils
mkdir -p outputs

# 2️⃣ Create empty helper files in utils/
touch utils/{__init__.py,tinynerf.py,secure_agg.py,plotting.py}

# 3️⃣ Create placeholder outputs files
touch outputs/{.gitkeep,logs.txt}

# 4️⃣ Create requirements.txt
cat > requirements.txt <<'EOF'
torch>=2.0.0
numpy
pillow
matplotlib
EOF

# 5️⃣ Create README.md
cat > README.md <<'EOF'
# DecentNerfs Prototype

Minimal CPU-friendly prototype of *DecentNeRFs: Decentralized Neural Radiance Fields from Crowdsourced Images*  
Demonstrates decentralized/federated NeRF training using the Trevi Fountain Phototourism dataset.

### Directory Layout
