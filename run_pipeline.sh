#!/bin/bash

set -e

python -m src.generator
python -m src.etl
python -m src.data_quality
python -m src.anomaly_detection
python -m streamlit run src/dashboard.py