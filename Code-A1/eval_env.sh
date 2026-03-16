#!/bin/bash
set -e

cd $(dirname "$0")
uv venv .venv-eval --python 3.10 --seed
source .venv-eval/bin/activate
uv pip install bigcodebench beautifulsoup4 blake3 chardet cryptography datetime Django dnspython docxtpl Faker flask_login flask_restful flask_wtf Flask-Mail flask folium gensim geopandas geopy holidays keras Levenshtein librosa lxml matplotlib mechanize natsort networkx nltk numba numpy opencv-python-headless openpyxl pandas Pillow prettytable psutil pycryptodome pyfakefs pyquery pytesseract pytest python_http_client python-dateutil python-docx python-Levenshtein-wheels pytz PyYAML requests_mock requests rsa scikit-image scikit-learn scipy seaborn selenium sendgrid shapely soundfile statsmodels sympy tensorflow textblob texttable Werkzeug wikipedia wordcloud wordninja WTForms xlrd xlwt xmltodict
uv pip install flash-attn --no-build-isolation
uv pip install 'vllm==0.11.0'
uv pip install cosmic-ray pytest pytest-cov
uv pip install wandb
uv pip install sandbox_fusion