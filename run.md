chmod a+x ./setup.sh
pip install -r requirements.txt
ollama serve
ollama pull qwen2.5:7b
python3 temp.py
