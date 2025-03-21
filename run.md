chmod a+x ./setup.sh
ollama serve
ollama pull qwen2.5:7b
python3 temp.py