#!/bin/bash
sudo apt-get update
sudo apt-get install -y docker python3 python3-pip
pip3 install matplotlib pandas
pip3 install python-dotenv langchain langchain_openai langchain_chroma openai chromadb tiktoken tavily-python
pip3 install fastapi pydantic uvicorn
pip3 install -U langchain-community
pip3 install -U langchain_huggingface
pip3 install -qU langchain-google-genai
pip3 install -U bitsandbytes
pip3 install transformers accelerate
pip3 install --upgrade transformers accelerate