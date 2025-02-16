# Install required libraries
!pip install torch torchvision torchaudio diffusers transformers fastapi uvicorn gradio nest_asyncio spacy nltk

# Download spaCy language model
!python -m spacy download en_core_web_sm
