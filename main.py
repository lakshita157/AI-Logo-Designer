from diffusers import StableDiffusionPipeline
!pip install diffusers

!pip install nest_asyncio

# Importing necessary libraries
import os
import openai
import json
import torch
import tensorflow as tf
import threading
import asyncio
import spacy
import nltk
from fastapi import FastAPI, HTTPException
from gradio import Interface
from diffusers import StableDiffusionPipeline
from langchain.prompts import PromptTemplate
from openai import ChatCompletion
from PIL import Image, ImageDraw, ImageFont
from fastapi.responses import FileResponse
import uvicorn

# Resolving nested asyncio issues
import nest_asyncio
nest_asyncio.apply()

# Setting up Hugging Face Stable Diffusion
MODEL_ID = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Initializing FastAPI app
app = FastAPI()

# Loading spaCy model and downloading NLTK resources
nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

# Function to preprocess text prompts
def preprocess_prompt(prompt: str) -> str:
    """
    Preprocesses the text prompt by extracting keywords using spaCy.
    """
    doc = nlp(prompt)
    keywords = [token.text for token in doc if token.is_alpha]
    return " ".join(keywords)

# Function to enhance prompts using OpenAI API
def enhance_prompt(prompt: str) -> str:
    """
    Enhances the prompt by using OpenAI GPT API.
    """
    api_key = os.getenv("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")  # Use environment variable for security
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key is not set.")
    openai.api_key = api_key
    completion = ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Enhance this logo design prompt: {prompt}"}]
    )
    return completion["choices"][0]["message"]["content"]

# Function to generate logos using Stable Diffusion
def generate_logo(prompt: str) -> str:
    """
    Generates a logo image using Stable Diffusion and saves it locally.
    """
    enhanced_prompt = preprocess_prompt(prompt)
    image = pipe(enhanced_prompt).images[0]
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((10, 10), "AI Logo", fill="white", font=font)
    output_path = "generated_logo.png"
    image.save(output_path)
    return output_path

# Gradio interface for user interaction
def gradio_interface(prompt: str) -> Image:
    """
    Gradio interface function for generating logos.
    """
    enhanced_prompt = enhance_prompt(prompt)
    logo_path = generate_logo(enhanced_prompt)
    return Image.open(logo_path)

# Placeholder for fine-tuning the model
def fine_tune_model(dataset_path: str):
    """
    Placeholder function for fine-tuning Stable Diffusion.
    """
    print(f"Fine-tuning model with dataset at {dataset_path}")
    # Implement fine-tuning logic if needed

# Gradio app setup
gradio_app = Interface(
    fn=gradio_interface,
    inputs="text",
    outputs="image",
    title="Enhanced AI Logo Designer",
    description="Enter a text prompt to generate an AI logo. OpenAI and Stable Diffusion power the system.",
)

# FastAPI endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the Enhanced AI Logo Designer API"}

@app.post("/generate/")
def generate_api(prompt: str):
    """
    API endpoint to generate a logo based on the input prompt.
    """
    try:
        enhanced_prompt = enhance_prompt(prompt)
        logo_path = generate_logo(enhanced_prompt)
        return FileResponse(logo_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Running Gradio and FastAPI concurrently with graceful shutdown
def run():
    """
    Runs Gradio and FastAPI concurrently in separate threads.
    """
    def start_gradio():
        gradio_app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)

    def start_fastapi():
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        server.run()

    try:
        gradio_thread = threading.Thread(target=start_gradio, daemon=True)
        gradio_thread.start()
        start_fastapi()
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        print("Cleaning up resources...")

if __name__ == "__main__":
