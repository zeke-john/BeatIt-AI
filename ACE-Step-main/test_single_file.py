#!/usr/bin/env python3
"""
Test script to process only ONE audio file to verify the pipeline works
Use this before running the full batch on GPU VM
"""

import os
# Fix tokenizers parallelism issue that causes hanging
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import torch
import torchaudio
from gptqmodel import GPTQModel
from gptqmodel.models.auto import MODEL_MAP, SUPPORTED_MODELS
from gptqmodel.models.base import BaseGPTQModel
from huggingface_hub import snapshot_download
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor
import google.generativeai as genai

from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration

# Import functions from the main script
from generate_prompts_lyrics import (
    load_model, 
    inference, 
    extract_artist_names, 
    parse_prompt_lyrics,
    Qwen2_5OmniThinkerGPTQ,
    MODEL_MAP,
    SUPPORTED_MODELS
)

def test_single_file(data_dir, gemini_api_key):
    print("🧪 TEST MODE: Processing only ONE file to verify pipeline")
    print(f"📁 Data directory: {data_dir}")
    
    # Formats supported by torchaudio
    extensions = {
        ".aac",
        ".flac", 
        ".m4a",
        ".mp3",
        ".ogg",
        ".wav",
    }

    # Find the first audio file
    print(f"\n🔍 Looking for audio files in: {data_dir}")
    audio_files = [f for f in sorted(os.listdir(data_dir)) if os.path.splitext(f)[1].lower() in extensions]
    
    if not audio_files:
        print("❌ No audio files found!")
        return
        
    # Take only the first file
    test_file = audio_files[0]
    print(f"🎵 Found {len(audio_files)} audio files total")
    print(f"🎯 Testing with: {test_file}")
    
    print("\n🤖 Loading Qwen model...")
    model, processor = load_model("Qwen/Qwen2.5-Omni-7B-GPTQ-Int4")
    print("✅ Qwen model loaded successfully")
    
    # Initialize Gemini
    print("\n🔗 Initializing Gemini API...")
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini API initialized successfully")

    # Process the single test file
    print(f"\n{'='*60}")
    print(f"🎵 TESTING: {test_file}")
    print(f"{'='*60}")
    
    stem, ext = os.path.splitext(test_file)
    file_path = os.path.join(data_dir, test_file)
    stem_path = os.path.join(data_dir, stem)
    prompt_path = stem_path + "_prompt.txt"
    lyrics_path = stem_path + "_lyrics.txt"

    # Extract artist names using Gemini
    artist_names = extract_artist_names(test_file, gemini_model)
    
    # Analyze audio with Qwen model
    print("\n🎧 Analyzing audio with Qwen model...")
    print("⏰ This may take 30-60 seconds on GPU...")
    content = inference(file_path, model, processor, False)
    print("✅ Audio analysis completed")
    
    # Parse and format the prompt
    prompt, lyrics = parse_prompt_lyrics(content, artist_names)

    # Save the files
    print(f"\n💾 Writing test files...")
    with open(prompt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(prompt)
    print(f"✅ Prompt saved to: {prompt_path}")
    
    with open(lyrics_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(lyrics)
    print(f"✅ Lyrics saved to: {lyrics_path}")
    
    print(f"\n🎉 TEST COMPLETED SUCCESSFULLY!")
    print(f"📊 Results:")
    print(f"   🎵 File: {test_file}")
    print(f"   🎤 Artist: {artist_names}")
    print(f"   ✨ Prompt: {prompt}")
    print(f"   🎼 Lyrics: {lyrics}")
    print(f"\n✅ Pipeline verified! Ready for GPU VM processing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test single file processing")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--gemini_api_key", type=str, required=True, help="Gemini API key")
    args = parser.parse_args()

    test_single_file(args.data_dir, args.gemini_api_key) 