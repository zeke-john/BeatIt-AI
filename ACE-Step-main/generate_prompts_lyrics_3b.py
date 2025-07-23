#!/usr/bin/env python3

import argparse
import os

import torch
import google.generativeai as genai
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from generate_prompts_lyrics import inference, parse_prompt_lyrics, extract_artist_names


def load_model(model_path: str):
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


def do_files(data_dir, overwrite, do_lyrics, gemini_api_key):
    model, processor = load_model("Qwen/Qwen2.5-Omni-7B")
    
    # Initialize Gemini
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

    # Formats supported by torchaudio
    extensions = {
        ".aac",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".wav",
    }

    for file in sorted(os.listdir(data_dir)):
        stem, ext = os.path.splitext(file)
        if ext.lower() not in extensions:
            continue

        file_path = os.path.join(data_dir, file)
        stem_path = os.path.join(data_dir, stem)
        prompt_path = stem_path + "_prompt.txt"
        lyrics_path = stem_path + "_lyrics.txt"

        need_prompt = overwrite or (not os.path.exists(prompt_path))
        need_lyrics = do_lyrics and (overwrite or (not os.path.exists(lyrics_path)))

        if not (need_prompt or need_lyrics):
            continue

        print(file)
        
        # Extract artist names using Gemini
        artist_names = extract_artist_names(file, gemini_model)
        print(f"Extracted artist names: {artist_names}")
        
        # We don't need to generate lyrics anymore, but we still call inference for tags
        # Set do_lyrics to False to avoid generating lyrics content
        content = inference(file_path, model, processor, False)
        prompt, lyrics = parse_prompt_lyrics(content, artist_names)

        if need_prompt:
            with open(prompt_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(prompt)
        if need_lyrics:
            with open(lyrics_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(lyrics)


@torch.inference_mode
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\data\audio")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lyrics", action="store_true")
    parser.add_argument("--gemini_api_key", type=str, required=True, help="Gemini API key for artist name extraction")
    args = parser.parse_args()

    do_files(
        data_dir=args.data_dir,
        overwrite=args.overwrite,
        do_lyrics=args.lyrics,
        gemini_api_key=args.gemini_api_key,
    )


if __name__ == "__main__":
    main()
