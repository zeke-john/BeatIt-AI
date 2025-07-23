#!/usr/bin/env python3

import argparse
import base64
import os
from io import BytesIO

import requests
import torchaudio
import google.generativeai as genai

from generate_prompts_lyrics import (
    PROMPT,
    PROMPT_LYRICS,
    parse_prompt_lyrics,
    read_audio,
    extract_artist_names,
)


def inference(file_path, host, port, do_lyrics, temperature):
    audio, sr = read_audio(file_path)

    buffer = BytesIO()
    torchaudio.save(buffer, audio, sample_rate=sr, format="wav")
    audio = base64.b64encode(buffer.getvalue()).decode("utf-8")
    del buffer

    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_LYRICS if do_lyrics else PROMPT,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio,
                            "format": "wav",
                        },
                    },
                ],
            },
        ],
        "cache_prompt": True,
        "temperature": temperature,
        "top_k": 50,
        "top_p": 1,
        "min_p": 0,
        "max_tokens": 1000,
    }

    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    content = response.json()
    content = content["choices"][0]["message"]["content"]
    return content


def do_files(data_dir, host, port, overwrite, do_lyrics, temperature, gemini_api_key):
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
        content = inference(file_path, host, port, False, temperature)
        prompt, lyrics = parse_prompt_lyrics(content, artist_names)

        if need_prompt:
            with open(prompt_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(prompt)
        if need_lyrics:
            with open(lyrics_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(lyrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\data\audio")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lyrics", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--gemini_api_key", type=str, required=True, help="Gemini API key for artist name extraction")
    args = parser.parse_args()

    do_files(
        data_dir=args.data_dir,
        host=args.host,
        port=args.port,
        overwrite=args.overwrite,
        do_lyrics=args.lyrics,
        temperature=args.temperature,
        gemini_api_key=args.gemini_api_key,
    )


if __name__ == "__main__":
    main()
