#!/usr/bin/env python3

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

QWEN_SAMPLE_RATE = 16000

QWEN_SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# Modified prompt to focus on hip-hop/rap genres
PROMPT = r"""Analyze the input hip-hop/rap beat audio:
1. `genre`: Most representative genres of the audio (must be related to hip-hop, rap, trap, drill, or RnB since this is a hip hop beat).
2. `subgenre`: Three or more tags of specific sub-genres and techniques (must be related to hip-hop, rap, trap, drill, or RnB styles since this is a hip hop beat).
3. `instrument`: All audibly present instruments in the audio, except vocal.
4. `tempo`: Tags describing the tempo of the audio. Do not use number or BPM.
5. `mood`: Tags describing the mood of the audio.
6. `has_vocal`: Whether there is any vocal in the audio.
7. `vocal`: If there is any vocal, then output a list of tags describing the vocal timbre. Otherwise, output an empty list.

Note: This is a hip-hop/rap beat, so ensure all genres and subgenres are related to hip-hop, rap, trap, drill, or RnB music styles.

Output format:
```json
{
  "genre": <str list>,
  "subgenre": <str list>,
  "instrument": <str list>,
  "tempo": <str list>,
  "mood": <str list>,
  "has_vocal": <bool>,
  "vocal": <str list>
}
```"""

PROMPT_LYRICS = r"""Analyze the input hip-hop/rap beat audio:
1. `genre`: Most representative genres of the audio (must be related to hip-hop, rap, trap, drill, or RnB since this is a hip hop beat).
2. `subgenre`: Three or more tags of specific sub-genres and techniques (must be related to hip-hop, rap, trap, drill, or RnB since this is a hip hop beat).
3. `instrument`: All audibly present instruments in the audio, except vocal.
4. `tempo`: Tags describing the tempo of the audio. Do not use number or BPM.
5. `mood`: Tags describing the mood of the audio.
6. `has_vocal`: Whether there is any vocal in the audio.
7. `vocal`: If there is any vocal, then output a list of tags describing the vocal timbre. Otherwise, output an empty list.
8. `lyrics`: If there is any vocal, then transcribe the lyrics and output at most 1000 characters. Otherwise, output an empty string. Use \n after each sentence.

Note: This is a hip-hop/rap beat, so ensure all genres and subgenres are related to hip-hop, rap, trap, drill, or RnB music styles.

Output format:
```json
{
  "genre": <str list>,
  "subgenre": <str list>,
  "instrument": <str list>,
  "tempo": <str list>,
  "mood": <str list>,
  "has_vocal": <bool>,
  "vocal": <str list>,
  "lyrics": <str>
}
```"""


@classmethod
def patched_from_config(cls, config, *args, **kwargs):
    kwargs.pop("trust_remote_code", None)
    model = cls._from_config(config, **kwargs)
    return model


Qwen2_5OmniForConditionalGeneration.from_config = patched_from_config


class Qwen2_5OmniThinkerGPTQ(BaseGPTQModel):
    loader = Qwen2_5OmniForConditionalGeneration
    base_modules = [
        "thinker.model.embed_tokens",
        "thinker.model.norm",
        "thinker.audio_tower",
        "thinker.model.rotary_emb",
    ]
    pre_lm_head_norm_module = "thinker.model.norm"
    require_monkeypatch = False
    layers_node = "thinker.model.layers"
    layer_type = "Qwen2_5OmniDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    def pre_quantize_generate_hook_start(self):
        self.thinker.audio_tower = self.thinker.audio_tower.to(
            self.quantize_config.device
        )

    def pre_quantize_generate_hook_end(self):
        self.thinker.audio_tower = self.thinker.audio_tower.to("cuda")

    def preprocess_dataset(self, sample):
        return sample


MODEL_MAP["qwen2_5_omni"] = Qwen2_5OmniThinkerGPTQ
SUPPORTED_MODELS.extend(["qwen2_5_omni"])


def load_model(model_path: str):
    if not os.path.exists(model_path):
        model_path = snapshot_download(repo_id=model_path)

    device_map = {
        "thinker.model": "cuda",
        "thinker.lm_head": "cuda",
        # "thinker.visual": "cuda",
        "thinker.audio_tower": "cuda",
        "talker": "cuda",
        "token2wav": "cuda",
    }

    model = GPTQModel.load(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


def read_audio(file_path):
    audio, sr = torchaudio.load(file_path)
    audio = audio[:, : sr * 360]
    if sr != QWEN_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, QWEN_SAMPLE_RATE)
        sr = QWEN_SAMPLE_RATE
    audio = audio.mean(dim=0, keepdim=True)
    return audio, sr


def inference(file_path, model, processor, do_lyrics):
    audio, _ = read_audio(file_path)
    audio = audio.numpy().squeeze(axis=0)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": PROMPT_LYRICS if do_lyrics else PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )

    # Copy tensors to GPU and match dtypes
    ks = list(inputs.keys())
    for k in ks:
        if hasattr(inputs[k], "to"):
            inputs[k] = inputs[k].to("cuda")
            if inputs[k].dtype.is_floating_point:
                inputs[k] = inputs[k].to(model.dtype)

    output_ids = model.thinker.generate(
        **inputs,
        max_new_tokens=1000,
        use_audio_in_video=False,
    )

    generate_ids = output_ids[:, inputs.input_ids.shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


def extract_artist_names(filename, gemini_model):
    """Extract artist names from filename using Gemini"""
    print(f"🎵 Extracting artist names from: {filename}")
    
    prompt = f"""Extract the artist name(s) from this music file name: "{filename}"

This is a hip-hop/rap beat file. The filename may contain multiple artists separated by various symbols like &, +, x, feat, ft, etc.

Rules:
1. Extract all artist names mentioned in the filename
2. Clean up the names (remove "Type Beat", "FREE", parentheses, etc.)
3. If multiple artists, separate them with " , "
4. Return only the artist name(s), nothing else
5. If no clear artist name is found, return "Unknown Artist"

Examples:
- "Drake & Central Cee Type Beat" → "Drake, Central Cee"
- "(FREE) J Cole x Logic Type Beat" → "J Cole, Logic"
- "21 Savage Type Beat" → "21 Savage"

Filename: {filename}
Artist name(s):"""

    try:
        print("🤖 Calling Gemini API for artist extraction...")
        response = gemini_model.generate_content(prompt)
        artist_names = response.text.strip()
        result = artist_names if artist_names else "Unknown Artist"
        print(f"✅ Artist names extracted: {result}")
        return result
    except Exception as e:
        print(f"❌ Error extracting artist names: {e}")
        return "Unknown Artist"


def parse_prompt_lyrics(content, artist_names):
    print("🔍 Parsing audio analysis response...")
    print(f"📝 Raw content length: {len(content)} characters")
    
    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    # Always use "[instrumental]" for lyrics
    lyrics = "[instrumental]"
    print(f"🎤 Lyrics set to: {lyrics}")
    
    try:
        tags = []
        data = json.loads(content)
        print(f"📊 Parsed JSON data: {data}")
        
        tags += data["genre"]
        tags += data["subgenre"]
        tags += data["instrument"]
        tags += data["tempo"]
        tags += data["mood"]
        tags += [x.strip() + " vocal" for x in data["vocal"] if x != "vocal"]

        tags = [x.strip().lower() for x in tags]
        # The order of tags does not matter, so we sort them here
        # Tags will be shuffled in training
        tags = sorted(set(tags))
        
        print(f"🏷️  Extracted tags: {tags}")
        
        # Format: artist name(s) + "hip hop rap beat" + actual tags
        prompt = f"{artist_names} hip hop rap beat, " + ", ".join(tags)
        print(f"✨ Final prompt: {prompt}")
        
    except Exception as e:
        print(f"❌ Failed to parse content: {e}")
        print(f"📄 Raw content: {content}")
        # Fallback prompt with artist name
        prompt = f"{artist_names} hip hop rap beat"
        print(f"🔄 Fallback prompt: {prompt}")

    return prompt, lyrics


def do_files(data_dir, overwrite, do_lyrics, gemini_api_key):
    print("🚀 Starting audio analysis and prompt generation...")
    print(f"📁 Data directory: {data_dir}")
    print(f"🔄 Overwrite mode: {overwrite}")
    print(f"🎤 Generate lyrics: {do_lyrics}")
    
    print("\n🤖 Loading Qwen model...")
    model, processor = load_model("Qwen/Qwen2.5-Omni-7B-GPTQ-Int4")
    print("✅ Qwen model loaded successfully")
    
    # Initialize Gemini
    print("\n🔗 Initializing Gemini API...")
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini API initialized successfully")

    # Formats supported by torchaudio
    extensions = {
        ".aac",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".wav",
    }

    print(f"\n🎵 Scanning for audio files in: {data_dir}")
    audio_files = [f for f in sorted(os.listdir(data_dir)) if os.path.splitext(f)[1].lower() in extensions]
    print(f"📊 Found {len(audio_files)} audio files to process")

    for i, file in enumerate(audio_files, 1):
        print(f"\n{'='*60}")
        print(f"🎵 Processing file {i}/{len(audio_files)}: {file}")
        print(f"{'='*60}")
        
        stem, ext = os.path.splitext(file)

        file_path = os.path.join(data_dir, file)
        stem_path = os.path.join(data_dir, stem)
        prompt_path = stem_path + "_prompt.txt"
        lyrics_path = stem_path + "_lyrics.txt"

        need_prompt = overwrite or (not os.path.exists(prompt_path))
        need_lyrics = do_lyrics and (overwrite or (not os.path.exists(lyrics_path)))

        print(f"📝 Need prompt: {need_prompt}")
        print(f"🎤 Need lyrics: {need_lyrics}")

        if not (need_prompt or need_lyrics):
            print("⏭️  Skipping - files already exist and overwrite is False")
            continue

        # Extract artist names using Gemini
        artist_names = extract_artist_names(file, gemini_model)
        
        # We don't need to generate lyrics anymore, but we still call inference for tags
        # Set do_lyrics to False to avoid generating lyrics content
        print("\n🎧 Analyzing audio with Qwen model...")
        content = inference(file_path, model, processor, False)
        print("✅ Audio analysis completed")
        
        prompt, lyrics = parse_prompt_lyrics(content, artist_names)

        if need_prompt:
            print(f"💾 Writing prompt to: {prompt_path}")
            with open(prompt_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(prompt)
            print("✅ Prompt file saved")
            
        if need_lyrics:
            print(f"💾 Writing lyrics to: {lyrics_path}")
            with open(lyrics_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(lyrics)
            print("✅ Lyrics file saved")
            
        print(f"✨ Completed processing: {file}")

    print(f"\n🎉 All done! Processed {len(audio_files)} files successfully")


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
