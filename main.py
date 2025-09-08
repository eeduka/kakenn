import sounddevice as sd
import numpy as np
import torchaudio
import torch
import os
import time
from transformers import pipeline

SAMPLE_RATE = 16000  # サンプリングレート
DURATION = 3         # 1回あたりの録音時間（秒）
OUTPUT_FILE = "output.wav"  # 一時保存用
MODEL = "openai/whisper-small"
LANGUAGE = "english"   # 小文字推奨

device = 0 if torch.cuda.is_available() else -1
print("使用デバイス:", "cuda" if device == 0 else "cpu")

pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL,
    device=device
)


def record_audio(filename=OUTPUT_FILE, duration=DURATION, samplerate=SAMPLE_RATE):

    print("録音中...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    audio_tensor = torch.from_numpy(audio.T)
    torchaudio.save(filename, audio_tensor, samplerate)
    print("録音終了。")


def recognize_and_control(filename=OUTPUT_FILE):
    result = pipe(
        filename,
        generate_kwargs={
            "language": LANGUAGE,
            "task": "transcribe"
        }
    )
    text = result["text"].strip().lower()
    print("認識結果:", text)

    if text in [" go", " go."]:
        print("➡ モーター回る")
    elif text in ["stop", "stop."]:
        print("モーター止まる")
    else:
        print("miss")


def main_loop():
    print("リアルタイム音声認識開始 ")
    try:
        while True:
            record_audio()
            recognize_and_control()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("終了しました")

if __name__ == "__main__":
    main_loop()
