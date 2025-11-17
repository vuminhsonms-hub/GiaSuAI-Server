import os
import uuid
import wave
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
import openai

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 16000  # trùng với ESP32

@app.post("/ask")
async def ask_audio(request: Request):
    # 1) Nhận raw PCM 16-bit mono từ ESP32
    raw = await request.body()
    temp_in = f"input_{uuid.uuid4()}.wav"
    temp_out = f"output_{uuid.uuid4()}.wav"

    # Ghi trực tiếp raw PCM thành file WAV (không cần numpy/soundfile)
    with wave.open(temp_in, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw)

    # 2) STT: giọng nói -> văn bản
    transcript = openai.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=open(temp_in, "rb"),
        response_format="text"
    )
    question = transcript
    print("Người hỏi:", question)

    # 3) GPT: trả lời như gia sư Vật lí
    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Bạn là gia sư Vật lí THPT, giải thích ngắn gọn, dễ hiểu, phù hợp học sinh Việt Nam."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    answer = completion.choices[0].message.content
    print("AI trả lời:", answer)

    # 4) TTS: văn bản -> WAV
    speech = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=answer,
        response_format="wav"
    )

    with open(temp_out, "wb") as f:
        f.write(speech.read())

    # Trả WAV cho ESP32
    return FileResponse(temp_out, media_type="audio/wav")
