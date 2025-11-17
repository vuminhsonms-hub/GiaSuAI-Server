import os
import uuid
import wave
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import openai

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 16000  # trùng với ESP32

@app.post("/ask")
async def ask_audio(request: Request):
    try:
        # 1) Nhận raw PCM 16-bit mono từ ESP32
        raw = await request.body()
        temp_in = f"input_{uuid.uuid4()}.wav"

        # Ghi raw PCM thành file WAV
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

        # 4) Trả về JSON
        return JSONResponse({"question": question, "answer": answer})

    except Exception as e:
        # Nếu có lỗi in ra log và trả về thông báo lỗi
        print("LỖI SERVER:", repr(e))
        return JSONResponse(
            {"error": "Server error", "detail": repr(e)},
            status_code=500
        )
