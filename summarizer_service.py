#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI שירות לתקצור טקסטים בעברית:
1. מקבל טקסט בעברית
2. מתרגם לאנגלית עם NLLB
3. יוצר תקציר של 5 נקודות עם Phi-3
4. מחזיר כל נקודה מיידית (streaming)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TextIteratorStreamer
import ollama
import torch
import json
import re
import time
from typing import Generator, AsyncGenerator, Dict, Any
import asyncio
import threading

# אתחול FastAPI
app = FastAPI(
    title="Hebrew Text Summarizer",
    description="שירות לתקצור טקסטים בעברית עם תרגום ו-streaming response",
    version="1.0.0"
)

# מודלים גלובליים - נטען פעם אחת בהתחלה
nllb_tokenizer = None
nllb_model = None


class TextRequest(BaseModel):
    text: str
    max_summary_points: int = 5


class SummaryPoint(BaseModel):
    point_number: int
    content: str
    timestamp: float


def initialize_nllb():
    """אתחול מודל NLLB לתרגום"""
    global nllb_tokenizer, nllb_model

    if nllb_tokenizer is None or nllb_model is None:
        print("🔄 טוען מודל NLLB...")
        model_name = "facebook/nllb-200-distilled-600M"
        nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("✅ מודל NLLB נטען בהצלחה")

def stream_translate_hebrew_to_english(hebrew_text: str):
    """תרגום מעברית לאנגלית עם זרימה (streaming)"""
    initialize_nllb()

    # קידוד הטקסט
    inputs = nllb_tokenizer(
        hebrew_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    # שפת יעד
    tgt_lang = "eng_Latn"
    tgt_id = nllb_tokenizer.convert_tokens_to_ids(tgt_lang)
    if tgt_id is None:
        raise ValueError(f"לא נמצא קוד שפה: {tgt_lang}")

    # יוצרים streamer שיחזיר טקסט בזמן אמת
    streamer = TextIteratorStreamer(nllb_tokenizer, skip_special_tokens=True)

    # הפעלת הגנרציה בת'רד נפרד כדי לא לחסום
    generation_kwargs = dict(
        **inputs,
        forced_bos_token_id=tgt_id,
        max_length=1000,
        num_beams=3,
        early_stopping=True,
        streamer=streamer
    )

    thread = threading.Thread(target=nllb_model.generate, kwargs=generation_kwargs)
    thread.start()

    # נחזיר generator שמפיק חלקים מהתרגום בזמן אמת
    for new_text in streamer:
        yield new_text


def translate_hebrew_to_english(hebrew_text: str) -> str:
    """תרגום מעברית לאנגלית"""
    initialize_nllb()

    inputs = nllb_tokenizer(
        hebrew_text,  # הטקסט בעברית שרוצים לתרגם
        return_tensors="pt",  # החזר במבנה של PyTorch tensors (במקום רשימות רגילות)
        max_length=1024,  # הגבל את הטקסט למקסימום 512 טוקנים (מילים/חלקי מילים)
        truncation=True  # אם הטקסט ארוך מ-512 טוקנים, חתוך אותו (במקום לזרוק שגיאה)
    )
    # קוד שפת יעד - אנגלית
    tgt_lang = "eng_Latn"
    tgt_id = nllb_tokenizer.convert_tokens_to_ids(tgt_lang)

    if tgt_id is None:
        raise ValueError(f"לא נמצא קוד שפה: {tgt_lang}")

    with torch.no_grad():
        generated_tokens = nllb_model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            max_length=1000,
            num_beams=3,
            early_stopping=True
        )

    translation = nllb_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation.strip()


def check_ollama_connection() -> bool:
    """בדיקת חיבור ל-Ollama"""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def ensure_phi3_model() -> bool:
    """וידוא שמודל Phi-3 זמין"""
    try:
        models = ollama.list()
        available_models = [m.get("model", "") for m in models.get("models", [])]

        model_name = "phi3:mini"
        if not any(model_name in model for model in available_models):
            print(f"🔄 מוריד מודל {model_name}...")
            ollama.pull(model_name)
            print(f"✅ מודל {model_name} הורד בהצלחה")

        return True
    except Exception as e:
        print(f"❌ שגיאה בהורדת מודל: {e}")
        return False


async def stream_summary_points_with_phi3(english_text: str, num_points: int = 5):
    """זרימת נקודות תקציר ישירות מ-Ollama (Phi-3) בזמן אמת.

    קורא את הפלט המוזרם של המודל, מזהה התחלה וסיום של כל נקודה ממוספרת,
    ומשיב כל נקודה מיד כשהיא הושלמה.
    """
    prompt = f"""Please create a concise summary of the following text in exactly {num_points} bullet points.
Each point must be a single line and start with an explicit number like "1. ", "2. ", etc.
Do not add any prose before or after the list. Only output the numbered list.

Text to summarize:
{english_text}

Summary:"""

    # נשתמש בזיהוי גבולות פשוט לפי "X. " כדי להבטיח פלואו יציב בסטרימינג
    expected_index = 1
    buffer = ""

    try:
        # נעביר את הסטרים של Ollama לת'רד נפרד ונעביר צ'אנקים בתור אסינכרוני
        queue: asyncio.Queue = asyncio.Queue(maxsize=0)

        def producer():
            try:
                stream = ollama.chat(
                    model="phi3:mini",
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'max_tokens': 500
                    },
                    stream=True
                )
                for chunk in stream:
                    try:
                        queue.put_nowait(chunk)
                    except Exception:
                        break
            except Exception as e:
                try:
                    queue.put_nowait({'error': str(e)})
                except Exception:
                    pass
            finally:
                try:
                    queue.put_nowait(None)
                except Exception:
                    pass

        threading.Thread(target=producer, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, dict) and 'error' in item:
                raise HTTPException(detail=item['error'])

            piece = item.get('message', {}).get('content', '')
            if not piece:
                continue
            buffer += piece

            # ננסה לחלץ נקודות שהושלמו: בין "N. " ל-"N+1. "
            while True:
                start_marker = f"{expected_index}. "
                next_marker = f"{expected_index + 1}. "

                start_pos = buffer.find(start_marker)
                if start_pos == -1:
                    break

                next_pos = buffer.find(next_marker, start_pos + len(start_marker))
                if next_pos == -1:
                    break

                point_text = buffer[start_pos + len(start_marker):next_pos]
                point_text = point_text.strip().strip("-•* ")
                if point_text:
                    yield expected_index, point_text
                    expected_index += 1
                buffer = buffer[next_pos:]

                if expected_index > num_points:
                    break

            if expected_index > num_points:
                break

        # סוף הזרם: פלש נקודה אחרונה אם קיימת
        if expected_index <= num_points:
            start_marker = f"{expected_index}. "
            start_pos = buffer.find(start_marker)
            if start_pos != -1:
                tail = buffer[start_pos + len(start_marker):]
                tail = tail.strip().strip("-•* ")
                if tail:
                    yield expected_index, tail

    except Exception as e:
        raise HTTPException(e)


def parse_summary_points(summary_text: str) -> list[str]:
    """חילוץ נקודות התקציר מהטקסט"""
    # חיפוש אחר נקודות ממוספרות
    points = []

    # תבניות שונות של מספור
    patterns = [
        r'^\d+\.\s*(.+)$',  # 1. נקודה
        r'^•\s*(.+)$',  # • נקודה
        r'^-\s*(.+)$',  # - נקודה
        r'^\*\s*(.+)$',  # * נקודה
    ]

    lines = summary_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        for pattern in patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                points.append(match.group(1).strip())
                break
        else:
            # אם השורה לא מתחילה במספר, אבל היא לא ריקה
            # ואין לנו הרבה נקודות, נוסיף אותה
            if len(points) < 3 and len(line) > 10:
                points.append(line)

    return points[:5]  # מגבילים ל-5 נקודות מקסימום


async def generate_streaming_summary(request: TextRequest) -> AsyncGenerator[str, None]:
    """יצירת תקציר עם streaming response"""

    try:
        # שלב 1: בדיקת חיבור ל-Ollama
        if not check_ollama_connection():
            error_msg = {"error": "לא ניתן להתחבר ל-Ollama. ודא שהשירות רץ: ollama serve"}
            yield f"data: {json.dumps(error_msg, ensure_ascii=False)}\n\n"
            return

        # שלב 2: וידוא מודל Phi-3
        if not ensure_phi3_model():
            error_msg = {"error": "לא ניתן לטעון את מודל Phi-3"}
            yield f"data: {json.dumps(error_msg, ensure_ascii=False)}\n\n"
            return

        # שלב 3: תרגום לאנגלית
        yield f"data: {json.dumps({'status': 'מתרגם טקסט לאנגלית...'}, ensure_ascii=False)}\n\n"

        try:
            english_text = translate_hebrew_to_english(request.text)
            yield f"data: {json.dumps({'status': 'תרגום הושלם', 'translated_text': english_text}, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_msg = {"error": f"שגיאה בתרגום: {str(e)}"}
            yield f"data: {json.dumps(error_msg, ensure_ascii=False)}\n\n"
            return

        # שלב 4: יצירת תקציר בסטרימינג ושידור נקודות מיידית
        yield f"data: {json.dumps({'status': 'יוצר תקציר בסטרימינג...'}, ensure_ascii=False)}\n\n"

        sent_points = 0
        try:
            async for idx, point in stream_summary_points_with_phi3(english_text, request.max_summary_points):
                summary_point = SummaryPoint(
                    point_number=idx,
                    content=point,
                    timestamp=time.time()
                )
                sent_points += 1
                yield f"data: {json.dumps({'summary_point': summary_point.dict()}, ensure_ascii=False)}\n\n"

                if sent_points >= request.max_summary_points:
                    break

            # סיום
            yield f"data: {json.dumps({'status': 'completed', 'total_points': sent_points}, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_msg = {"error": f"שגיאה ביצירת תקציר בסטרימינג: {str(e)}"}
            yield f"data: {json.dumps(error_msg, ensure_ascii=False)}\n\n"
            return

    except Exception as e:
        error_msg = {"error": f"שגיאה כללית: {str(e)}"}
        yield f"data: {json.dumps(error_msg, ensure_ascii=False)}\n\n"


@app.on_event("startup")
async def startup_event():
    """אתחול השירות"""
    print("🚀 מתחיל שירות תקצור טקסטים")
    print("🔄 טוען מודלי תרגום...")
    initialize_nllb()


@app.get("/")
async def root():
    """דף בית"""
    return {
        "message": "שירות תקצור טקסטים בעברית",
        "endpoints": {
            "summarize": "/summarize - תקצור טקסט (POST)",
            "health": "/health - בדיקת תקינות"
        }
    }


@app.get("/health")
async def health_check():
    """בדיקת תקינות השירות"""
    ollama_status = check_ollama_connection()

    return {
        "status": "healthy" if ollama_status else "unhealthy",
        "services": {
            "nllb_translation": nllb_model is not None,
            "ollama_phi3": ollama_status
        },
        "timestamp": time.time()
    }


@app.post("/summarize")
async def summarize_text(request: TextRequest):
    """
    תקצור טקסט בעברית עם streaming response

    מקבל טקסט בעברית ומחזיר תקציר של עד 5 נקודות.
    כל נקודה נשלחת מיד כשהיא מוכנה.
    """

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="טקסט לא יכול להיות ריק")

    if len(request.text) < 50:
        raise HTTPException(status_code=400, detail="טקסט קצר מדי לתקצור (מינימום 50 תווים)")

    if request.max_summary_points < 1 or request.max_summary_points > 100:
        raise HTTPException(status_code=400, detail="מספר נקודות התקציר חייב להיות בין 1-100")

    return StreamingResponse(
        generate_streaming_summary(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# נקודת כניסה לריצה מקומית
if __name__ == "__main__":
    import uvicorn

    print("🚀 מפעיל שירות FastAPI על פורט 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)