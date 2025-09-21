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
    """מודל לבקשת תקצור טקסט"""
    text: str  # הטקסט לתקצור
    max_points: int = 5  # מספר נקודות התקציר (1-10)
    temperature: float = 0.7  # רמת יצירתיות המודל (0.1-1.0): נמוך = יותר פרדיקטיבילי, גבוה = יותר יצירתי
    top_p: float = 0.9  # סף הסתברות לבחירת מילים (0.1-1.0): נמוך = מיקוד במילים הסבירות ביותר
    max_tokens: int = 500  # מספר מקסימלי של טוקנים (מילים בקירוב) לכל נקודה


class SummaryPoint(BaseModel):
    """מודל לנקודת תקציר בודדת"""
    point_number: int  # מספר הנקודה (1, 2, 3...)
    content: str  # תוכן הנקודה באנגלית
    timestamp: float  # זמן יצירת הנקודה (Unix timestamp)


def initialize_nllb():
    """
    אתחול מודל NLLB לתרגום

    טוען את המודל NLLB-200 (No Language Left Behind) של Meta
    שתומך ב-200+ שפות כולל עברית ← → אנגלית
    """
    global nllb_tokenizer, nllb_model

    if nllb_tokenizer is None or nllb_model is None:
        print("🔄 טוען מודל NLLB...")
        model_name = "facebook/nllb-200-distilled-600M"
        nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("✅ מודל NLLB נטען בהצלחה")


def translate(text: str, tgt_lang: str, *, stream: bool = False, max_tokens: int = 500):
    """
    תרגום גנרי בעזרת NLLB

    Args:
        text: הטקסט לתרגום
        tgt_lang: קוד שפת היעד (למשל: "eng_Latn" לאנגלית, "heb_Hebr" לעברית)
        stream: האם להחזיר תוצאות בזרימה (True) או בבת אחת (False)
        max_tokens: מספר מקסימלי של טוקנים בתרגום

    Returns:
        - stream=False: מחזיר מחרוזת תרגום מלאה
        - stream=True: מחזיר גנרטור שמפיק חלקי תרגום בזמן אמת

    קודי שפות רלוונטים:
        - עברית: "heb_Hebr"
        - אנגלית: "eng_Latn"
    """
    initialize_nllb()

    if not stream:
        # נתיב תרגום רגיל - מחזיר תוצאה מלאה
        inputs = nllb_tokenizer(
            text,
            return_tensors="pt",  # PyTorch tensors
            max_length=1024,  # אורך מקסימלי של הקלט
            truncation=True  # קטע טקסט ארוך מדי
        )

        # המרת קוד השפה לID מספרי
        tgt_id = nllb_tokenizer.convert_tokens_to_ids(tgt_lang)
        if tgt_id is None:
            raise ValueError(f"לא נמצא קוד שפה: {tgt_lang}")

        with torch.no_grad():  # בלי חישובים לשימוש עתידי
            generated_tokens = nllb_model.generate(
                **inputs,
                forced_bos_token_id=tgt_id,  # אילוץ שפת היעד
                max_length=1024,  # אורך מקסימלי של הפלט
                num_beams=3,  # כמות נתיבים לחקירה
                early_stopping=True  # עצור כשמגיעים לסוף הגיוני
            )

        translation = nllb_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation.strip()

    # נתיב סטרימינג - מחזיר גנרטור לתרגום בזמן אמת
    def _stream_generator():
        """גנרטור פנימי לסטרימינג תרגום"""
        inputs = nllb_tokenizer(
            text,
            return_tensors="pt",
            max_length=max_tokens,
            truncation=True
        )

        model_device = next(nllb_model.parameters()).device
        input_ids = inputs["input_ids"][:1].to(model_device)#המספרים שמייצגים את התוכן
        attention_mask = inputs["attention_mask"][:1].to(model_device) #תוכן אמיתי

        # בדיקת תקינות קוד השפה
        tgt_id = nllb_tokenizer.convert_tokens_to_ids(tgt_lang)
        if tgt_id is None:
            raise ValueError(f"לא נמצא קוד שפה: {tgt_lang}")

        # הגדרת TextIteratorStreamer לקבלת טוקנים בזמן אמת
        streamer = TextIteratorStreamer(
            nllb_tokenizer,
            skip_special_tokens=True,  # לא להציג טוקנים טכניים
            skip_prompt=True  # לא להציג את הקלט המקורי
        )

        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            forced_bos_token_id=tgt_id,
            max_new_tokens=max_tokens,  # מספר טוקנים חדשים מקסימלי
            do_sample=False,  # דטרמיניסטי (לא רנדומלי)
            num_beams=1,  # ללא beam search לסטרימינג מהיר
            streamer=streamer
        )

        def _run_generate():
            """פונקציה להרצת הגנרציה בתהליכון נפרד"""
            with torch.no_grad():
                nllb_model.generate(**generation_kwargs)

        # הרצה בתהליכון נפרד כדי לא לחסום את הסטרימינג
        thread = threading.Thread(target=_run_generate)
        thread.start()

        # זרימת הטוקנים החדשים ברגע שהם נוצרים
        for new_text in streamer:
            yield new_text

    return _stream_generator()


def check_ollama_connection() -> bool:
    """
    בדיקת חיבור ל-Ollama

    Returns:
        True אם Ollama זמין, False אחרת
    """
    try:
        ollama.list()  # בדיקה פשוטה של זמינות השירות
        return True
    except Exception:
        return False


def ensure_phi3_model() -> bool:
    """
    וידוא שמודל Phi-3 זמין ב-Ollama

    Returns:
        True אם המודל זמין או הורד בהצלחה, False אחרת

    Note:
        Phi-3-mini הוא מודל שפה קטן ויעיל של Microsoft
        מתאים לתקצורים ומשימות NLP בסיסיות
    """
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


async def stream_summary_points_with_phi3(english_text: str, num_points: int = 5, *, temperature: float = 0.7,
                                          top_p: float = 0.9, max_tokens: int = 500):
    """
    זרימת נקודות תקציר ישירות מ-Ollama (Phi-3) בזמן אמת

    Args:
        english_text: הטקסט באנגלית לתקצור
        num_points: מספר נקודות התקציר הרצוי
        temperature: רמת יצירתיות (0.1-1.0)
            - 0.1-0.3: מאוד שמרני ופרדיקטיבילי
            - 0.4-0.7: מאוזן (מומלץ)
            - 0.8-2.0: יצירתי ובלתי צפוי
        top_p: הסתברות מצטברת לבחירת מילים (0.1-1.0)
            - 0.1-0.5: מיקוד במילים הסבירות ביותר
            - 0.6-0.9: מאוזן (מומלץ)
            - 0.95-1.0: שיקול של כל המילים האפשריות
        max_tokens: מספר מקסימלי של טוקנים לכל נקודה

    Yields:
        tuples של (מספר_נקודה, תוכן_הנקודה) עבור כל נקודה שהושלמה

    Note:
        הפונקציה מזהה התחלה וסיום של כל נקודה ממוספרת בזמן אמת
        ומשיבה כל נקודה מיד כשהיא מוכנה, ללא המתנה לסיום כל הטקסט
    """
    # בניית prompt מובנה למודל
    prompt = f"""Please create a summary of the following text in exactly {num_points} numbered bullet points.
Each point must be a single line, start with an explicit number like "1. ", "2. ", etc., and be at most {max_tokens} characters long.
Do not add any prose before or after the list. Only output the numbered list.

Text to summarize:
{english_text}

Summary:"""

    # משתנים לזיהוי גבולות נקודות
    expected_index = 1  # הנקודה הבאה שאנו מצפים לה
    buffer = ""  # אגירת טקסט עד שנקודה מושלמת

    try:
        # יצירת queue אסינכרוני להעברת נתונים בין תהליכונים
        queue: asyncio.Queue = asyncio.Queue(maxsize=0)

        def producer():
            """
            פונקצית producer לקבלת streaming response מ-Ollama
            רצה בתהליכון נפרד כדי לא לחסום את האירוע הראשי
            """
            try:
                stream = ollama.chat(
                    model="phi3:mini",
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        'temperature': float(temperature),  # המרה מפורשת לבטיחות
                        'top_p': float(top_p),
                        'max_tokens': int(max_tokens)
                    },
                    stream=True  # הפעלת streaming mode
                )

                # העברת כל chunk לqueue
                for chunk in stream:
                    try:
                        queue.put_nowait(chunk)
                    except Exception:
                        break

            except Exception as e:
                # שליחת שגיאה לqueue
                try:
                    queue.put_nowait({'error': str(e)})
                except Exception:
                    pass
            finally:
                # סיגנל סיום
                try:
                    queue.put_nowait(None)
                except Exception:
                    pass

        # הפעלת producer בתהליכון נפרד
        threading.Thread(target=producer, daemon=True).start()

        # לולאה ראשית לעיבוד chunks
        while True:
            item = await queue.get()

            # בדיקת סיום או שגיאה
            if item is None:
                break
            if isinstance(item, dict) and 'error' in item:
                raise HTTPException(detail=item['error'])

            # חילוץ הטקסט מה-chunk
            piece = item.get('message', {}).get('content', '')
            if not piece:
                continue

            buffer += piece

            # ניסיון לזהות נקודות שהושלמו
            # אלגוריתם: חפש "N. " ואחריו "N+1. " - מה שביניהם זו נקודה שלמה
            while True:
                start_marker = f"{expected_index}. "
                next_marker = f"{expected_index + 1}. "

                start_pos = buffer.find(start_marker)
                if start_pos == -1:
                    break  # לא מצאנו את ההתחלה של הנקודה הנוכחית

                next_pos = buffer.find(next_marker, start_pos + len(start_marker))
                if next_pos == -1:
                    break  # לא מצאנו את ההתחלה של הנקודה הבאה

                # חילוץ תוכן הנקודה (מה שבין ההתחלות)
                point_text = buffer[start_pos + len(start_marker):next_pos]
                point_text = point_text.strip().strip("-•* ")  # ניקוי תווים מיותרים

                if point_text:
                    yield expected_index, point_text
                    expected_index += 1

                # הסרת החלק שעובד מהbuffer
                buffer = buffer[next_pos:]

                if expected_index > num_points:
                    break

            if expected_index > num_points:
                break

        # טיפול בנקודה אחרונה שעלולה להישאר בbuffer
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


async def generate_streaming_summary(request: TextRequest) -> AsyncGenerator[str, None]:
    """
    יצירת תקציר עם streaming response

    הפונקציה מבצעת את התהליך הבא:
    1. בדיקת חיבור ל-Ollama
    2. וידוא זמינות מודל Phi-3
    3. תרגום הטקסט העברי לאנגלית
    4. יצירת תקציר נקודות באנגלית (בסטרימינג)
    5. תרגום כל נקודה חזרה לעברית (בסטרימינג)

    Args:
        request: אובייקט TextRequest עם פרמטרי התקציר

    Yields:
        מחרוזות JSON בפורמט Server-Sent Events
        כל מחרוזת מתחילה ב-"data: " ומסתיימת ב-"\n\n"

    JSON Types שמושבים:
        - {"error": "הודעת שגיאה"}
        - {"english_text": "התרגום המלא לאנגלית"}
        - {"status": "הודעת סטטוס"}
        - {"summary_point": {point_number, content, timestamp}}
        - {"summary_point_hebrew_piece": {point_number, piece}}
        - {"summary_point_hebrew": {point_number, content}}
        - {"status": "completed", "total_points": מספר}
    """
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

        # שלב 3: תרגום לאנגלית (ללא סטרימינג - לקבלת תוצאה מלאה)
        try:
            english_text = translate(request.text, "eng_Latn", stream=False)
        except Exception as e:
            error_msg = {"error": f"שגיאה בתרגום: {str(e)}"}
            yield f"data: {json.dumps(error_msg, ensure_ascii=False)}\n\n"
            return

        # שידור התרגום המלא ללקוח
        yield f"data: {json.dumps({'english_text': english_text}, ensure_ascii=False)}\n\n"

        # שלב 4: יצירת תקציר בסטרימינג
        yield f"data: {json.dumps({'status': 'יוצר תקציר בזמן אמת...'}, ensure_ascii=False)}\n\n"

        sent_points = 0
        try:
            # זרימת נקודות תקציר מ-Phi-3
            async for idx, point in stream_summary_points_with_phi3(
                    english_text,
                    request.max_points,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens
            ):
                # יצירת אובייקט נקודה
                summary_point = SummaryPoint(
                    point_number=idx,
                    content=point,
                    timestamp=time.time()
                )
                sent_points += 1

                # שליחת הנקודה באנגלית
                yield f"data: {json.dumps({'summary_point': summary_point.dict()}, ensure_ascii=False)}\n\n"

                # שלב 5: תרגום הנקודה לעברית בסטרימינג
                heb_text = ""
                try:
                    # תרגום חלק-חלק עם streaming
                    for piece in translate(point, "heb_Hebr", stream=True, max_tokens=request.max_tokens):
                        heb_text += piece
                        # שליחת כל חלק בנפרד
                        yield f"data: {json.dumps({'summary_point_hebrew_piece': {'point_number': idx, 'piece': piece}}, ensure_ascii=False)}\n\n"

                    # שליחת התרגום המלא לעברית
                    yield f"data: {json.dumps({'summary_point_hebrew': {'point_number': idx, 'content': heb_text}}, ensure_ascii=False)}\n\n"

                except Exception as te:
                    # שגיאה בתרגום נקודה ספציפית
                    yield f"data: {json.dumps({'error': f'שגיאה בתרגום נקודה {idx} לעברית: {str(te)}'}, ensure_ascii=False)}\n\n"

                if sent_points >= request.max_points:
                    break

            # הודעת סיום
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
    """
    אתחול השירות

    מופעל אוטומטית כשהשירות עולה
    טוען את מודלי התרגום מראש לביצועים טובים יותר
    """
    print("🚀 מתחיל שירות תקצור טקסטים")
    print("🔄 טוען מודלי תרגום...")
    initialize_nllb()


@app.get("/health")
async def health_check():
    """
    בדיקת תקינות השירות

    Returns:
        מידע על סטטוס השירות ורכיביו השונים

    Response:
        {
            "status": "healthy/unhealthy",
            "services": {
                "nllb_translation": bool,
                "ollama_phi3": bool
            },
            "timestamp": float
        }
    """
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

    מקבל טקסט בעברית ומחזיר תקציר של עד 10 נקודות.
    כל נקודה נשלחת מיד כשהיא מוכנה, גם באנגלית וגם בתרגום לעברית.

    Args:
        request: TextRequest עם הפרמטרים הבאים:
            - text: הטקסט לתקצור (מינימום 50 תווים)
            - max_points: מספר נקודות (1-10, ברירת מחדל: 5)
            - temperature: רמת יצירתיות (0.1-1.0, ברירת מחדל: 0.7)
            - top_p: סף הסתברות (0.1-1.0, ברירת מחדל: 0.9)
            - max_tokens: מקס טוקנים לנקודה (ברירת מחדל: 500)

    Returns:
        StreamingResponse: זרם של אירועים ב-JSON format

    Raises:
        HTTPException 400: אם הטקסט ריק, קצר מדי, או פרמטרים לא תקינים

    Example:
        POST /summarize
        {
            "text": "טקסט ארוך בעברית...",
            "max_points": 5,
            "temperature": 0.5
        }

    Stream Events:
        - english_text: התרגום המלא לאנגלית
        - status: הודעות סטטוס
        - summary_point: נקודה באנגלית
        - summary_point_hebrew_piece: חלק מתרגום לעברית
        - summary_point_hebrew: תרגום שלם לעברית
        - error: הודעות שגיאה
        - completed: סיום העיבוד עם סיכום
    """

    # בדיקות תקינות הקלט
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="טקסט לא יכול להיות ריק")

    if len(request.text) < 50:
        raise HTTPException(status_code=400, detail="טקסט קצר מדי לתקצור (מינימום 50 תווים)")

    if request.max_points < 1 or request.max_points > 10:
        raise HTTPException(status_code=400, detail="מספר נקודות התקציר חייב להיות בין 1-10")

    if not (0.1 <= request.temperature <= 2.0):
        raise HTTPException(status_code=400, detail="temperature חייב להיות בין 0.1-2.0")

    if not (0.1 <= request.top_p <= 1.0):
        raise HTTPException(status_code=400, detail="top_p חייב להיות בין 0.1-1.0")

    if request.max_tokens < 50 or request.max_tokens > 2000:
        raise HTTPException(status_code=400, detail="max_tokens חייב להיות בין 50-2000")

    return StreamingResponse(
        generate_streaming_summary(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",  # מניעת cache בדפדפן
            "Connection": "keep-alive",  # שמירת החיבור פתוח
            "X-Accel-Buffering": "no"  # מניעת buffering ב-nginx
        }
    )


# נקודת כניסה לריצה מקומית
if __name__ == "__main__":
    import uvicorn

    print("🚀 מפעיל שירות FastAPI על פורט 8000")
    print("📖 תיעוד API זמין ב: http://localhost:8000/docs")
    print("🏥 בדיקת תקינות: http://localhost:8000/health")
    print("🎯 שימוש:")
    print("   POST /summarize עם JSON: {\"text\": \"טקסט בעברית...\"}")
    print("   Response: Server-Sent Events stream")

    uvicorn.run(app, host="0.0.0.0", port=8000)