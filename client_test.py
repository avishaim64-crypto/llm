import requests
import json
import time


def test_summarizer_service(params=None):
    """בדיקת שירות התקצור"""

    url = "http://localhost:8000/summarize"

    # טקסט לדוגמה בעברית
    hebrew_text = """
    ירושלים היא אחת הערים העתיקות והמשמעותיות בעולם, ועומדת במרכז ההיסטוריה הדתית, התרבותית והפוליטית של המזרח התיכון כבר אלפי שנים. ראשיתה כיישוב מתוארכת לתקופת הברונזה המוקדמת, בסביבות האלף הרביעי לפנה״ס. במקרא נזכרת ירושלים כעיר היבוסים, עד לכיבושה בידי דוד המלך במאה ה־10 לפנה״ס, אז הפכה לבירת ממלכת ישראל המאוחדת. שלמה, בנו של דוד, בנה את בית המקדש הראשון, שהיווה מוקד פולחני ורוחני מרכזי לעם היהודי.

לאורך הדורות, ירושלים נכבשה פעמים רבות. בשנת 586 לפנה״ס חרב בית המקדש הראשון בידי נבוכדנאצר מלך בבל, והעיר עברה תחת שלטון בבלי ולאחר מכן פרסי. עם כיבושי אלכסנדר מוקדון במאה ה־4 לפנה״ס נכנסה תחת השפעה הלניסטית. בשנת 63 לפנה״ס נכבשה בידי פומפיוס הרומי, ולאחר מכן הייתה לחלק מהאימפריה הרומית. בשנת 70 לספירה חרב בית המקדש השני במרד הגדול, אירוע שעיצב את זהות העם היהודי לדורות. בהמשך, במאה ה־2 לספירה, נבנתה "אליה קפיטולינה" על חורבות העיר, ונאסרה ישיבת יהודים בה.

עם התפשטות האסלאם במאה ה־7 נכבשה ירושלים בידי הח׳ליף עומר, ונבנו בה כיפת הסלע ומסגד אל־אקצא, שהפכו את העיר לאתר קדוש גם למוסלמים. במאה ה־11 נכבשה בידי הצלבנים, ששלטו בה עד חזרת צלאח א־דין בשנת 1187. בתקופה העות׳מאנית, שהחלה במאה ה־16, נבנו חומות העיר המוכרות לנו כיום.

במאה ה־19 החלה ירושלים להתפתח מחדש כמרכז בינלאומי, עם קהילות נוצריות, מוסלמיות ויהודיות שגדלו מחוץ לחומות. בתקופת המנדט הבריטי (1917–1948) הפכה למוקד מתיחות בין יהודים לערבים. עם קום מדינת ישראל נקבעה כבירתה, אף כי העיר חולקה עד 1967, אז אוחדה לאחר מלחמת ששת הימים.

כיום ירושלים היא עיר מודרנית ותוססת, המשלבת היסטוריה עתיקה עם חיים עכשוויים, ומשמשת מוקד עלייה לרגל לשלוש הדתות המונותאיסטיות הגדולות
    """

    data = {
        "text": hebrew_text,
        "max_summary_points": 5,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 500,
    }

    if params:
        data.update(params)

    print("🚀 שולח בקשה לשירות התקצור...")
    print(f"📝 טקסט מקורי: {hebrew_text[:100]}...")
    print("-" * 60)

    try:
        response = requests.post(
            url,
            json=data,
            stream=True,
            headers={
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            },
            timeout=120
        )

        if response.status_code != 200:
            print(f"❌ שגיאה בשירות: {response.status_code}")
            print(response.text)
            return
        
        

        print(f"(זמן: {time.strftime('%H:%M:%S', time.localtime())})")
        print("📡 מקבל תגובות בזמן אמת:")
        print("=" * 40)

        buffer = ""
        point_counter = 0

        # הגדלנו chunk_size ל־1024 כדי לקרוא בבת אחת יותר תווים
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if not chunk:
                continue

            buffer += chunk

            # נבדוק אם יש לפחות קטע SSE שלם (מופרד ב־\n\n)
            while '\n\n' in buffer:
                part, buffer = buffer.split('\n\n', 1)

                # חלק יכול להכיל כמה שורות, נפרק אותן
                for line in part.splitlines():
                    if not line.startswith('data: '):
                        continue
                    data_json = line[6:]
                    try:
                        data = json.loads(data_json)

                        # סטטוסים כלליים
                        if 'status' in data:
                            print(f"ℹ️  סטטוס: {data['status']}", flush=True)

                        # נקודות תקציר
                        elif 'summary_point' in data:
                            point = data['summary_point']
                            point_counter += 1
                            print(f"\n📋 נקודה {point['point_number']}:", flush=True)
                            print(f"   {point['content']}", flush=True)
                            print(f"   (זמן: {time.strftime('%H:%M:%S', time.localtime(point['timestamp']))})",
                                  flush=True)

                        elif 'summary_point_hebrew' in data:
                            final_he = data['summary_point_hebrew']
                            print('', flush=True)  # סיום שורה
                            print(f"🟦 תרגום לעברית לנקודה {final_he['point_number']}: {final_he['content']}", flush=True)

                        # סיום התקציר
                        elif data.get('status') == 'completed':
                            print(f"\n✅ הושלם! סך הכל {data.get('total_points', 0)} נקודות", flush=True)
                            break

                        # שגיאה
                        elif 'error' in data:
                            print(f"❌ שגיאה: {data['error']}", flush=True)
                            break

                    except json.JSONDecodeError as e:
                        print(f"⚠️  שגיאה בפענוח JSON: {e}", flush=True)
                        continue

        print("\n" + "=" * 60)
        print(f"🏁 סיום - התקבלו {point_counter} נקודות תקציר")

    except requests.exceptions.RequestException as e:
        print(f"❌ שגיאת חיבור: {e}")
        print("💡 ודא שהשירות רץ על http://localhost:8000")


def test_health_check():
    """בדיקת תקינות השירות"""
    print("🏥 בודק תקינות השירות...")

    try:
        response = requests.get("http://localhost:8000/health", timeout=10)

        if response.status_code == 200:
            health_data = response.json()
            print("✅ השירות תקין")
            print(f"📊 סטטוס שירותים:")
            for service, status in health_data.get('services', {}).items():
                emoji = "✅" if status else "❌"
                print(f"   {emoji} {service}: {status}")
        else:
            print(f"❌ השירות לא תקין (סטטוס: {response.status_code})")

    except requests.exceptions.RequestException as e:
        print(f"❌ לא ניתן להתחבר לשירות: {e}")


if __name__ == "__main__":
    print("🧪 מתחיל בדיקת שירות תקצור טקסטים")
    print("=" * 60)

    # תחילה בדיקת תקינות
    test_health_check()
    print()

    # הדגמות שונות של פרמטרים
    scenarios = [
        {"name": "ברירת מחדל", "params": {}},
        {"name": "מנותק קשר", "params": {"temperature": 1.5, "top_p": 1}},
        #{"name": "טמפרטורה גבוהה", "params": {"temperature": 1.0}},
        #{"name": "טמפרטורה נמוכה", "params": {"temperature": 0.2}},
        #{"name": "top_p גבוה", "params": {"top_p": 1}},
        #{"name": "מגבלת טוקנים קטנה", "params": {"max_tokens": 150}},
    ]

    for s in scenarios:
        print("\n" + "-" * 60)
        print(f"🔧 תרחיש: {s['name']}")
        test_summarizer_service(params=s["params"])
