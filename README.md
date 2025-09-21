# Hebrew Text Summarizer - מערכת תקצור ותרגום טקסטים בעברית

מערכת מתקדמת לתקצור טקסטים בעברית עם תמיכה בזמן אמת (streaming), המשלבת תרגום מכונה וכלי AI מתקדמים.

## 🌟 תכונות עיקריות

- **תקצור אוטומטי**: מקבל טקסט בעברית ומפיק תקציר של עד 10 נקודות מרכזיות
- **תמיכה בזמן אמת**: תוצאות מתקבלות בזמן אמת ללא המתנה לסיום התהליך המלא
- **תרגום דו-כיווני**: עברית ↔ אנגלית באמצעות מודל NLLB של Meta
- **ממשק גרפי ידידותי**: GUI מובנה עם תמיכה בעברית וקיצורי דרך
- **פרמטרים מתקדמים**: שליטה מלאה ברמת היצירתיות והדיוק של התקציר
- **תרחישים מוכנים**: הגדרות מוקדמות לשימושים שונים

## 🏗️ ארכיטקטורה

המערכת מורכבת משני רכיבים עיקריים:

### 1. שירות Backend (`summarizer_service.py`)
- **FastAPI Server**: API RESTful עם תמיכה ב-Server-Sent Events
- **מודל NLLB**: תרגום מתקדם תומך 200+ שפות (Meta)
- **מודל Phi-3**: מודל שפה קטן ויעיל לתקצור (Microsoft)
- **Ollama Integration**: הרצת מודלי AI מקומיים

### 2. ממשק משתמש (`gui.py`)
- **Tkinter GUI**: ממשק גרפי פשוט ויעיל
- **Real-time Updates**: עדכון מיידי של תוצאות
- **Parameter Control**: שליטה בכל פרמטרי המודל
- **Multi-threading**: לא חוסם את הממשק במהלך העיבוד

## 📋 דרישות מערכת

### תוכנות חיצוניות
- **Python 3.8+**
- **Ollama** ([הורדה](https://ollama.ai/))
- **חיבור אינטרנט** (לטעינת מודל NLLB בפעם הראשונה)

### Python Packages
```bash
pip install -r requirements.txt
```

## 🚀 התקנה והרצה

### שלב 1: התקנת Ollama
```bash
# Windows/Mac/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# הפעלת השירות
ollama serve
```

### שלב 2: הכנת הסביבה
```bash
# שכפול הפרויקט
git clone https://github.com/avishaim64-crypto/llm
cd hebrew-text-summarizer

# התקנת dependencies
pip install -r requirements.txt
```

### שלב 3: הפעלת השירות
```bash
# הרצת ה-backend
python summarizer_service.py
```
השירות יעלה על: `http://localhost:8000`

### שלב 4: הפעלת הממשק הגרפי
בטרמינל נפרד:
```bash
# הרצת ה-GUI
python gui.py
```

## 🎯 שימוש

### באמצעות הממשק הגרפי
1. הדביקו טקסט בעברית בשדה הקלט
2. התאימו פרמטרים (או בחרו תרחיש מוכן)
3. לחצו "שליחה"
4. צפו בתוצאות בזמן אמת

### באמצעות API
```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "טקסט בעברית להמחיש...",
    "max_points": 5,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 500
  }'
```

## ⚙️ פרמטרים מתקדמים

### Temperature (0.1-1.0)
- **0.1-0.3**: שמרני ופרדיקטיבילי
- **0.4-0.7**: מאוזן (מומלץ)
- **0.8-1.0**: יצירתי ובלתי צפוי

### Top-p (0.1-1.0)
- **0.1-0.5**: מיקוד במילים הסבירות ביותר
- **0.6-0.9**: מאוזן (מומלץ)
- **0.95-1.0**: שיקול כל המילים האפשריות

### Max Tokens
- מספר מקסימלי של מילים (בקירוב) לכל נקודת תקציר
- טווח מומלץ: 200-800

### Max Points
- מספר נקודות התקציר הרצוי (1-10)

## 🔧 תרחישים מוכנים

המערכת כוללת תרחישים מוקדמים:

- **ברירת מחדל**: הגדרות אופטימליות לרוב השימושים
- **טמפרטורה נמוכה**: תקציר שמרני ומדויק
- **טמפרטורה גבוהה**: תקציר יצירתי וגמיש
- **מגבלת טוקנים קטנה**: תקציר קצר ומרוכז

## 🔍 API Documentation

### בדיקת תקינות
```bash
GET /health
```

### תקצור טקסט
```bash
POST /summarize
Content-Type: application/json

{
  "text": "string",
  "max_points": 5,
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 500
}
```

### תגובות Streaming
התגובה מגיעה בפורמט Server-Sent Events:

```javascript
data: {"english_text": "התרגום המלא לאנגלית"}
data: {"status": "יוצר תקציר בזמן אמת..."}
data: {"summary_point": {"point_number": 1, "content": "...", "timestamp": 1234567890}}
data: {"summary_point_hebrew_piece": {"point_number": 1, "piece": "חלק מהתרגום"}}
data: {"summary_point_hebrew": {"point_number": 1, "content": "תרגום מלא"}}
data: {"status": "completed", "total_points": 5}
```

## 🛠️ פתרון בעיות נפוצות

### Ollama לא מגיב
```bash
# בדיקת סטטוס
ollama list

# הפעלה מחדש
ollama serve
```

### מודל Phi-3 חסר
```bash
# הורדה ידנית
ollama pull phi3:mini
```

### שגיאת זיכרון
- הפחיתו את `max_tokens`
- השתמשו במודל קטן יותר
- סגרו יישומים אחרים

### תרגום איטי
- הראשון פעם לוקח זמן (הורדת NLLB)
- התרגומים הבאים מהירים יותר

## 📁 מבנה הפרויקט

```
hebrew-text-summarizer/
├── summarizer_service.py    # שירות Backend
├── gui.py                  # ממשק גרפי
├── requirements.txt        # תלותות Python
├── README.md              # מדריך זה
└── examples/              # דוגמאות טקסט
    ├── high_temperature.txt
    ├── low_temperature.txt
    └── contextless.txt
```

## 🧪 דוגמאות שימוש

### טמפרטורה נמוכה
### טמפרטורה גבוהה


## 🔒 אבטחה והגבלות

- השירות מיועד לרשת מקומית בלבד (`localhost`)
- אין אימות מובנה - מתאים לפיתוח ושימוש אישי
- נתונים לא נשמרים בשירות

