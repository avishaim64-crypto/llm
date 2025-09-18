#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
הדגמת ההבדל בין return ל-yield
"""
import time


# 1. פונקציה רגילה עם return
def regular_function():
    results = []
    for i in range(5):
        print(f"מעבד פריט {i + 1}...")
        time.sleep(1)  # סימולציה של עבודה ארוכה
        results.append(f"תוצאה {i + 1}")

    return results  # מחזיר הכל בבת אחת בסוף


# 2. Generator עם yield
def generator_function():
    for i in range(5):
        print(f"מעבד פריט {i + 1}...")
        time.sleep(1)  # סימולציה של עבודה ארוכה
        yield f"תוצאה {i + 1}"  # מחזיר פריט אחד ו"משהה" את הפונקציה
        # הפונקציה ממשיכה מכאן בקריאה הבאה


# 3. השוואה בין השניים
def demonstrate_difference():
    print("=== פונקציה רגילה עם return ===")
    start = time.time()

    # עם return - נחכה עד שהכל יסתיים
    results = regular_function()

    end = time.time()
    print(f"קיבלתי {len(results)} תוצאות אחרי {end - start:.1f} שניות")
    for result in results:
        print(f"  - {result}")

    print("\n" + "=" * 50)
    print("=== Generator עם yield ===")

    # עם yield - נקבל תוצאות אחת אחרי השנייה
    start = time.time()
    for i, result in enumerate(generator_function(), 1):
        elapsed = time.time() - start
        print(f"קיבלתי תוצאה {i} אחרי {elapsed:.1f} שניות: {result}")


# 4. דוגמה למה שקורה ב-FastAPI שלנו
def streaming_summary_demo():
    """דמיון למה שקורה בשירות שלנו"""

    def simulate_our_service():
        # שלב 1: תרגום
        yield '{"status": "מתרגם טקסט..."}'
        time.sleep(2)  # סימולציה של תרגום
        yield '{"status": "תרגום הושלם"}'

        # שלב 2: יצירת תקציר
        yield '{"status": "יוצר תקציר..."}'
        time.sleep(3)  # סימולציה של יצירת תקציר

        # שלב 3: שליחת נקודות אחת אחת
        points = [
            "ישראל מובילה בתחום הטכנולוגיה",
            "חברות רבות משקיעות בבינה מלאכותית",
            "יש שיתופי פעולה בינלאומיים",
            "המדינה נחשבת חדשנית",
            "השקעות גדולות בתחום"
        ]

        for i, point in enumerate(points, 1):
            time.sleep(1)  # מעט השהיה בין נקודות
            yield f'{{"summary_point": {{"point_number": {i}, "content": "{point}"}}}}'

        yield '{"status": "completed"}'

    print("🚀 מדמה את השירות שלנו:")
    print("-" * 40)

    for message in simulate_our_service():
        current_time = time.strftime("%H:%M:%S")
        print(f"[{current_time}] {message}")


# 5. יתרונות של yield
def explain_yield_benefits():
    print("🔥 יתרונות של yield:")
    print("1. זיכרון - לא צריך לשמור הכל בזיכרון בו-זמנית")
    print("2. זמן תגובה - המשתמש מקבל תוצאות מיד")
    print("3. חוויית משתמש - רואה התקדמות במקום המתנה")
    print("4. יעילות - אפשר לעצור באמצע אם צריך")

    print("\n📊 דוגמה לחיסכון בזיכרון:")

    # גרסה שצורכת הרבה זיכרון
    def memory_heavy():
        big_list = [i ** 2 for i in range(1000000)]  # יוצר רשימה של מיליון איברים
        return big_list

    # גרסה חסכונית עם yield
    def memory_efficient():
        for i in range(1000000):
            yield i ** 2  # מחשב רק איבר אחד בכל פעם

    print("❌ memory_heavy() - שומר מיליון מספרים בזיכרון")
    print("✅ memory_efficient() - מחשב מספר אחד בכל פעם")


if __name__ == "__main__":
    # demonstrate_difference()
    # print("\n" + "="*60 + "\n")
    streaming_summary_demo()
    print("\n" + "=" * 60 + "\n")
    explain_yield_benefits()