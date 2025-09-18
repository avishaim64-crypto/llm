import threading
import json
import time
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import requests


SERVICE_URL = "http://localhost:8000/summarize"


DEFAULT_HEBREW_TEXT = (
    """
    ירושלים היא אחת הערים העתיקות והמשמעותיות בעולם, ועומדת במרכז ההיסטוריה הדתית, התרבותית והפוליטית של המזרח התיכון כבר אלפי שנים. ראשיתה כיישוב מתוארכת לתקופת הברונזה המוקדמת, בסביבות האלף הרביעי לפנה״ס. במקרא נזכרת ירושלים כעיר היבוסים, עד לכיבושה בידי דוד המלך במאה ה־10 לפנה״ס, אז הפכה לבירת ממלכת ישראל המאוחדת. שלמה, בנו של דוד, בנה את בית המקדש הראשון, שהיווה מוקד פולחני ורוחני מרכזי לעם היהודי.

    לאורך הדורות, ירושלים נכבשה פעמים רבות. בשנת 586 לפנה״ס חרב בית המקדש הראשון בידי נבוכדנאצר מלך בבל, והעיר עברה תחת שלטון בבלי ולאחר מכן פרסי. עם כיבושי אלכסנדר מוקדון במאה ה־4 לפנה״ס נכנסה תחת השפעה הלניסטית. בשנת 63 לפנה״ס נכבשה בידי פומפיוס הרומי, ולאחר מכן הייתה לחלק מהאימפריה הרומית. בשנת 70 לספירה חרב בית המקדש השני במרד הגדול, אירוע שעיצב את זהות העם היהודי לדורות. בהמשך, במאה ה־2 לספירה, נבנתה "אליה קפיטולינה" על חורבות העיר, ונאסרה ישיבת יהודים בה.

    עם התפשטות האסלאם במאה ה־7 נכבשה ירושלים בידי הח׳ליף עומר, ונבנו בה כיפת הסלע ומסגד אל־אקצא, שהפכו את העיר לאתר קדוש גם למוסלמים. במאה ה־11 נכבשה בידי הצלבנים, ששלטו בה עד חזרת צלאח א־דין בשנת 1187. בתקופה העות׳מאנית, שהחלה במאה ה־16, נבנו חומות העיר המוכרות לנו כיום.

    במאה ה־19 החלה ירושלים להתפתח מחדש כמרכז בינלאומי, עם קהילות נוצריות, מוסלמיות ויהודיות שגדלו מחוץ לחומות. בתקופת המנדט הבריטי (1917–1948) הפכה למוקד מתיחות בין יהודים לערבים. עם קום מדינת ישראל נקבעה כבירתה, אף כי העיר חולקה עד 1967, אז אוחדה לאחר מלחמת ששת הימים.

    כיום ירושלים היא עיר מודרנית ותוססת, המשלבת היסטוריה עתיקה עם חיים עכשוויים, ומשמשת מוקד עלייה לרגל לשלוש הדתות המונותאיסטיות הגדולות
    """
).strip()


SCENARIOS = [
    {"name": "ברירת מחדל", "params": {}},
    {"name": "מנותק קשר", "params": {"temperature": 1.5, "top_p": 1}},
    {"name": "טמפרטורה גבוהה", "params": {"temperature": 1.0}},
    {"name": "טמפרטורה נמוכה", "params": {"temperature": 0.2}},
    {"name": "top_p גבוה", "params": {"top_p": 1}},
    {"name": "מגבלת טוקנים קטנה", "params": {"max_tokens": 150}},
    # ניתן להוסיף תרחישים נוספים כאן
]


class SummarizerGUI:
    """
    ממשק גרפי (GUI) לתקשורת עם שירות סיכום טקסטים בזמן אמת.
    שולח טקסט בעברית לשרת (באמצעות API מקומי), מציג תרגום,
    ונקודות תקציר באנגלית ובעברית בזמן אמת.
    """
    def __init__(self, root: tk.Tk) -> None:
        """
        אתחול ה-GUI והגדרת משתנים ראשיים.
        :param root: חלון ראשי של tkinter
        """
        self.root = root
        self.root.title("תקצור טקסטים - GUI")
        try:
            self.root.iconbitmap(default=None)
        except Exception:
            pass

        # פרמטרים ברירת מחדל
        self.temperature_var = tk.DoubleVar(value=0.7)  # רמת יצירתיות
        self.top_p_var = tk.DoubleVar(value=0.9)        # גרעין מילים (nucleus sampling)
        self.max_tokens_var = tk.IntVar(value=500)      # מספר טוקנים מקסימלי
        self.scenario_var = tk.StringVar(value=SCENARIOS[0]["name"])

        # יצירת כל רכיבי הממשק
        self._build_widgets()

    def _build_widgets(self) -> None:
        """
        בניית רכיבי הממשק (שדות טקסט, פרמטרים, כפתורים ותיבות פלט).
        כולל הסברים על כל פרמטר.
        """
        container = ttk.Frame(self.root, padding=10)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # שדה טקסט קלט
        ttk.Label(container, text="טקסט בעברית").grid(row=0, column=0, sticky="w")
        self.input_text = tk.Text(container, height=10, wrap="word")
        self.input_text.grid(row=1, column=0, columnspan=6, sticky="nsew", pady=(0, 8))
        self.input_text.insert("1.0", DEFAULT_HEBREW_TEXT)
        self._enable_paste_shortcuts(self.input_text)

        # --- פרמטרים במסגרות נפרדות ---
        param_frame = ttk.Frame(container)
        param_frame.grid(row=2, column=0, columnspan=6, sticky="nsew", pady=(0, 8))

        # Temperature
        temp_container = ttk.Frame(param_frame)
        temp_container.grid(row=0, column=0, padx=4, sticky="w")
        ttk.Label(temp_container, text="temperature").pack(anchor="w")
        self.temperature_var = tk.DoubleVar(value=0.7)
        self.temp_spin = tk.Spinbox(
            temp_container, from_=0.0, to=2.0, increment=0.1,
            textvariable=self.temperature_var, width=8
        )
        self.temp_spin.pack(anchor="w")
        ttk.Label(temp_container, text="רמת יצירתיות/אקראיות במודל: נמוך=צפוי, גבוה=יצירתי", foreground="gray").pack(
            anchor="w")

        # Top-p
        top_p_container = ttk.Frame(param_frame)
        top_p_container.grid(row=0, column=1, padx=4, sticky="w")
        ttk.Label(top_p_container, text="top_p").pack(anchor="w")
        self.top_p_var = tk.DoubleVar(value=0.9)
        self.top_p_spin = tk.Spinbox(
            top_p_container, from_=0.0, to=1.0, increment=0.05,
            textvariable=self.top_p_var, width=8
        )
        self.top_p_spin.pack(anchor="w")
        ttk.Label(top_p_container, text="ככל שהערך גבוה יותר, המודל בוחר מתוך מגוון רחב יותר של מילים; ככל שהערך נמוך, הוא מוגבל למילים הסבירות ביותר.",
                  foreground="gray").pack(anchor="w")

        # Max tokens
        max_tokens_container = ttk.Frame(param_frame)
        max_tokens_container.grid(row=0, column=2, padx=4, sticky="w")
        ttk.Label(max_tokens_container, text="max_tokens").pack(anchor="w")
        self.max_tokens_var = tk.IntVar(value=500)
        self.max_tokens_spin = tk.Spinbox(
            max_tokens_container, from_=1, to=8192, increment=10,
            textvariable=self.max_tokens_var, width=8
        )
        self.max_tokens_spin.pack(anchor="w")
        ttk.Label(max_tokens_container, text="מספר טוקנים מקסימלי שהמודל יפיק", foreground="gray").pack(anchor="w")



        # בחירת תרחיש
        ttk.Label(container, text="תרחיש").grid(row=3, column=0, sticky="w", pady=(8, 0))
        scenario_names = [s["name"] for s in SCENARIOS]
        self.scenario_menu = ttk.OptionMenu(
            container,
            self.scenario_var,
            scenario_names[0],
            *scenario_names,
            command=self._on_scenario_change,
        )
        self.scenario_menu.grid(row=3, column=1, sticky="w", pady=(8, 0))

        # כפתור שליחה
        self.send_button = ttk.Button(container, text="שליחה", command=self._on_send_clicked)
        self.send_button.grid(row=3, column=5, sticky="e", pady=(8, 0))

        # פלט: תרגום לאנגלית (מלא)
        ttk.Label(container, text="תרגום לאנגלית (שלב ראשון)").grid(row=4, column=0, sticky="w", pady=(12, 0))
        self.english_text = tk.Text(container, height=6, wrap="word", state="disabled")
        self.english_text.grid(row=5, column=0, columnspan=6, sticky="nsew")

        # פלט: נקודות תקציר באנגלית (למעלה)
        ttk.Label(container, text="נקודות תקציר (אנגלית)").grid(row=6, column=0, sticky="w", pady=(12, 0))
        self.summary_text = tk.Text(container, height=8, wrap="word")
        self.summary_text.grid(row=7, column=0, columnspan=6, sticky="nsew")

        # פלט: נקודות/תרגום לעברית (למטה)
        ttk.Label(container, text="נקודות תקציר (עברית) / תרגומים").grid(row=8, column=0, sticky="w", pady=(12, 0))
        self.translation_text = tk.Text(container, height=8, wrap="word", state="disabled")
        self.translation_text.grid(row=9, column=0, columnspan=6, sticky="nsew")

        # גדלי עמודות/שורות
        for col in range(6):
            container.columnconfigure(col, weight=1)
        container.rowconfigure(1, weight=1)
        container.rowconfigure(5, weight=1)
        container.rowconfigure(7, weight=2)
        container.rowconfigure(9, weight=1)

    def _enable_paste_shortcuts(self, widget: tk.Text) -> None:
        """
        הפעלת קיצורי דרך להדבקה (Ctrl+V, Shift+Insert).
        """
        widget.bind("<Control-v>", lambda e: (widget.event_generate("<<Paste>>"), "break"))
        widget.bind("<Control-V>", lambda e: (widget.event_generate("<<Paste>>"), "break"))
        widget.bind("<Shift-Insert>", lambda e: (widget.event_generate("<<Paste>>"), "break"))

    def _on_scenario_change(self, selected_name: str) -> None:
        """
       עדכון ערכי הפרמטרים (temperature, top_p, max_tokens וכו׳)
       לפי תרחיש שנבחר מתפריט ה־dropdown.
       """
        for s in SCENARIOS:
            if s["name"] == selected_name:
                params = s.get("params", {})
                if "temperature" in params:
                    self.temperature_var.set(params["temperature"])
                else:
                    self.temperature_var.set(0.7)
                if "top_p" in params:
                    self.top_p_var.set(params["top_p"])
                else:
                    self.top_p_var.set(0.9)
                if "max_tokens" in params:
                    self.max_tokens_var.set(params["max_tokens"])
                else:
                    self.max_tokens_var.set(500)

                break

    def _append_summary(self, text: str) -> None:
        """
        הוספת טקסט לחלון התקציר באנגלית.
        :param text: טקסט שיוצג למשתמש
        """
        self.summary_text.insert("end", text + "\n")
        self.summary_text.see("end")

    def _append_translation(self, text: str) -> None:
        """
        הוספת טקסט לחלון התרגום לעברית.
        :param text: טקסט שיוצג למשתמש
        """
        self.translation_text.configure(state="normal")
        self.translation_text.insert("end", text + "\n")
        self.translation_text.configure(state="disabled")
        self.translation_text.see("end")

    def _set_english_text(self, text: str) -> None:
        """
        הצגת תרגום מלא לאנגלית (פלט ראשוני מהמודל).
        """
        self.english_text.configure(state="normal")
        self.english_text.delete("1.0", "end")
        self.english_text.insert("end", text)
        self.english_text.configure(state="disabled")
        self.english_text.see("end")

    def _clear_outputs(self) -> None:
        """
        איפוס כל חלונות הפלט (תקציר, תרגום, אנגלית).
        """
        self.translation_text.configure(state="normal")
        self.translation_text.delete("1.0", "end")
        self.translation_text.configure(state="disabled")
        self.summary_text.delete("1.0", "end")
        self.english_text.configure(state="normal")
        self.english_text.delete("1.0", "end")
        self.english_text.configure(state="disabled")

    def _on_send_clicked(self) -> None:
        """
        אירוע לחיצה על כפתור "שליחה":
        1. קריאת הטקסט מהמשתמש
        2. בניית payload (פרמטרים שנשלחים לשרת)
        3. הפעלת thread להרצת בקשה בזמן אמת
        """
        text_value = self.input_text.get("1.0", "end").strip()
        if not text_value:
            messagebox.showwarning("אזהרה", "נא להזין טקסט בעברית")
            return

        # כאן בונים את הבקשה לשרת – כולל פרמטרים שהמשתמש בחר
        payload = {
            "text": text_value,
            "max_summary_points": 5,
            "temperature": float(self.temperature_var.get()),
            "top_p": float(self.top_p_var.get()),
            "max_tokens": int(self.max_tokens_var.get()),
        }

        # איפוס פלט והתחלת סטרימינג
        self._clear_outputs()
        self.send_button.configure(state="disabled")
        self._append_summary(f"(זמן: {time.strftime('%H:%M:%S', time.localtime())})")
        self._append_summary("מקבל תגובות בזמן אמת:\n" + ("=" * 20))

        thread = threading.Thread(target=self._stream_request, args=(payload,), daemon=True)
        thread.start()

    def _stream_request(self, payload: dict) -> None:
        """
        שליחת בקשת POST לשירות הסיכום (API).
        התשובה מתקבלת כזרם (Server-Sent Events, SSE).
        הפונקציה קוראת צ'אנקים בזמן אמת ומעדכנת את ה-GUI.
        """
        try:
            response = requests.post(
                SERVICE_URL,
                json=payload,
                stream=True,  # מצב סטרימינג (לא ממתינים לסיום מלא)
                headers={
                    "Accept": "text/event-stream",  # מבקשים SSE
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
                timeout=120,
            )

            # קוד סטטוס לא תקין
            if response.status_code != 200:
                self.root.after(0, lambda: self._append_summary(f"שגיאה בשירות: {response.status_code}"))
                self.root.after(0, lambda: self._append_summary(response.text))
                return

            buffer = ""
            completed = False

            # קריאת הזרם בצ'אנקים
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if not chunk:
                    continue
                buffer += chunk

                # עיבוד הודעות SSE לפי "\n\n"
                while "\n\n" in buffer:
                    part, buffer = buffer.split("\n\n", 1)
                    for line in part.splitlines():
                        if not line.startswith("data: "):
                            continue
                        data_json = line[6:]
                        try:
                            data = json.loads(data_json)

                            # 🔹 מציג תרגום מלא לאנגלית
                            if "english_text" in data:
                                et = data.get("english_text", "")
                                self.root.after(0, lambda t=et: self._set_english_text(t))

                            # 🔹 סטטוס ביניים (למשל: "מתרגם", "מסכם"...)
                            if "status" in data and data.get("status") != "completed":
                                self.root.after(0, lambda s=data["status"]: self._append_summary(f"סטטוס: {s}"))

                            # 🔹 נקודת תקציר באנגלית
                            elif "summary_point" in data:
                                point = data["summary_point"]
                                point_number = point.get("point_number")
                                content = point.get("content", "")
                                self.root.after(0, lambda n=point_number, c=content: self._append_summary(f"\nנקודה {n}: {c}"))

                            # 🔹 תרגום נקודת תקציר לעברית
                            elif "summary_point_hebrew" in data:
                                final_he = data["summary_point_hebrew"]
                                pn = final_he.get("point_number")
                                ct = final_he.get("content", "")
                                self.root.after(0, lambda n=pn, c=ct: self._append_translation(f"תרגום לנקודה {n}: {c}"))

                            # 🔹 סיום
                            if data.get("status") == "completed":
                                total = data.get("total_points", 0)
                                self.root.after(0, lambda t=total: self._append_summary(f"\nהושלם! סך הכל {t} נקודות"))
                                completed = True
                                break

                            # 🔹 טיפול בשגיאות
                            if "error" in data:
                                err = data["error"]
                                self.root.after(0, lambda e=err: self._append_summary(f"שגיאה: {e}"))
                                completed = True
                                break

                        except json.JSONDecodeError as e:
                            self.root.after(0, lambda ex=e: self._append_summary(f"שגיאה בפענוח JSON: {ex}"))
                            continue
                if completed:
                    break
        except requests.exceptions.RequestException as e:
            self.root.after(0, lambda ex=e: self._append_summary(f"שגיאת חיבור: {ex}"))
        finally:
            # משחררים את כפתור השליחה חזרה לשימוש
            self.root.after(0, lambda: self.send_button.configure(state="normal"))

def main() -> None:
    root = tk.Tk()
    app = SummarizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()


