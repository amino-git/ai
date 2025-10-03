# app.py
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="السيرة الذاتية - أمين خالد", page_icon="💼", layout="wide")

# ---------- إعداد الوضع (نهاري/ليلي) ----------
theme = st.sidebar.radio("اختر الوضع:", ["🌞 نهاري", "🌙 ليلي"])

if theme == "🌙 ليلي":
    BG = "#0f0f0f"
    TEXT = "#eaeaea"
    CARD = "#1b1b1b"
    ACCENT = "#bdbdbd"
else:
    BG = "#f5f5f5"
    TEXT = "#0b0b0b"
    CARD = "#ffffff"
    ACCENT = "#444444"

# ---------- CSS وتصميم ----------
st.markdown(f"""
    <style>
        :root {{
            --bg: {BG};
            --text: {TEXT};
            --card: {CARD};
            --accent: {ACCENT};
        }}
        html, body, .reportview-container .main {{
            background-color: var(--bg);
            color: var(--text);
        }}
        .cv-container {{
            max-width: 1100px;
            margin: 20px auto;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        }}
        .card {{
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.02));
            background-color: var(--card);
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            margin-bottom: 18px;
        }}
        .profile-pic {{
            width: 160px;
            height: 160px;
            border-radius: 50%;
            border: 6px solid rgba(0,0,0,0.06);
            object-fit: cover;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }}
        .name {{
            font-size: 28px;
            font-weight: 700;
            text-align: center;
            margin-top: 8px;
            color: var(--text);
        }}
        .subtitle {{
            text-align: center;
            color: var(--accent);
            margin-bottom: 6px;
        }}
        .contact a {{
            color: var(--text);
            text-decoration: none;
        }}
        .skill-bar {{
            height: 10px;
            border-radius: 8px;
            background: rgba(0,0,0,0.06);
            overflow: hidden;
        }}
        .skill-fill {{
            height: 100%;
            border-radius: 8px;
            background: linear-gradient(90deg, #666, #222);
        }}
        .project-title {{
            font-weight: 600;
            margin-bottom: 6px;
        }}
        .muted {{ color: var(--accent); font-size:13px; }}
        .print-btn {{
            display:flex; justify-content:center; margin-top:8px;
        }}
        /* responsive */
        @media (max-width: 768px) {{
            .profile-pic {{ width: 130px; height:130px; }}
        }}
    </style>
""", unsafe_allow_html=True)

# ---------- البيانات الوهمية (عدِّلها إذا حبيت) ----------
profile = {
    "name": "أمين خالد الجبري",
    "title": "طالب تقنية معلومات • مطور ويب • مهتم بالذكاء الاصطناعي",
    "location": "اليمن",
    "email": "amin.khaled.ali@gmail.com",
    "phone": "+967 77X XXX XXX",
    "website": "https://amin.kesug.com",
    "linkedin": "https://linkedin.com/in/yourusername",
    "github": "https://github.com/yourusername",
    # صورة بروفايل تمثلية - غيّرها للرابط حقيقتك
    "profile_image": "https://i.pravatar.cc/300?img=12",
    "summary": (
        "طالب في السنة الثالثة بتقنية المعلومات، مهتم بتطوير الويب، قواعد البيانات، وتعلم الآلة. "
        "عملت على مشاريع ويب باستخدام PHP وMySQL وPython، وشاركت في تصميم أنظمة بسيطة وتطبيقات تفاعلية."
    )
}

skills = {
    "Python": 88,
    "HTML & CSS": 92,
    "JavaScript": 72,
    "MySQL / SQL": 78,
    "Machine Learning": 66,
    "Streamlit": 85,
    "Bootstrap": 88
}

experiences = [
    {
        "title": "مطور ويب (متدرب)",
        "company": "مشروع جامعي - Workaway Clone",
        "period": "يونيو 2024 - سبتمبر 2024",
        "desc": "بناء نظام تسجيل للمستخدمين، بروفايلات، ونظام مراسلة بسيط باستخدام PHP وMySQL وAJAX."
    },
    {
        "title": "مساعد في محل بقالة (دوام جزئي)",
        "company": "بقالة المدينة",
        "period": "منذ 2025-03",
        "desc": "إدارة الطلبات وخدمة العملاء، ونظم جدولة التوريد."
    }
]

education = [
    {
        "degree": "بكالوريوس تقنية المعلومات (قيد الدراسة)",
        "school": "جامعة محلية",
        "period": "2022 - الآن",
        "desc": "المواد الأساسية: هياكل البيانات، شبكات، قواعد بيانات، نظم تشغيل."
    },
]

projects = [
    {
        "name": "Workaway Clone",
        "desc": "موقع للتواصل بين العمال والمضيفين مع لوحة تحكم ونظام ملفات شخصية.",
        "link": "https://github.com/yourusername/workaway-clone"
    },
    {
        "name": "نظام دعاية ذكي",
        "desc": "نظام تجريبي لتحليل النصوص وتصنيف المشاعر باستخدام نموذج ML مبسط.",
        "link": "https://github.com/yourusername/ai-project"
    },
    {
        "name": "ألعاب Streamlit",
        "desc": "مجموعة ألعاب تفاعلية صغيرة (تخمين رقم، تيك تاك تو، hangman).",
        "link": "https://yourgame.streamlit.app"
    }
]

certificates = [
    {"name": "دورة Python للمبتدئين", "from": "منصة Coursera", "year": "2023"},
    {"name": "مقدمة في ML", "from": "منصة Udemy", "year": "2024"}
]

languages = [
    {"lang": "العربية", "level": "اللغة الأم"},
    {"lang": "الإنجليزية", "level": "متوسط - جيد"}
]

# ---------- محتوى الصفحة ----------
st.markdown("<div class='cv-container'>", unsafe_allow_html=True)

# Header: صورة + اسم + أزرار
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        # صورة دائرية
        st.markdown(f"<img class='profile-pic' src='{profile['profile_image']}' alt='profile' />", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='name'>{profile['name']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='subtitle'>{profile['title']}</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='muted' style='text-align:center'>{profile['location']} • تحديث: {datetime.now().year}</p>", unsafe_allow_html=True)
        # روابط تفاعلية
        st.markdown(
            f"""
            <div style='text-align:center; margin-top:8px;'>
                <a href="{profile['github']}" target="_blank">🔗 GitHub</a> &nbsp;&nbsp; | &nbsp;&nbsp;
                <a href="{profile['linkedin']}" target="_blank">🔗 LinkedIn</a> &nbsp;&nbsp; | &nbsp;&nbsp;
                <a href="mailto:{profile['email']}">✉️ Email</a> &nbsp;&nbsp; | &nbsp;&nbsp;
                <a href="{profile['website']}" target="_blank">🌐 Website</a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        # زر طباعة (JS)
        st.markdown("""
            <div class="print-btn">
                <button onclick="window.print()" style="padding:10px 14px; border-radius:8px; border:none; cursor:pointer;">
                    🖨️ طباعة / حفظ PDF
                </button>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# صف رئيسي: عمود معلومات وعمود محتوى
left_col, right_col = st.columns([1.2, 2.2])

# ----- العمود الأيسر (معلومات سريعة وبطاقات) -----
with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 👤 النبذة")
    st.write(profile["summary"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📞 تواصل")
    st.write(f"- الموقع: [{profile['website']}]({profile['website']})")
    st.write(f"- البريد: {profile['email']}")
    st.write(f"- الهاتف: {profile['phone']}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🛠️ المهارات")
    # عرض المهارات مع progress bars مخصصة
    for s, v in skills.items():
        st.markdown(f"<div style='margin-bottom:6px;'><b>{s}</b> <span class='muted'> {v}%</span></div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='skill-bar'>
                <div class='skill-fill' style='width:{v}%; background: linear-gradient(90deg, rgba(50,50,50,0.9), rgba(30,30,30,0.9));'></div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🗣️ اللغات")
    for lg in languages:
        st.write(f"- **{lg['lang']}** — {lg['level']}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🏆 الشهادات")
    for c in certificates:
        st.write(f"- **{c['name']}** — {c['from']} ({c['year']})")
    st.markdown("</div>", unsafe_allow_html=True)

# ----- العمود الأيمن (الخبرات، التعليم، المشاريع) -----
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 💼 الخبرات العملية")
    for e in experiences:
        st.markdown(f"**{e['title']}** — *{e['company']}*")
        st.markdown(f"<span class='muted'>{e['period']}</span>", unsafe_allow_html=True)
        st.write(e['desc'])
        st.markdown("---")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎓 التعليم")
    for ed in education:
        st.markdown(f"**{ed['degree']}** — *{ed['school']}*")
        st.markdown(f"<span class='muted'>{ed['period']}</span>", unsafe_allow_html=True)
        st.write(ed['desc'])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📂 المشاريع البارزة")
    # بطاقات مشاريع
    for p in projects:
        st.markdown(f"<div style='padding:10px 0;'>")
        st.markdown(f"<div class='project-title'>{p['name']}</div>", unsafe_allow_html=True)
        st.write(p['desc'])
        st.markdown(f"[عرض المشروع / الكود]({p['link']})", unsafe_allow_html=True)
        st.markdown("</div>")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📈 ملخص احترافي")
    st.write("- مهارات تقنية قوية مع قدرة على التعلم السريع.")
    st.write("- تواصل فعّال والعمل ضمن فريق.")
    st.write("- شغف بتطوير المنتجات وحل المشاكل بطريقة عملية.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- قسم التذييل: روابط + تحميل PDF ----------
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🔗 روابط سريعة")
st.markdown(f"- GitHub: [{profile['github']}]({profile['github']})")
st.markdown(f"- LinkedIn: [{profile['linkedin']}]({profile['linkedin']})")
st.markdown(f"- الموقع: [{profile['website']}]({profile['website']})")
st.markdown("")

# زر تحميل PDF — طريقتين:
# 1) إذا رفعت ملف PDF في نفس مجلد المشروع باسم 'CV_Ameen.pdf' فعّل السطر التالي واستخدمه.
# 2) أو ضع رابط خارجي مباشر في pdf_url (مثلاً رابط Raw في GitHub أو Google Drive public link).
pdf_url = ""  # <-- حط هنا رابط PDF المباشر لو عندك (مثلاً: https://raw.githubusercontent.com/username/repo/main/CV_Ameen.pdf)

if pdf_url:
    st.markdown(f"[📥 تحميل السيرة الذاتية PDF]({pdf_url})", unsafe_allow_html=True)
else:
    st.markdown("📥 لتحميل PDF: ارفع ملف `CV_Ameen.pdf` في مستودع المشروع أو ضع رابط مباشر في المتغير `pdf_url` داخل الكود.", unsafe_allow_html=True)
    # تعليمة لتحميل محلي لو رفعته في المشروع - فعّل السطر التالي لو حطيت الملف في نفس المجلد:
    # with open("CV_Ameen.pdf", "rb") as f:
    #     st.download_button("تحميل PDF (مباشر)", data=f, file_name="CV_Ameen.pdf", mime="application/pdf")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)  # نهاية cv-container
