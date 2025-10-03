import streamlit as st
from pathlib import Path

# ---------------- إعداد الصفحة ----------------
st.set_page_config(
    page_title="السيرة الذاتية - أمين خالد",
    page_icon="💼",
    layout="wide"
)

# ---------------- الوضع الليلي والنهاري ----------------
mode = st.radio("اختر الوضع:", ["🌞 نهاري", "🌙 ليلي"])

if mode == "🌞 نهاري":
    bg_color = "#ffffff"
    text_color = "#000000"
    card_color = "#f2f2f2"
else:
    bg_color = "#0e0e0e"
    text_color = "#ffffff"
    card_color = "#1a1a1a"

st.markdown(
    f"""
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .card {{
            background-color: {card_color};
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        }}
        .profile-pic {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 50%;
            width: 150px;
            border: 4px solid gray;
        }}
        .section-title {{
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 10px;
            color: {text_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- صورة شخصية ----------------
st.image("https://i.ibb.co/8d4pFms/profile-pic.png", caption="أمين خالد", width=150)

# ---------------- معلومات أساسية ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown(f"""
# 💼 السيرة الذاتية  
**الاسم:** أمين خالد  
**المسمى:** طالب تقنية معلومات 🎓 | مطور ويب 💻 | مهتم بالذكاء الاصطناعي 🤖  
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- أقسام السيرة الذاتية ----------------
# قسم الخبرات
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📌 الخبرات العملية</div>", unsafe_allow_html=True)
st.markdown("""
- 💻 تطوير مواقع ويب باستخدام **HTML, CSS, JS, PHP, MySQL**
- 🤖 تصميم نماذج تعلم آلي باستخدام **Python, Scikit-learn, TensorFlow**
- 🛠 إدارة قواعد بيانات MySQL و PostgreSQL
- ☁️ استضافة مواقع على InfinityFree + ربط SSL
""")
st.markdown("</div>", unsafe_allow_html=True)

# قسم التعليم
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🎓 التعليم</div>", unsafe_allow_html=True)
st.markdown("""
- بكالوريوس تقنية معلومات – جامعة صنعاء (2022 - 2026)
- دورة تطوير ويب شاملة – Udemy
- دورة الذكاء الاصطناعي – Coursera
""")
st.markdown("</div>", unsafe_allow_html=True)

# قسم المهارات
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🛠 المهارات</div>", unsafe_allow_html=True)
st.markdown("""
- 💡 التفكير النقدي وحل المشكلات  
- 🌐 تطوير مواقع تفاعلية  
- 🐍 برمجة بايثون  
- 🎨 تصميم واجهات أنيقة بـ Bootstrap  
- ⚡ السرعة في التعلم والتجربة  
""")
st.markdown("</div>", unsafe_allow_html=True)

# قسم المشاريع
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🚀 المشاريع</div>", unsafe_allow_html=True)
st.markdown("""
- 📱 مشروع موقع "WorkAway Yemen" للتوظيف التعاوني  
- 🤖 تطبيق لتحليل المشاعر باستخدام الذكاء الاصطناعي  
- 🕹 لعبة مغامرات ثنائية الأبعاد باستخدام GDevelop  
- 📦 مشروع تخرج "شحن" لتوصيل الأغراض بين المحافظات  
""")
st.markdown("</div>", unsafe_allow_html=True)

# قسم التواصل
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📞 التواصل</div>", unsafe_allow_html=True)
st.markdown("""
- 📧 البريد: ameen@example.com  
- 🔗 GitHub: [github.com/ameen](https://github.com/)  
- 🔗 LinkedIn: [linkedin.com/in/ameen](https://linkedin.com)  
""")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- زر تحميل PDF ----------------
cv_path = Path("CV_Ameen.pdf")

if cv_path.exists():
    with open(cv_path, "rb") as pdf_file:
        st.download_button(
            label="📑 تحميل السيرة الذاتية PDF",
            data=pdf_file,
            file_name="CV_Ameen.pdf",
            mime="application/pdf"
        )
else:
    st.warning("⚠️ ملف PDF غير موجود حالياً. الرجاء إنشاؤه يدوياً.")
