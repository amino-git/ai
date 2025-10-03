import streamlit as st
import base64

# إعداد الصفحة
st.set_page_config(page_title="CV - أمين", page_icon="💼", layout="wide")

# ======= الوضع الليلي / النهاري =======
dark_mode = st.sidebar.radio("اختر الوضع:", ["🌞 نهاري", "🌙 ليلي"])
if dark_mode == "🌙 ليلي":
    bg_color = "#0f0f0f"
    text_color = "#f5f5f5"
    box_color = "#1e1e1e"
else:
    bg_color = "#f5f5f5"
    text_color = "#0f0f0f"
    box_color = "#ffffff"

st.markdown(
    f"""
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .cv-box {{
            background-color: {box_color};
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }}
        .profile-pic {{
            border-radius: 50%;
            width: 180px;
            margin-bottom: 20px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.3);
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ======= الصورة الشخصية =======
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("https://via.placeholder.com/200", caption="أمين خالد", width=180, output_format="PNG", use_column_width=False)

# ======= الاسم والنبذة =======
st.markdown(f"<h1 style='text-align:center; color:{text_color};'>💼 السيرة الذاتية</h1>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style='text-align:center; font-size:18px; color:{text_color};'>
    طالب تقنية معلومات 🎓 | مطور ويب 💻 | مهتم بالذكاء الاصطناعي 🤖
    </div>
    """,
    unsafe_allow_html=True,
)

# ======= تحميل PDF =======
def get_binary_file_downloader(file_path, file_label):
    with open(file_path, "rb") as file:
        btn = st.download_button(
            label=f"📥 {file_label}",
            data=file,
            file_name="CV_Ameen.pdf",
            mime="application/pdf"
        )
    return btn

st.markdown("### 📑 تحميل السيرة الذاتية")
# ضع ملف CV_Ameen.pdf في مجلد المشروع
get_binary_file_downloader("CV_Ameen.pdf", "تحميل ملف PDF")

# ======= الأقسام =======
st.markdown("### 👤 البيانات الشخصية")
with st.container():
    st.write("""
    - 📍 اليمن  
    - 📧 amin.khaled.ali@gmail.com  
    - 📱 777-XXX-XXX  
    - 🌐 [موقعي الشخصي](https://amin.kesug.com)
    """)

st.markdown("### 🎓 التعليم")
st.write("- بكالوريوس تقنية معلومات – جامعة ... (2022 - حتى الآن)")

st.markdown("### 💼 الخبرات")
st.write("""
- تطوير مواقع ويب (PHP, MySQL, Bootstrap).  
- بناء نماذج ذكاء اصطناعي (ML, NLP).  
- تصميم ألعاب بسيطة (GDevelop, Streamlit).  
""")

st.markdown("### 🛠️ المهارات")
skills = {
    "Python": 80,
    "HTML / CSS": 90,
    "JavaScript": 70,
    "SQL": 75,
    "Machine Learning": 65
}
for skill, level in skills.items():
    st.progress(level / 100)
    st.write(f"**{skill}** - {level}%")

st.markdown("### 📂 المشاريع")
st.write("""
- 🌐 [Workaway Clone](https://github.com/yourrepo) – موقع للتواصل بين العمال والمضيفين.  
- 🤖 [مشروع ذكاء اصطناعي](https://github.com/yourrepo) – تصنيف النصوص والمشاعر.  
- 🎮 [ألعاب Streamlit](https://yourgame.streamlit.app) – ألعاب تفاعلية مستضافة على Streamlit.  
""")

# ======= روابط التواصل =======
st.markdown("### 📞 للتواصل")
col1, col2, col3 = st.columns(3)
with col1:
    st.link_button("🌐 GitHub", "https://github.com/yourusername")
with col2:
    st.link_button("🔗 LinkedIn", "https://linkedin.com/in/yourusername")
with col3:
    st.link_button("✉️ Email", "mailto:amin.khaled.ali@gmail.com")
