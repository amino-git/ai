import streamlit as st

# Page config
st.set_page_config(page_title="My CV", page_icon="💼", layout="wide")

# ====== Custom CSS for Styling ======
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
            color: #333333;
            font-family: 'Arial', sans-serif;
        }
        .profile-pic {
            border-radius: 50%;
            width: 150px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.3);
        }
        .header {
            text-align: center;
            padding: 20px;
        }
        .header h1 {
            margin-bottom: 5px;
        }
        .social-icons a {
            margin: 0 10px;
            text-decoration: none;
            font-size: 22px;
        }
        .card {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        .timeline {
            border-left: 3px solid #00aaff;
            padding-left: 20px;
            margin-top: 20px;
        }
        .skill-card {
            background-color: #333;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ====== Header Section ======
st.markdown("<div class='header'>", unsafe_allow_html=True)
st.image("https://via.placeholder.com/150", caption="Ameen Khaled", use_container_width=False, output_format="PNG", width=150)
st.markdown("<h1>John Robert Smith</h1>", unsafe_allow_html=True)
st.markdown("Web Designer & Developer | UX/UI Expert", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Social Icons
st.markdown("""
<div class="social-icons" style="text-align:center;">
    <a href="#">🌐</a>
    <a href="#">💼</a>
    <a href="#">🐦</a>
    <a href="#">📷</a>
    <a href="#">📧</a>
</div>
""", unsafe_allow_html=True)

# Buttons
col1, col2 = st.columns(2)
with col1:
    st.link_button("📥 Download CV", "https://your-cv-link.com")
with col2:
    st.link_button("✉️ Contact Me", "mailto:yourmail@gmail.com")

# ====== Experience Section ======
st.markdown("## 📌 Experience")
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card timeline'>", unsafe_allow_html=True)
    st.subheader("Chief Project Manager")
    st.write("Lorem Ipsum Technology - South Africa")
    st.write("Jan 2012 - Dec 2015")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse varius.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card timeline'>", unsafe_allow_html=True)
    st.subheader("Senior UX/UI Designer")
    st.write("Lorem Ipsum Technology - USA")
    st.write("2016 - 2019")
    st.write("Vivamus luctus eros aliquet convallis ultricies.")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card timeline'>", unsafe_allow_html=True)
    st.subheader("Frontend Developer")
    st.write("Lorem Ipsum - Germany")
    st.write("2019 - 2021")
    st.write("Duis aute irure dolor in reprehenderit in voluptate velit esse cillum.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card timeline'>", unsafe_allow_html=True)
    st.subheader("AI Engineer")
    st.write("Tech Corp - UK")
    st.write("2021 - Present")
    st.write("Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia.")
    st.markdown("</div>", unsafe_allow_html=True)

# ====== Skills Section ======
st.markdown("## 🛠️ Skills")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='skill-card'>", unsafe_allow_html=True)
    st.subheader("Professional")
    st.write("UI/UX Design - 90%")
    st.progress(0.9)
    st.write("PHP Web Apps - 80%")
    st.progress(0.8)
    st.write("Android Dev - 70%")
    st.progress(0.7)
    st.write("Adobe Tools - 70%")
    st.progress(0.7)
    st.write("MS Office - 80%")
    st.progress(0.8)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='skill-card'>", unsafe_allow_html=True)
    st.subheader("Personal")
    st.write("Committed - 100%")
    st.progress(1.0)
    st.write("Leadership - 90%")
    st.progress(0.9)
    st.write("Punctual - 95%")
    st.progress(0.95)
    st.write("Communicative - 90%")
    st.progress(0.9)
    st.write("Analytical Skill - 80%")
    st.progress(0.8)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='skill-card'>", unsafe_allow_html=True)
    st.subheader("Languages")
    st.write("English - 100%")
    st.progress(1.0)
    st.write("Spanish - 90%")
    st.progress(0.9)
    st.write("French - 80%")
    st.progress(0.8)
    st.write("Arabic - 70%")
    st.progress(0.7)
    st.write("Hindi - 60%")
    st.progress(0.6)
    st.markdown("</div>", unsafe_allow_html=True)
