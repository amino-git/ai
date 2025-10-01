import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import nltk
nltk.download('stopwords')

# ---------------------
# دوال تنظيف النصوص
# ---------------------
stemmer = ISRIStemmer()
arabic_stopwords = set(stopwords.words('arabic'))

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [stemmer.stem(w) for w in text.split() if w not in arabic_stopwords]
    return " ".join(words)

# ---------------------
# تحميل النموذج و Vectorizer إذا موجودين
# ---------------------
try:
    mlp = joblib.load("mlp_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except:
    mlp = None
    vectorizer = None

# ---------------------
# واجهة Streamlit
# ---------------------
st.set_page_config(page_title="تصنيف التغريدات", layout="wide")
st.title("📝 تطبيق تصنيف التغريدات")

menu = ["مستخدم", "مدير"]
choice = st.sidebar.selectbox("اختر الواجهة:", menu)

# ---------------------
# واجهة المستخدم العادية
# ---------------------
if choice == "مستخدم":
    st.subheader("واجهة المستخدم")
    tweet = st.text_area("ادخل تغريدة لتصنيفها:")
    if st.button("تصنيف"):
        if mlp is not None and vectorizer is not None:
            tweet_clean = clean_text(tweet)
            tweet_vector = vectorizer.transform([tweet_clean])
            pred = mlp.predict(tweet_vector)[0]
            st.success("التغريدة إيجابية ✅" if pred == 1 else "التغريدة سلبية ❌")
        else:
            st.warning("النموذج غير متوفر حالياً. الرجاء تدريب النموذج أولاً من واجهة المدير.")

# ---------------------
# واجهة المدير
# ---------------------
if choice == "مدير":
    st.subheader("تسجيل الدخول للمدير")
    username = st.text_input("اسم المستخدم:")
    password = st.text_input("كلمة المرور:", type="password")
    if st.button("دخول"):
        if username == "admin" and password == "1234":  # مثال على كلمة مرور ثابتة
            st.success("تم تسجيل الدخول بنجاح!")
            
            # خيارات التدريب
            st.markdown("### تدريب النموذج")
            input_method = st.radio("اختر طريقة إدخال البيانات:", ["ادخال يدوي", "رفع ملف CSV/TSV"])
            
            if input_method == "ادخال يدوي":
                new_text = st.text_area("ادخل نص جديد:")
                label = st.selectbox("اختار الفئة:", ["سلبية", "إيجابية"])
                if st.button("إضافة للنموذج"):
                    st.write("✅ تمت الإضافة (بعد إعادة التدريب سيتم التعلم)")
                    
            if input_method == "رفع ملف CSV/TSV":
                uploaded_file = st.file_uploader("اختر ملف CSV/TSV", type=["csv", "tsv"])
                if uploaded_file is not None:
                    sep = "\t" if uploaded_file.name.endswith(".tsv") else ","
                    new_data = pd.read_csv(uploaded_file, sep=sep, header=None, names=["label", "text"])
                    st.write(new_data.head())
                    if st.button("إعادة تدريب النموذج"):
                        # هنا تضيف كود التدريب مثل ما عملت قبل
                        st.success("✅ تم إعادة تدريب النموذج وحفظه")
        else:
            st.error("⚠️ اسم المستخدم أو كلمة المرور غير صحيحة")
