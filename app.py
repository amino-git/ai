import streamlit as st
import pandas as pd
import joblib
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import nltk

# ---------------------------
# Google Drive API
# ---------------------------
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

# تحميل stopwords للغة العربية
nltk.download('stopwords', quiet=True)

# ---------------------------
# تهيئة الجلسة
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "mlp" not in st.session_state:
    st.session_state.mlp = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ---------------------------
# الاتصال بـ Google Drive
# ---------------------------
def connect_drive():
    try:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],  # تجيب البيانات من secrets.toml
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build("drive", "v3", credentials=creds)
        return service
    except Exception as e:
        st.error(f"⚠️ خطأ في الاتصال بجوجل درايف: {e}")
        return None

def upload_to_drive(service, file_path, file_name):
    file_metadata = {"name": file_name}
    media = MediaFileUpload(file_path, resumable=True)
    f = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return f.get("id")

# ---------------------------
# تحميل النموذج المدرب مسبقًا إذا كان موجود
# ---------------------------
if st.session_state.mlp is None or st.session_state.vectorizer is None:
    if os.path.exists("mlp_model.pkl") and os.path.exists("tfidf_vectorizer.pkl"):
        st.session_state.mlp = joblib.load("mlp_model.pkl")
        st.session_state.vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---------------------------
# دالة تنظيف النصوص
# ---------------------------
stemmer = ISRIStemmer()
arabic_stopwords = set(stopwords.words('arabic'))

def clean_text(text, lang="ar"):
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)  # يسمح بالعربية والإنجليزية
    text = re.sub(r"\s+", " ", text).strip()
    
    if lang == "ar":
        words = [stemmer.stem(w) for w in text.split() if w not in arabic_stopwords]
    else:
        words = [w for w in text.split() if len(w) > 1]  # الإنجليزية بدون stemmer
    
    return " ".join(words)

# ---------------------------
# الواجهة - نفس كودك بدون تغيير
# ---------------------------
st.title("Amily 📝")
st.markdown("---")

st.subheader("واجهة المستخدم")
user_tweet = st.text_input("ادخل تغريدتك هنا")
lang = st.radio("اختر اللغة", ["Arabic", "English"], key="user_lang")

if st.button("صنف التغريدة (المستخدم)"):
    if st.session_state.mlp and st.session_state.vectorizer:
        tweet_clean = clean_text(user_tweet, lang="ar" if lang=="Arabic" else "en")
        if tweet_clean.strip() == "":
            st.warning("⚠️ النص لا يحتوي على كلمات صالحة للتصنيف.")
        else:
            tweet_vector = st.session_state.vectorizer.transform([tweet_clean])
            pred = st.session_state.mlp.predict(tweet_vector)[0]
            st.info("➡️ التغريدة إيجابية" if pred == 1 else "➡️ التغريدة سلبية")
    else:
        st.warning("⚠️ النموذج غير مدرب بعد")

st.markdown("---")

# ---------------------------
# تسجيل الدخول للمدير
# ---------------------------
if not st.session_state.logged_in:
    st.subheader("🔒 تسجيل الدخول للمدير")
    username = st.text_input("اسم المستخدم", key="admin_user")
    password = st.text_input("كلمة المرور", type="password", key="admin_pass")
    if st.button("دخول", key="login_btn"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("تم تسجيل الدخول بنجاح!")
        else:
            st.error("اسم المستخدم أو كلمة المرور غير صحيحة")

# ---------------------------
# واجهة المدير
# ---------------------------
if st.session_state.logged_in:
    st.title("👨‍💼 واجهة المدير")
    
    st.subheader("رفع ملفات التدريب CSV/TSV")
    file = st.file_uploader("ارفع ملف CSV أو TSV للتدريب", type=["csv","tsv"], key="train_file")
    file_lang = st.radio("اختر لغة الملف", ["Arabic","English"], key="file_lang")
    
    if file:
        sep = "\t" if file.name.endswith(".tsv") else ","
        df = pd.read_csv(file, sep=sep, header=None, names=["label","text"])
        df = df.dropna(subset=["label","text"])
        st.success(f"✅ تم رفع الملف بنجاح: {file.name} ({len(df)} سطر)")

        # تنظيف النصوص بحسب اللغة المختارة
        if file_lang == "Arabic":
            df = df[df['text'].str.contains(r'[\u0600-\u06FF]', na=False)]
        df["clean_text"] = df["text"].apply(lambda x: clean_text(x, lang="ar" if file_lang=="Arabic" else "en"))
        df = df[df["clean_text"].str.strip() != ""]

        # توازن البيانات
        df_majority = df[df.label=="neg"]
        df_minority = df[df.label=="pos"]
        if len(df_minority) > 0 and len(df_majority) > 0:
            df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
            df_balanced = pd.concat([df_majority, df_minority_upsampled])
        else:
            df_balanced = df.copy()

        if st.button("تدريب النموذج على الملف"):
            if df_balanced.empty:
                st.error("⚠️ لا توجد نصوص صالحة للتدريب بعد التنظيف!")
            else:
                with st.spinner("⏳ جاري التدريب..."):
                    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
                    X_train = vectorizer.fit_transform(df_balanced["clean_text"])
                    y_train = df_balanced["label"].map({"neg":0,"pos":1})
                    mlp = MLPClassifier(hidden_layer_sizes=(150,50), max_iter=50, random_state=42)
                    mlp.fit(X_train, y_train)
                    
                    st.session_state.mlp = mlp
                    st.session_state.vectorizer = vectorizer
                    joblib.dump(mlp, "mlp_model.pkl")
                    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
                    
                    st.success("✅ تم التدريب بنجاح!")

                    # ✅ رفع الملفين تلقائياً إلى Google Drive
                    service = connect_drive()
                    if service:
                        try:
                            model_id = upload_to_drive(service, "mlp_model.pkl", "mlp_model.pkl")
                            vec_id = upload_to_drive(service, "tfidf_vectorizer.pkl", "tfidf_vectorizer.pkl")
                            st.success(f"📤 تم رفع الملفات إلى Google Drive\nModel ID: {model_id}\nVectorizer ID: {vec_id}")
                        except Exception as e:
                            st.error(f"فشل رفع الملفات: {e}")
