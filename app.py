import streamlit as st
import pandas as pd
import joblib
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# تحميل stopwords للغة العربية
nltk.download('stopwords', quiet=True)

# حالة الجلسة
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "mlp" not in st.session_state:
    st.session_state.mlp = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# تحميل النموذج المدرب مسبقًا
if st.session_state.mlp is None or st.session_state.vectorizer is None:
    if os.path.exists("mlp_model.pkl") and os.path.exists("tfidf_vectorizer.pkl"):
        st.session_state.mlp = joblib.load("mlp_model.pkl")
        st.session_state.vectorizer = joblib.load("tfidf_vectorizer.pkl")

# تنظيف النصوص
def clean_text(text, lang="ar"):
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)
    if lang == "ar":
        text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)
    else:
        text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# إعداد الواجهة
st.set_page_config(page_title="Amily 📝", layout="centered")

# ========= واجهة المستخدم ==========
st.title("Amily 📝")
st.subheader("واجهة المستخدم")

user_tweet = st.text_input("ادخل تغريدتك هنا")
lang = st.radio("اختر اللغة", ["Arabic", "English"], key="user_lang")

if st.button("صنف التغريدة (المستخدم)"):
    if st.session_state.mlp and st.session_state.vectorizer:
        tweet_clean = clean_text(user_tweet, lang="ar" if lang=="Arabic" else "en")
        if tweet_clean.strip() == "":
            st.warning("⚠️ النص فارغ.")
        else:
            tweet_vector = st.session_state.vectorizer.transform([tweet_clean])
            pred = st.session_state.mlp.predict(tweet_vector)[0]
            st.info("➡️ التغريدة إيجابية" if pred == 1 else "➡️ التغريدة سلبية")
    else:
        st.warning("⚠️ النموذج غير مدرب بعد.")

st.markdown("---")

# ========= تسجيل دخول المدير ==========
if not st.session_state.logged_in:
    st.subheader("🔒 تسجيل الدخول للمدير")
    username = st.text_input("اسم المستخدم")
    password = st.text_input("كلمة المرور", type="password")
    if st.button("دخول"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("تم تسجيل الدخول!")
        else:
            st.error("خطأ في تسجيل الدخول")

# ========= واجهة المدير ==========
if st.session_state.logged_in:
    st.title("👨‍💼 واجهة المدير")
    
    # ========== تدريب تغريدة واحدة ==========
    st.subheader("✨ تدريب تغريدة واحدة")

    single_tweet = st.text_input("اكتب تغريدة لتدريب النموذج")
    single_label = st.radio("نوع التغريدة", ["إيجابية", "سلبية"])

    if st.button("تدريب التغريدة"):
        if single_tweet.strip() == "":
            st.warning("⚠️ لا يمكن تدريب نص فارغ.")
        else:
            clean = clean_text(single_tweet)

            # إذا لا يوجد نموذج → أنشئ واحدًا جديدًا
            if st.session_state.vectorizer is None:
                st.session_state.vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1,2))
                X_init = st.session_state.vectorizer.fit_transform([clean])
            else:
                try:
                    X_init = st.session_state.vectorizer.transform([clean])
                except:
                    st.warning("⚠️ النموذج يحتاج إعادة تدريب كامل لأن البيانات الجديدة خارج المفردات.")
                    X_init = st.session_state.vectorizer.fit_transform([clean])

            y_value = 1 if single_label == "إيجابية" else 0

            if st.session_state.mlp is None:
                st.session_state.mlp = MLPClassifier(hidden_layer_sizes=(200,100), warm_start=True, max_iter=1)
                st.session_state.mlp.partial_fit(X_init, [y_value], classes=[0,1])
            else:
                st.session_state.mlp.partial_fit(X_init, [y_value])

            joblib.dump(st.session_state.mlp, "mlp_model.pkl")
            joblib.dump(st.session_state.vectorizer, "tfidf_vectorizer.pkl")

            st.success("✅ تم تدريب التغريدة بنجاح وتحسين النموذج!")

    st.markdown("---")

    # ========== رفع ملف التدريب ==========
    st.subheader("رفع ملفات التدريب CSV/TSV")
    file = st.file_uploader("ارفع الملف", type=["csv","tsv"])
    file_lang = st.radio("لغة الملف", ["Arabic","English"])

    if file:
        sep = "\t" if file.name.endswith(".tsv") else ","
        df = pd.read_csv(file, sep=sep, header=None, names=["label","text"])
        df = df.dropna()

        if file_lang == "Arabic":
            df = df[df["text"].str.contains(r'[\u0600-\u06FF]')]

        df["clean_text"] = df["text"].apply(lambda x: clean_text(x, "ar" if file_lang=="Arabic" else "en"))
        df = df[df["clean_text"].str.strip() != ""]

        df_majority = df[df.label=="neg"]
        df_minority = df[df.label=="pos"]

        if len(df_minority) > 0 and len(df_majority) > 0:
            df_minority_up = resample(df_minority, replace=True, n_samples=len(df_majority))
            df_bal = pd.concat([df_majority, df_minority_up])
        else:
            df_bal = df.copy()

        if st.button("تدريب النموذج على الملف"):
            if df_bal.empty:
                st.error("⚠️ الملف لا يحتوي على بيانات مناسبة.")
            else:
                with st.spinner("⏳ جاري التدريب..."):
                    vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1,2))
                    X = vectorizer.fit_transform(df_bal["clean_text"])
                    y = df_bal["label"].map({"neg":0,"pos":1})

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                    mlp = MLPClassifier(hidden_layer_sizes=(200,100), warm_start=True, max_iter=1)

                    progress_bar = st.progress(0)
                    for epoch in range(20):
                        mlp.fit(X_train, y_train)
                        progress_bar.progress((epoch+1)/20)

                    st.session_state.mlp = mlp
                    st.session_state.vectorizer = vectorizer

                    joblib.dump(mlp, "mlp_model.pkl")
                    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

                    st.success("✅ تم التدريب!")

                y_pred = mlp.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.info(f"دقة النموذج: {acc:.2f}")
