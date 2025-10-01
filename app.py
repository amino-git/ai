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

# ---------------------------
# واجهة المستخدم العادي
# ---------------------------
st.title("📝 تصنيف التغريدات")
st.subheader("واجهة المستخدم")

user_tweet = st.text_input("ادخل تغريدتك هنا")
if st.button("صنف التغريدة (المستخدم)"):
    if st.session_state.mlp and st.session_state.vectorizer:
        tweet_clean = clean_text(user_tweet)
        tweet_vector = st.session_state.vectorizer.transform([tweet_clean])
        pred = st.session_state.mlp.predict(tweet_vector)[0]
        st.info("➡️ التغريدة إيجابية" if pred == 1 else "➡️ التغريدة سلبية")
    else:
        st.warning("⚠️ النموذج غير مدرب بعد")

st.markdown("---")

# ---------------------------
# واجهة تسجيل الدخول للمدير
# ---------------------------
if not st.session_state.logged_in:
    st.subheader("🔒 تسجيل الدخول للمدير")
    username = st.text_input("اسم المستخدم")
    password = st.text_input("كلمة المرور", type="password")
    if st.button("دخول"):
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

    # --- رفع ملفات التدريب ---
    st.subheader("رفع ملفات التدريب CSV/TSV")
    file = st.file_uploader("ارفع ملف CSV أو TSV للتدريب", type=["csv", "tsv"])
    
    if file:
        sep = "\t" if file.name.endswith(".tsv") else ","
        df = pd.read_csv(file, sep=sep, header=None, names=["label", "text"])
        st.success(f"✅ تم رفع الملف بنجاح: {file.name} ({len(df)} سطر)")
        
        df["clean_text"] = df["text"].apply(clean_text)

        df_majority = df[df.label=="neg"]
        df_minority = df[df.label=="pos"]
        if len(df_minority) > 0 and len(df_majority) > 0:
            df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
            df_balanced = pd.concat([df_majority, df_minority_upsampled])
        else:
            df_balanced = df.copy()

        if st.button("تدريب النموذج على الملف"):
            with st.spinner("⏳ جاري التدريب..."):
                vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
                X_train = vectorizer.fit_transform(df_balanced["clean_text"])
                y_train = df_balanced["label"].map({"neg":0,"pos":1})
                mlp = MLPClassifier(hidden_layer_sizes=(150,50), max_iter=50, random_state=42)
                mlp.fit(X_train, y_train)
                
                st.session_state.mlp = mlp
                st.session_state.vectorizer = vectorizer

                # حفظ النموذج والـ vectorizer
                joblib.dump(st.session_state.mlp, "mlp_model.pkl")
                joblib.dump(st.session_state.vectorizer, "tfidf_vectorizer.pkl")

                st.success("✅ تم التدريب بنجاح!")

    st.markdown("---")

    # --- تجربة التغريدات ---
    st.subheader("تجربة النموذج بتغريدات جديدة")
    new_tweet = st.text_input("ادخل تغريدة للتصنيف")
    if st.button("صنف التغريدة"):
        if st.session_state.mlp and st.session_state.vectorizer:
            tweet_clean = clean_text(new_tweet)
            tweet_vector = st.session_state.vectorizer.transform([tweet_clean])
            pred = st.session_state.mlp.predict(tweet_vector)[0]
            st.info("➡️ التغريدة إيجابية" if pred == 1 else "➡️ التغريدة سلبية")
        else:
            st.warning("⚠️ النموذج غير مدرب بعد")

    st.markdown("---")

    # --- تدريب من تغريدة واحدة ---
    st.subheader("تدريب النموذج من تغريدة واحدة")
    tweet_to_train = st.text_input("ادخل تغريدة للتدريب")
    if tweet_to_train:
        y_label = st.radio("اختر التصنيف للتغريدة", ["pos", "neg"])
        if st.button("تدريب النموذج على هذه التغريدة"):
            if st.session_state.mlp and st.session_state.vectorizer:
                tweet_clean = clean_text(tweet_to_train)
                X_new = st.session_state.vectorizer.transform([tweet_clean])
                y_new = [1 if y_label=="pos" else 0]
                st.session_state.mlp.partial_fit(X_new, y_new)

                # حفظ النموذج والـ vectorizer بعد التدريب
                joblib.dump(st.session_state.mlp, "mlp_model.pkl")
                joblib.dump(st.session_state.vectorizer, "tfidf_vectorizer.pkl")

                st.success("✅ تم تحديث النموذج بالتغريدة")
            else:
                st.warning("⚠️ النموذج غير مدرب بعد")
