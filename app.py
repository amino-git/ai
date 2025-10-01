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
# إعدادات الصفحة وشريط الوضع الليلي
# ---------------------------
st.set_page_config(page_title="Amily 📝", layout="centered", initial_sidebar_state="auto")

with st.sidebar:
    st.title("الإعدادات")
    
    # الوضع الليلي
    st.session_state.dark_mode = st.checkbox("🌙 تفعيل الوضع الليلي", value=st.session_state.dark_mode)
    
    # متغير الحالة للزر
    if "show_info" not in st.session_state:
        st.session_state.show_info = False


    # زر Toggle
    if st.button("وصف المشروع", key="info_btn"):
        st.session_state.show_info = not st.session_state.show_info

    # عرض/إخفاء المعلومات
    if st.session_state.show_info:
        st.markdown(
            """
            <div style='font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif; font-size:14px; line-height:1.5; color:black;'>
            <h3>📝 وصف المشروع</h3>
Amily هو نظام لتصنيف التغريدات إلى **إيجابية وسلبية**، ويدعم كلاً من **اللغة العربية والإنجليزية**.  
النظام يسمح للمستخدم بتحليل النصوص وتجربة التغريدات الجديدة مباشرة بعد التدريب، ويهدف لتسهيل تصنيف النصوص بسرعة ودقة، سواء لأغراض تعليمية أو تحليل بيانات وسائل التواصل الاجتماعي.<br>
تم تصميم المشروع ليكون سهل الاستخدام، ويمكن لأي شخص بدون خبرة سابقة تجربة تصنيف التغريدات وفهم عمل نماذج الذكاء الاصطناعي على النصوص.

<h3>⚙️ طريقة العمل</h3>
- رفع ملفات CSV أو TSV تحتوي على التغريدات مع تصنيفها (pos/neg).<br>
- تنظيف النصوص تلقائياً من الروابط، الرموز، الأرقام، الوسوم، واستبعاد الكلمات الشائعة.<br>
- تدريب النموذج على البيانات المدخلة باستخدام <strong>MLPClassifier</strong> و <strong>TF-IDF Vectorizer</strong>.<br>
- تجربة التغريدات الجديدة لمعرفة تصنيفها مباشرة.<br>
- إمكانية تحديث النموذج من تغريدة واحدة لتعزيز دقة التصنيف بسرعة وسهولة.<br>

<h3>📖 التعليمات</h3>
1. اختر لغة الملف قبل التدريب لتجنب الأخطاء في تنظيف النصوص.<br>
2. لا تستخدم ملفات فارغة أو نصوص غير صالحة، لتفادي أي مشاكل أثناء التدريب.<br>
3. بعد إتمام التدريب، يمكن تجربة أي تغريدة جديدة مباشرة في واجهة المستخدم.<br>
4. يفضل رفع ملفات بحجم متوسط لضمان سرعة التدريب وتحسين أداء النموذج.<br>
5. النظام مخصص للأغراض التعليمية والتجريبية، ويتيح تعلم مهارات الذكاء الاصطناعي وتحليل النصوص عملياً.<br>

<h3>👨‍💻 عن المطور</h3>
المطور: أمين خالد الجبري<br>
الوظيفة: طالب في جامعة الجزيرة، قسم تقنية المعلومات، المستوى الرابع<br>
سنة التطوير: 2025<br>
البريد الإلكتروني: <a href="mailto:amin.khaled.ali@gmail.com">amin.khaled.ali@gmail.com</a><br>
واتساب: <a href="https://wa.me/967775941498" target="_blank">+967 775941498</a><br>
ملاحظات: المشروع تم تطويره كجزء من دراسة تقنية المعلومات، ويهدف إلى التعلم العملي واكتساب مهارات الذكاء الاصطناعي وتحليل النصوص بشكل احترافي.
وما زال قيد التدريب حيث وصلت دقته حاليا الى 78% فقط.
            """, unsafe_allow_html=True
        )

# تحسين تصميم الزر
st.markdown("""
    <style>
    .stButton>button {
        background-color: #2a2a2a;  /* اللون الأساسي: أسود خفيف */
        color: #ffffff;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
        width: 100%;
        font-family: "Segoe UI",Tahoma,Geneva,Verdana,sans-serif;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1f1f1f;  /* عند التمرير: أسود أغمق */
    }
    </style>
""", unsafe_allow_html=True)




# تطبيق الوضع الليلي
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        body { background-color: #1e1e2e; color: #f5f5f5; }
        .stButton>button { background-color: #2a2a40; color: #f5f5f5; border-radius:8px; padding:0.5em 1em; font-weight:bold; }
        .stTextInput>div>div>input { background-color: #2a2a40; color: #f5f5f5; border-radius:5px; padding:0.5em; }
        .stRadio>div>div { background-color: #2a2a40; color:#f5f5f5; border-radius:5px; padding:0.3em; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body { background-color: #ffffff; color: #0a0a23; }
        .stButton>button { background-color: #0a0a23; color: #ffffff; border-radius:8px; padding:0.5em 1em; font-weight:bold; }
        .stTextInput>div>div>input { background-color: #f0f0f0; color: #0a0a23; border-radius:5px; padding:0.5em; }
        .stRadio>div>div { background-color: #f0f0f0; color:#0a0a23; border-radius:5px; padding:0.3em; }
        </style>
    """, unsafe_allow_html=True)

# ---------------------------
# واجهة المستخدم
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

    st.markdown("---")

    st.subheader("تجربة التغريدات الجديدة")
    new_tweet = st.text_input("ادخل تغريدة للتصنيف", key="new_tweet_admin")
    new_lang = st.radio("اختر لغة التغريدة", ["Arabic","English"], key="new_lang_admin")
    if st.button("صنف التغريدة", key="predict_new"):
        if st.session_state.mlp and st.session_state.vectorizer:
            tweet_clean = clean_text(new_tweet, lang="ar" if new_lang=="Arabic" else "en")
            if tweet_clean.strip() == "":
                st.warning("⚠️ النص لا يحتوي على كلمات صالحة للتصنيف.")
            else:
                tweet_vector = st.session_state.vectorizer.transform([tweet_clean])
                pred = st.session_state.mlp.predict(tweet_vector)[0]
                st.info("➡️ التغريدة إيجابية" if pred == 1 else "➡️ التغريدة سلبية")
        else:
            st.warning("⚠️ النموذج غير مدرب بعد")

    st.markdown("---")

    st.subheader("تدريب النموذج من تغريدة واحدة")
    tweet_to_train = st.text_input("ادخل تغريدة للتدريب", key="train_one_admin")
    y_label = st.radio("اختر التصنيف للتغريدة", ["pos","neg"], key="label_one_admin")
    if st.button("تدريب النموذج على هذه التغريدة", key="train_one_btn"):
        if st.session_state.mlp and st.session_state.vectorizer:
            tweet_clean = clean_text(tweet_to_train, lang="ar" if new_lang=="Arabic" else "en")
            if tweet_clean.strip() == "":
                st.warning("⚠️ النص لا يحتوي على كلمات صالحة للتدريب.")
            else:
                X_new = st.session_state.vectorizer.transform([tweet_clean])
                y_new = [1 if y_label=="pos" else 0]
                st.session_state.mlp.partial_fit(X_new, y_new)
                joblib.dump(st.session_state.mlp, "mlp_model.pkl")
                joblib.dump(st.session_state.vectorizer, "tfidf_vectorizer.pkl")
                st.success("✅ تم تحديث النموذج بالتغريدة")

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
    <div style='text-align:center; opacity:0.4; margin-top:30px;'>
        © 2025 Amin Al Gbri
    </div>
    """, unsafe_allow_html=True
)
