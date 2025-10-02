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
import numpy as np  # <-- إضافة للمصفوفات وإدارة التقدم


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
def clean_text(text, lang="ar"):
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # إزالة الروابط
    text = re.sub(r"@\w+", "", text)             # إزالة الـ mentions
    text = re.sub(r"#", "", text)                # إزالة #
    text = re.sub(r"\d+", "", text)              # إزالة الأرقام
    if lang == "ar":
        text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)  # إزالة الرموز غير العربية
    else:
        text = re.sub(r"[^\w\s]", "", text)               # إزالة الرموز للإنجليزية
    text = re.sub(r"\s+", " ", text).strip()             # إزالة الفراغات الزائدة
    
    return text

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

    # زر Toggle لعرض وصف المشروع
    if st.button("وصف المشروع", key="info_btn"):
        st.session_state.show_info = not st.session_state.show_info

    # عرض/إخفاء المعلومات
    if st.session_state.show_info:
        st.markdown(
            """
            <div style='font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif; font-size:14px; line-height:1.6; color:#222;'>

            <h3>📝 وصف المشروع</h3>
            <p>Amily هو نظام لتصنيف التغريدات إلى إيجابية وسلبية، ويدعم كل من اللغة العربية والإنجليزية.  
            يتيح للمستخدم تحليل النصوص وتجربة التغريدات الجديدة مباشرة بعد التدريب، مما يسهل تصنيف النصوص بسرعة ودقة، سواء لأغراض تعليمية أو تحليل بيانات وسائل التواصل الاجتماعي.</p>
            <p>تم تصميم المشروع ليكون سهل الاستخدام، ويمكن لأي شخص بدون خبرة سابقة تجربة تصنيف التغريدات وفهم عمل نماذج الذكاء الاصطناعي على النصوص.</p>

            <h3>⚙️ طريقة العمل</h3>
            <ul>
                <li>رفع ملفات CSV أو TSV تحتوي على التغريدات مع تصنيفها (pos/neg).</li>
                <li>تنظيف النصوص تلقائيًا من الروابط، الرموز، الأرقام، الوسوم، واستبعاد الكلمات الشائعة.</li>
                <li>تدريب النموذج على البيانات المدخلة باستخدام <strong>MLPClassifier</strong> و <strong>TF-IDF Vectorizer</strong>.</li>
                <li>تجربة التغريدات الجديدة لمعرفة تصنيفها مباشرة.</li>
                <li>تحديث النموذج من تغريدة واحدة لتعزيز دقة التصنيف بسرعة وسهولة.</li>
            </ul>

            <h3>📖 التعليمات</h3>
            <ol>
                <li>اختر لغة الملف قبل التدريب لتجنب الأخطاء في تنظيف النصوص.</li>
                <li>لا تستخدم ملفات فارغة أو نصوص غير صالحة لتفادي أي مشاكل أثناء التدريب.</li>
                <li>بعد التدريب، يمكن تجربة أي تغريدة جديدة مباشرة في واجهة المستخدم.</li>
                <li>يفضل رفع ملفات بحجم متوسط لضمان سرعة التدريب وتحسين أداء النموذج.</li>
                <li>النظام مخصص للأغراض التعليمية والتجريبية، ويتيح تعلم مهارات الذكاء الاصطناعي وتحليل النصوص عمليًا.</li>
            </ol>

            <h3>📌 ملاحظة هامة حول التدريب أونلاين</h3>
            <p>أثناء تدريب النموذج على Streamlit Cloud، البيانات تُحفظ مؤقتًا، وعند تحديث الصفحة أو إعادة تشغيل التطبيق، يتم فقدان كل التدريب.</p>
            <p>لقد حاولنا ربط التخزين مع Google Drive لحفظ النموذج والبيانات بشكل دائم، وتم تفعيل Google Cloud، لكن التخزين المشترك يحتاج إلى Google Workspace المدفوع.</p>
            <p>كما حاولنا استخدام AWS S3، لكن مشكلة اختيار اسم Bucket فريد عالميًا جعلت الحل صعب التطبيق.</p>
            <p>لذلك، Colab أو GitHub غير مناسبين للتدريب المباشر وحفظ البيانات بشكل دائم وآمن.</p>

            <h3>👨‍💻 عن المطور</h3>
            <p>المطور: أمين خالد الجبري<br>
            الوظيفة: طالب في جامعة الجزيرة، قسم تقنية المعلومات، المستوى الرابع<br>
            سنة التطوير: 2025<br>
            البريد الإلكتروني: <a href="mailto:amin.khaled.ali@gmail.com">amin.khaled.ali@gmail.com</a><br>
            واتساب: <a href="https://wa.me/967775941498" target="_blank">+967 775941498</a></p>
            <p>ملاحظات: المشروع تم تطويره كجزء من دراسة تقنية المعلومات، ويهدف إلى التعلم العملي واكتساب مهارات الذكاء الاصطناعي وتحليل النصوص بشكل احترافي. النموذج الحالي وصل دقته إلى حوالي 78%.</p>

            </div>
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
                    vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1,2))
                    X = vectorizer.fit_transform(df_balanced["clean_text"])
                    y = df_balanced["label"].map({"neg":0,"pos":1})

                    # تقسيم بيانات للتقييم
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # إعداد MLP مع warm_start للتكرار اليدوي
                    mlp = MLPClassifier(hidden_layer_sizes=(200,100), max_iter=1, warm_start=True, random_state=42)


                    st.markdown("""
                        <style>
                        /* تغيير لون شريط التقدم */
                        div.stProgress > div > div > div > div {
                            background-color:green;  /* لون أصفر غامق */
                        }
                        </style>
                    """, unsafe_allow_html=True)


                    progress_bar = st.progress(0)
                    n_epochs = 20  # عدد الدورات التدريبية
                    for epoch in range(n_epochs):
                        mlp.fit(X_train, y_train)
                        progress_bar.progress((epoch+1)/n_epochs)

                    # حفظ النموذج
                    st.session_state.mlp = mlp
                    st.session_state.vectorizer = vectorizer
                    joblib.dump(mlp, "mlp_model.pkl")
                    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
                    
                    st.success("✅ تم التدريب بنجاح!")

                # ---------------------------
                # تقييم النموذج
                y_pred = mlp.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                report_dict = classification_report(
                    y_test, y_pred, target_names=["Negative","Positive"], output_dict=True
                )

                # ---------------------------
                # جدول HTML واحد أنيق وصغير
                st.subheader("📊 Model Evaluation Summary")
                st.markdown(f"""
                <style>
                    .metrics-table {{
                        width: 90%;
                        margin:auto;
                        border-collapse: collapse;
                        font-family: Arial, sans-serif;
                        font-size: 12px;
                        text-align: center;
                    }}
                    .metrics-table th, .metrics-table td {{
                        border: 1px solid #ccc;
                        padding: 4px;
                    }}
                    .metrics-table th {{
                        background-color: #222;
                        color: #fff;
                    }}
                    .metrics-table tr:nth-child(even) {{
                        background-color: #f9f9f9;
                    }}
                    .metrics-table tr:hover {{
                        background-color: #ececec;
                    }}
                    .section-header {{
                        background-color: #555;
                        color: #fff;
                        font-size: 13px;
                    }}
                </style>

                <table class="metrics-table">
                    <tr class="section-header"><th colspan="2">Overall Metrics</th></tr>
                    <tr><td>Accuracy</td><td>{acc*100:.2f}%</td></tr>
                    <tr><td>Precision</td><td>{prec*100:.2f}%</td></tr>
                    <tr><td>Recall</td><td>{rec*100:.2f}%</td></tr>
                    <tr><td>F1-score</td><td>{f1*100:.2f}%</td></tr>
                </table>
                """, unsafe_allow_html=True)


 
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

  
# ---------------------------
# Footer
# ---------------------------
# st.markdown(
#     """
#     <div style='text-align:center; opacity:0.4; margin-top:30px;'>
#         © 2025 Amin Al Gbri
#     </div>
#     """, unsafe_allow_html=True
# )





