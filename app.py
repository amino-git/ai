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
if "classes_known" not in st.session_state:
    st.session_state.classes_known = None

# ---------------------------
# تحميل النموذج المدرب مسبقًا إذا كان موجود
# ---------------------------

if st.session_state.mlp is None or st.session_state.vectorizer is None:
    if os.path.exists("mlp_model.pkl") and os.path.exists("tfidf_vectorizer.pkl"):
        st.session_state.mlp = joblib.load("mlp_model.pkl")
        st.session_state.vectorizer = joblib.load("tfidf_vectorizer.pkl")
        st.session_state.classes_known = np.array([0, 1])

# ---------------------------
# دالة تنظيف النصوص
# ---------------------------

def clean_text(text, lang="ar"):
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", "", text) # إزالة الروابط
    text = re.sub(r"@\w+", "", text) # إزالة الـ mentions
    text = re.sub(r"#", "", text) # إزالة #
    text = re.sub(r"\d+", "", text) # إزالة الأرقام
    if lang == "ar":
        text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text) # إزالة الرموز غير العربية
    else:
        text = re.sub(r"[^\w\s]", "", text) # إزالة الرموز للإنجليزية
    text = re.sub(r"\s+", " ", text).strip() # إزالة الفراغات الزائدة
    return text

# ---------------------------
# دالة التدريب على تغريدة واحدة
# ---------------------------

def train_on_single_tweet(tweet, sentiment, lang, mlp, vectorizer):
    """تدريب النموذج على تغريدة واحدة"""
    try:
        # تنظيف التغريدة
        cleaned_tweet = clean_text(tweet, lang="ar" if lang=="Arabic" else "en")
        
        if cleaned_tweet.strip() == "":
            return False, "⚠️ النص لا يحتوي على كلمات صالحة للتدريب."
        
        # تحويل التغريدة إلى متجه
        tweet_vector = vectorizer.transform([cleaned_tweet])
        
        # تحويل التصنيف إلى رقم
        sentiment_label = 1 if sentiment == "إيجابية" else 0
        
        # تعريف الفئات الممكنة إذا لم تكن معرفة مسبقاً
        if st.session_state.classes_known is None:
            st.session_state.classes_known = np.array([0, 1])
        
        # استخدام partial_fit مع تعريف الفئات
        mlp.partial_fit(tweet_vector, [sentiment_label], classes=st.session_state.classes_known)
        
        return True, f"✅ تم تدريب النموذج بنجاح على التغريدة ({sentiment})"
        
    except Exception as e:
        return False, f"❌ حدث خطأ أثناء التدريب: {str(e)}"

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
            <p>Amily هو نظام لتصنيف التغريدات إلى إيجابية وسلبية، ويدعم كل من اللغة العربية والإنجليزية. يتيح للمستخدم تحليل النصوص وتجربة التغريدات الجديدة مباشرة بعد التدريب، مما يسهل تصنيف النصوص بسرعة ودقة، سواء لأغراض تعليمية أو تحليل بيانات وسائل التواصل الاجتماعي.</p>
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
            
            <h3>👨‍💻 عن المطور</h3>
            <p>المطور: أمين خالد الجبري<br>
            الوظيفة: طالب في جامعة الجزيرة، قسم تقنية المعلومات، المستوى الرابع<br>
            سنة التطوير: 2025<br>
            البريد الإلكتروني: <a href="mailto:amin.khaled.ali@gmail.com">amin.khaled.ali@gmail.com</a><br>
            واتساب: <a href="https://wa.me/967775941498" target="_blank">+967 775941498</a></p>
            
            <p>ملاحظات: المشروع تم تطويره كجزء من دراسة تقنية المعلومات، ويهدف إلى التعلم العملي واكتساب مهارات الذكاء الاصطناعي وتحليل النصوص بشكل احترافي.</p>
            </div>
            """, unsafe_allow_html=True
        )
    
    # تحسين تصميم الزر
    st.markdown("""
    <style>
    .stButton>button {
        background-color: #2a2a2a;
        color: #ffffff;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
        width: 100%;
        font-family: "Segoe UI",Tahoma,Geneva,Verdana,sans-serif;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1f1f1f;
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
    .stTextArea>div>div>textarea { background-color: #2a2a40; color: #f5f5f5; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    body { background-color: #ffffff; color: #0a0a23; }
    .stButton>button { background-color: #0a0a23; color: #ffffff; border-radius:8px; padding:0.5em 1em; font-weight:bold; }
    .stTextInput>div>div>input { background-color: #f0f0f0; color: #0a0a23; border-radius:5px; padding:0.5em; }
    .stRadio>div>div { background-color: #f0f0f0; color:#0a0a23; border-radius:5px; padding:0.3em; }
    .stTextArea>div>div>textarea { background-color: #f0f0f0; color: #0a0a23; }
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
    
    # ---------------------------
    # قسم التدريب على تغريدة واحدة
    # ---------------------------
    st.subheader("🎯 التدريب على تغريدة واحدة")
    
    single_tweet = st.text_area("ادخل التغريدة للتدريب", placeholder="اكتب التغريدة هنا...", height=100)
    single_tweet_lang = st.radio("اختر لغة التغريدة", ["Arabic", "English"], key="single_tweet_lang")
    tweet_sentiment = st.radio("اختر تصنيف التغريدة", ["إيجابية", "سلبية"], key="tweet_sentiment")
    
    if st.button("تدريب النموذج على هذه التغريدة", key="train_single"):
        if not single_tweet.strip():
            st.warning("⚠️ الرجاء إدخال تغريدة للتدريب")
        elif st.session_state.mlp is None or st.session_state.vectorizer is None:
            st.warning("⚠️ يجب تدريب النموذج أولاً على ملف بيانات قبل التدريب على تغريدة واحدة")
        else:
            with st.spinner("⏳ جاري تحديث النموذج..."):
                success, message = train_on_single_tweet(
                    single_tweet, 
                    tweet_sentiment, 
                    single_tweet_lang,
                    st.session_state.mlp,
                    st.session_state.vectorizer
                )
                
                if success:
                    # حفظ النموذج المحدث
                    joblib.dump(st.session_state.mlp, "mlp_model.pkl")
                    st.success(message)
                    
                    # عرض التغريدة بعد التنظيف
                    cleaned_tweet = clean_text(single_tweet, lang="ar" if single_tweet_lang=="Arabic" else "en")
                    st.info(f"📝 التغريدة بعد التنظيف: {cleaned_tweet}")
                else:
                    st.error(message)
    
    st.markdown("---")
    
    # ---------------------------
    # قسم رفع ملفات التدريب
    # ---------------------------
    st.subheader("📁 رفع ملفات التدريب CSV/TSV")
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
                    
                    # إعداد MLP
                    mlp = MLPClassifier(hidden_layer_sizes=(200,100), max_iter=100, random_state=42)
                    
                    st.markdown("""
                    <style>
                    div.stProgress > div > div > div > div {
                        background-color: green;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    progress_bar = st.progress(0)
                    
                    # تدريب النموذج
                    mlp.fit(X_train, y_train)
                    progress_bar.progress(100)
                    
                    # حفظ النموذج وتعريف الفئات
                    st.session_state.mlp = mlp
                    st.session_state.vectorizer = vectorizer
                    st.session_state.classes_known = np.unique(y)
                    
                    joblib.dump(mlp, "mlp_model.pkl")
                    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
                    st.success("✅ تم التدريب بنجاح!")
                    
                    # تقييم النموذج
                    y_pred = mlp.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    # عرض النتائج
                    st.subheader("📊 نتائج التقييم")
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
                    </style>
                    <table class="metrics-table">
                        <tr><td>الدقة (Accuracy)</td><td>{acc*100:.2f}%</td></tr>
                        <tr><td>الدقة (Precision)</td><td>{prec*100:.2f}%</td></tr>
                        <tr><td>الاستدعاء (Recall)</td><td>{rec*100:.2f}%</td></tr>
                        <tr><td>مقياس F1</td><td>{f1*100:.2f}%</td></tr>
                    </table>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ---------------------------
    # قسم تجربة التغريدات الجديدة
    # ---------------------------
    st.subheader("🔍 تجربة التغريدات الجديدة")
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
                proba = st.session_state.mlp.predict_proba(tweet_vector)[0]
                st.info(f"➡️ التغريدة إيجابية (الثقة: {proba[1]*100:.1f}%)" if pred == 1 else f"➡️ التغريدة سلبية (الثقة: {proba[0]*100:.1f}%)")
        else:
            st.warning("⚠️ النموذج غير مدرب بعد")

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
    <hr>
    <div style='text-align: center; color: #666; font-size: 14px;'>
        © 2025 Amin Al Gbri
    </div>
    """, unsafe_allow_html=True
)
