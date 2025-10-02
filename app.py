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

# مكتبات Google Drive عبر OAuth
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# تحميل stopwords للغة العربية
nltk.download('stopwords', quiet=True)

# ---------------------------
# إعداد Google Drive (OAuth)
# ---------------------------
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_JSON = "credentials.json"  # ملف Client ID و Client Secret من Google Cloud

# التحقق من وجود token سابق
if os.path.exists("token.pkl"):
    creds = joblib.load("token.pkl")
else:
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_JSON, SCOPES)
    creds = flow.run_local_server(port=0)  # يفتح نافذة لتسجيل دخول Gmail
    joblib.dump(creds, "token.pkl")  # حفظ التوكن لاستخدامه لاحقًا

drive_service = build('drive', 'v3', credentials=creds)

# معرف المجلد في Google Drive (ضعه إذا أردت حفظ الملفات في مجلد محدد)
DRIVE_FOLDER_ID = "1mOXjtLO5q6lKgt8cCeVBlGVZdIhTOl7W"  # None يعني رفعها في My Drive مباشرة

def upload_to_drive(local_file, drive_folder_id=DRIVE_FOLDER_ID):
    try:
        file_name = os.path.basename(local_file)
        media = MediaFileUpload(local_file, resumable=True)

        if drive_folder_id:
            query = f"name='{file_name}' and '{drive_folder_id}' in parents and trashed=false"
            result = drive_service.files().list(q=query, fields="files(id, name)").execute()
            files = result.get('files', [])
        else:
            files = []

        if files:
            file_id = files[0]['id']
            drive_service.files().update(fileId=file_id, media_body=media).execute()
        else:
            file_metadata = {"name": file_name}
            if drive_folder_id:
                file_metadata["parents"] = [drive_folder_id]
            drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()

        st.success(f"✅ تم رفع {file_name} بنجاح إلى Google Drive!")
    except Exception as e:
        st.error(f"❌ فشل رفع {local_file} إلى Google Drive:\n{e}")

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
# تحميل النموذج إذا كان موجوداً
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
    text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if lang == "ar":
        words = [stemmer.stem(w) for w in text.split() if w not in arabic_stopwords]
    else:
        words = [w for w in text.split() if len(w) > 1]
    return " ".join(words)

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

        if file_lang == "Arabic":
            df = df[df['text'].str.contains(r'[\u0600-\u06FF]', na=False)]
        df["clean_text"] = df["text"].apply(lambda x: clean_text(x, lang="ar" if file_lang=="Arabic" else "en"))
        df = df[df["clean_text"].str.strip() != ""]

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
                    
                    # رفع الملفات إلى Google Drive
                    upload_to_drive("mlp_model.pkl")
                    upload_to_drive("tfidf_vectorizer.pkl")

                    st.success("✅ تم التدريب ورفع الملفات إلى Google Drive!")

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
                
                # رفع الملفات الجديدة إلى Google Drive
                upload_to_drive("mlp_model.pkl")
                upload_to_drive("tfidf_vectorizer.pkl")

                st.success("✅ تم تحديث النموذج بالتغريدة ورفع الملفات إلى Google Drive")

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
