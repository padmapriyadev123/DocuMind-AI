import streamlit as st
from pdf_processor import extract_text_from_pdf, chunk_pages
from embeddings import get_embeddings
from vector_store import build_index, load_index, search
from qa_chain import answer_question

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject mobile viewport meta tag
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* FIX TEXT VISIBILITY */
.stApp, .stApp * {
    color: #ffffff !important;
}

/* INPUT FIX */
.stTextInput > div > div > input {
    background-color: #1e1e2f !important;
    color: #ffffff !important;
    border: 1px solid #444 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 15px !important;
    caret-color: #ffffff !important;
    /* Mobile: prevent zoom on focus (iOS requires font-size >= 16px) */
    font-size: 16px !important;
}
.stTextInput input::placeholder {
    color: #aaaaaa !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    /* Touch-friendly tap target */
    min-height: 44px !important;
    border-radius: 10px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
}

/* FILE UPLOADER TEXT VISIBILITY */
[data-testid="stFileUploader"] {
    background: #1e1e2f !important;
    border: 2px dashed #667eea !important;
    border-radius: 12px !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"] div {
    color: #000000 !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] small {
    color: #000000 !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] section {
    background: #ffffff !important;
    color: #000000 !important;
}
[data-testid="stFileUploader"] button {
    color: #ffffff !important;
    background: rgba(102,126,234,0.2) !important;
    border: 1px solid rgba(102,126,234,0.5) !important;
}

/* Chat answer box */
div[style*="background:rgba(255,255,255,0.07)"] {
    background: #2a2a40 !important;
    color: #ffffff !important;
}
div[style*="linear-gradient"] {
    color: #ffffff !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
}

/* Messages */
.stSuccess {
    background: rgba(72,187,120,0.15) !important;
    color: #9ae6b4 !important;
}
.stError {
    background: rgba(245,101,101,0.15) !important;
    color: #feb2b2 !important;
}
.stWarning {
    background: rgba(237,137,54,0.15) !important;
    color: #fbd38d !important;
}

/* Quick action buttons */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(102,126,234,0.15) !important;
    border: 1px solid rgba(102,126,234,0.4) !important;
    color: #a3bffa !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 12px !important;
    margin-bottom: 4px !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
    min-height: 44px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(102,126,234,0.35) !important;
    border-color: rgba(102,126,234,0.8) !important;
    color: #ffffff !important;
    transform: translateX(3px) !important;
}

/* ── DESKTOP padding ── */
.block-container {
    padding: 2rem !important;
    max-width: 100% !important;
}

/* SELECTBOX FIX */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #667eea !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #000000 !important;
}
[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] [role="option"] {
    color: #000000 !important;
    background-color: #ffffff !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] [role="option"]:hover {
    background-color: #e8e8f0 !important;
    color: #000000 !important;
}

/* ════════════════════════════════════════
   MOBILE RESPONSIVE STYLES
   ════════════════════════════════════════ */

@media (max-width: 768px) {

    /* KEY FIX: sidebar becomes a floating overlay on mobile,
       so it never shrinks the main content area */
    [data-testid="stSidebar"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        height: 100dvh !important;
        z-index: 9999 !important;
        min-width: 82vw !important;
        max-width: 82vw !important;
        box-shadow: 6px 0 24px rgba(0,0,0,0.6) !important;
        overflow-y: auto !important;
        transition: transform 0.25s ease !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1rem !important;
        min-width: 82vw !important;
        width: 82vw !important;
    }

    /* Main content always full width on mobile */
    .main {
        margin-left: 0 !important;
        padding-left: 0 !important;
        width: 100vw !important;
        max-width: 100vw !important;
    }

    .main .block-container {
        padding: 1rem 0.75rem !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    /* Hide the sidebar collapse arrow - use hamburger instead */
    [data-testid="collapsedControl"] {
        display: block !important;
        z-index: 10000 !important;
    }

    h1 { font-size: 1.3rem !important; }

    /* Chat bubbles */
    div[style*="max-width:72%"] {
        max-width: 90% !important;
        font-size: 13px !important;
        word-break: break-word !important;
    }
    div[style*="max-width:78%"] {
        max-width: 94% !important;
        font-size: 13px !important;
        word-break: break-word !important;
    }

    /* All buttons full width */
    .stButton > button {
        width: 100% !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
        min-height: 48px !important;
    }

    /* Stat boxes stack vertically on mobile */
    div[style*="display:flex; gap:12px"] {
        flex-direction: column !important;
        gap: 8px !important;
    }

    /* Prevent iOS zoom on input focus */
    .stTextInput > div > div > input {
        font-size: 16px !important;
    }

    [data-testid="stFileUploader"] {
        padding: 12px !important;
    }
}

/* Small phones */
@media (max-width: 480px) {
    .main .block-container {
        padding: 0.5rem 0.4rem !important;
    }
    h1 { font-size: 1.1rem !important; }

    span[style*="border-radius:20px"] {
        font-size: 10px !important;
        padding: 2px 8px !important;
    }
}

/* Touch devices */
@media (hover: none) and (pointer: coarse) {
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: none !important;
    }
    .stButton > button {
        min-height: 48px !important;
    }
    select, input, textarea {
        font-size: 16px !important;
    }
}

</style>
""", unsafe_allow_html=True)

# ── User accounts ────────────────────────────────────────────
USERS = {
    "admin": "admin123",
    "demo":  "demo123",
    "user1": "pass123",
}

# ── UI Translations ───────────────────────────────────────────
UI_TEXT = {
    "English": {
        "signed_in_as": "Signed in as",
        "response_language": "🌐 RESPONSE LANGUAGE",
        "upload_documents": "📂 UPLOAD DOCUMENTS",
        "file_ready": "file(s) ready",
        "process_pdfs": "⚙️ Process PDFs",
        "index_stats": "INDEX STATS",
        "chunks": "chunks",
        "docs": "docs",
        "quick_actions": "⚡ QUICK ACTIONS",
        "sign_out": "🚪 Sign Out",
        "page_title": "Ask your documents",
        "page_subtitle": "Upload PDFs on the left → Process → Ask anything below",
        "ask_placeholder": "💬  Ask anything about your documents...",
        "ask_btn": "Ask →",
        "clear_chat": "🗑️ Clear chat history",
        "no_docs": "Upload and process PDFs from the sidebar to begin",
        "searching": "🔍 Searching documents...",
        "max_pdf_warning": "Max 10 PDFs. Only first 10 will be used.",
        "indexed_success": "chunks indexed!",
        "starting": "Starting...",
        "reading": "Reading",
        "generating_embeddings": "Generating embeddings...",
        "building_index": "Building index...",
        "done": "Done!",
        "quick_actions_list": [
            ("🔑 Key Concepts",   "What are the key concepts and important topics in these documents?"),
            ("📝 Summary",        "Give me a detailed summary of all the documents."),
            ("❓ Important Q&A",  "Generate important questions and answers from these documents."),
            ("📋 MCQ Quiz",       "Create 5 multiple choice quiz questions with 4 options each from the documents."),
            ("📌 Key Terms",      "List all important keywords and technical terms with their definitions."),
            ("🗂️ Topics List",    "What are all the main topics covered across these documents?"),
        ],
        "login_subtitle": "Your intelligent PDF assistant",
        "username_label": "Username",
        "password_label": "Password",
        "username_placeholder": "e.g. admin",
        "password_placeholder": "Enter your password",
        "sign_in_btn": "Sign In →",
        "wrong_credentials": "❌ Wrong username or password. Try: admin / admin123",
    },
    "Hindi": {
        "signed_in_as": "साइन इन किया है",
        "response_language": "🌐 उत्तर की भाषा",
        "upload_documents": "📂 दस्तावेज़ अपलोड करें",
        "file_ready": "फ़ाइल(ें) तैयार हैं",
        "process_pdfs": "⚙️ PDF प्रोसेस करें",
        "index_stats": "इंडेक्स आँकड़े",
        "chunks": "खंड",
        "docs": "दस्तावेज़",
        "quick_actions": "⚡ त्वरित क्रियाएं",
        "sign_out": "🚪 साइन आउट",
        "page_title": "अपने दस्तावेज़ों से पूछें",
        "page_subtitle": "बाईं ओर PDF अपलोड करें → प्रोसेस करें → नीचे कुछ भी पूछें",
        "ask_placeholder": "💬  अपने दस्तावेज़ों के बारे में कुछ भी पूछें...",
        "ask_btn": "पूछें →",
        "clear_chat": "🗑️ चैट इतिहास साफ़ करें",
        "no_docs": "शुरू करने के लिए साइडबार से PDF अपलोड और प्रोसेस करें",
        "searching": "🔍 दस्तावेज़ खोज रहे हैं...",
        "max_pdf_warning": "अधिकतम 10 PDF। केवल पहली 10 उपयोग की जाएंगी।",
        "indexed_success": "खंड इंडेक्स किए गए!",
        "starting": "शुरू हो रहा है...",
        "reading": "पढ़ रहे हैं",
        "generating_embeddings": "एम्बेडिंग बना रहे हैं...",
        "building_index": "इंडेक्स बना रहे हैं...",
        "done": "पूरा हुआ!",
        "quick_actions_list": [
            ("🔑 मुख्य अवधारणाएं",   "इन दस्तावेज़ों में मुख्य अवधारणाएं और महत्वपूर्ण विषय क्या हैं?"),
            ("📝 सारांश",             "सभी दस्तावेज़ों का विस्तृत सारांश दें।"),
            ("❓ महत्वपूर्ण प्रश्नोत्तर", "इन दस्तावेज़ों से महत्वपूर्ण प्रश्न और उत्तर बनाएं।"),
            ("📋 MCQ प्रश्नोत्तरी",   "दस्तावेज़ों से 4 विकल्पों वाले 5 बहुविकल्पीय प्रश्न बनाएं।"),
            ("📌 मुख्य शब्द",         "सभी महत्वपूर्ण कीवर्ड और तकनीकी शब्दों को उनकी परिभाषाओं के साथ सूचीबद्ध करें।"),
            ("🗂️ विषय सूची",         "इन दस्तावेज़ों में कौन से मुख्य विषय हैं?"),
        ],
        "login_subtitle": "आपका बुद्धिमान PDF सहायक",
        "username_label": "उपयोगकर्ता नाम",
        "password_label": "पासवर्ड",
        "username_placeholder": "जैसे admin",
        "password_placeholder": "अपना पासवर्ड दर्ज करें",
        "sign_in_btn": "साइन इन करें →",
        "wrong_credentials": "❌ गलत उपयोगकर्ता नाम या पासवर्ड। प्रयास करें: admin / admin123",
    },
    "Kannada": {
        "signed_in_as": "ಸೈನ್ ಇನ್ ಆಗಿದ್ದಾರೆ",
        "response_language": "🌐 ಉತ್ತರದ ಭಾಷೆ",
        "upload_documents": "📂 ದಾಖಲೆಗಳನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ",
        "file_ready": "ಫೈಲ್(ಗಳು) ಸಿದ್ಧವಾಗಿವೆ",
        "process_pdfs": "⚙️ PDF ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಿ",
        "index_stats": "ಇಂಡೆಕ್ಸ್ ಅಂಕಿಅಂಶಗಳು",
        "chunks": "ತುಣುಕುಗಳು",
        "docs": "ದಾಖಲೆಗಳು",
        "quick_actions": "⚡ ತ್ವರಿತ ಕ್ರಿಯೆಗಳು",
        "sign_out": "🚪 ಸೈನ್ ಔಟ್",
        "page_title": "ನಿಮ್ಮ ದಾಖಲೆಗಳನ್ನು ಕೇಳಿ",
        "page_subtitle": "ಎಡಭಾಗದಲ್ಲಿ PDF ಅಪ್ಲೋಡ್ ಮಾಡಿ → ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಿ → ಕೆಳಗೆ ಯಾವುದಾದರೂ ಕೇಳಿ",
        "ask_placeholder": "💬  ನಿಮ್ಮ ದಾಖಲೆಗಳ ಬಗ್ಗೆ ಏನಾದರೂ ಕೇಳಿ...",
        "ask_btn": "ಕೇಳಿ →",
        "clear_chat": "🗑️ ಚಾಟ್ ಇತಿಹಾಸ ತೆರವುಗೊಳಿಸಿ",
        "no_docs": "ಪ್ರಾರಂಭಿಸಲು ಸೈಡ್‌ಬಾರ್‌ನಿಂದ PDF ಅಪ್ಲೋಡ್ ಮಾಡಿ ಮತ್ತು ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಿ",
        "searching": "🔍 ದಾಖಲೆಗಳನ್ನು ಹುಡುಕುತ್ತಿದ್ದೇವೆ...",
        "max_pdf_warning": "ಗರಿಷ್ಠ 10 PDF. ಮೊದಲ 10 ಮಾತ್ರ ಬಳಸಲಾಗುತ್ತದೆ.",
        "indexed_success": "ತುಣುಕುಗಳನ್ನು ಇಂಡೆಕ್ಸ್ ಮಾಡಲಾಗಿದೆ!",
        "starting": "ಪ್ರಾರಂಭವಾಗುತ್ತಿದೆ...",
        "reading": "ಓದುತ್ತಿದ್ದೇವೆ",
        "generating_embeddings": "ಎಂಬೆಡಿಂಗ್‌ಗಳನ್ನು ರಚಿಸುತ್ತಿದ್ದೇವೆ...",
        "building_index": "ಇಂಡೆಕ್ಸ್ ನಿರ್ಮಿಸುತ್ತಿದ್ದೇವೆ...",
        "done": "ಮುಗಿಯಿತು!",
        "quick_actions_list": [
            ("🔑 ಪ್ರಮುಖ ಪರಿಕಲ್ಪನೆಗಳು",  "ಈ ದಾಖಲೆಗಳಲ್ಲಿ ಪ್ರಮುಖ ಪರಿಕಲ್ಪನೆಗಳು ಮತ್ತು ವಿಷಯಗಳು ಯಾವುವು?"),
            ("📝 ಸಾರಾಂಶ",                "ಎಲ್ಲಾ ದಾಖಲೆಗಳ ವಿವರವಾದ ಸಾರಾಂಶ ನೀಡಿ."),
            ("❓ ಮಹತ್ವದ ಪ್ರಶ್ನೋತ್ತರ",   "ಈ ದಾಖಲೆಗಳಿಂದ ಮಹತ್ವದ ಪ್ರಶ್ನೆಗಳು ಮತ್ತು ಉತ್ತರಗಳನ್ನು ರಚಿಸಿ."),
            ("📋 MCQ ರಸಪ್ರಶ್ನೆ",         "ದಾಖಲೆಗಳಿಂದ 4 ಆಯ್ಕೆಗಳೊಂದಿಗೆ 5 ಬಹು ಆಯ್ಕೆ ಪ್ರಶ್ನೆಗಳನ್ನು ರಚಿಸಿ."),
            ("📌 ಪ್ರಮುಖ ಪದಗಳು",          "ಎಲ್ಲಾ ಮಹತ್ವದ ಕೀವರ್ಡ್‌ಗಳು ಮತ್ತು ತಾಂತ್ರಿಕ ಪದಗಳನ್ನು ಅವುಗಳ ವ್ಯಾಖ್ಯಾನದೊಂದಿಗೆ ಪಟ್ಟಿ ಮಾಡಿ."),
            ("🗂️ ವಿಷಯ ಪಟ್ಟಿ",           "ಈ ದಾಖಲೆಗಳಲ್ಲಿ ಯಾವ ಮುಖ್ಯ ವಿಷಯಗಳಿವೆ?"),
        ],
        "login_subtitle": "ನಿಮ್ಮ ಬುದ್ಧಿವಂತ PDF ಸಹಾಯಕ",
        "username_label": "ಬಳಕೆದಾರ ಹೆಸರು",
        "password_label": "ಪಾಸ್‌ವರ್ಡ್",
        "username_placeholder": "ಉದಾ. admin",
        "password_placeholder": "ನಿಮ್ಮ ಪಾಸ್‌ವರ್ಡ್ ನಮೂದಿಸಿ",
        "sign_in_btn": "ಸೈನ್ ಇನ್ ಮಾಡಿ →",
        "wrong_credentials": "❌ ತಪ್ಪು ಬಳಕೆದಾರ ಹೆಸರು ಅಥವಾ ಪಾಸ್‌ವರ್ಡ್. ಪ್ರಯತ್ನಿಸಿ: admin / admin123",
    },
    "Tamil": {
        "signed_in_as": "உள்நுழைந்தீர்கள்",
        "response_language": "🌐 பதில் மொழி",
        "upload_documents": "📂 ஆவணங்களை பதிவேற்றவும்",
        "file_ready": "கோப்பு(கள்) தயாராக உள்ளன",
        "process_pdfs": "⚙️ PDF செயலாக்கவும்",
        "index_stats": "குறியீட்டு புள்ளிவிவரங்கள்",
        "chunks": "பகுதிகள்",
        "docs": "ஆவணங்கள்",
        "quick_actions": "⚡ விரைவு நடவடிக்கைகள்",
        "sign_out": "🚪 வெளியேறு",
        "page_title": "உங்கள் ஆவணங்களிடம் கேளுங்கள்",
        "page_subtitle": "இடதுபுறத்தில் PDF பதிவேற்றவும் → செயலாக்கவும் → கீழே எதையும் கேளுங்கள்",
        "ask_placeholder": "💬  உங்கள் ஆவணங்களைப் பற்றி எதையும் கேளுங்கள்...",
        "ask_btn": "கேளுங்கள் →",
        "clear_chat": "🗑️ அரட்டை வரலாற்றை அழிக்கவும்",
        "no_docs": "தொடங்க பக்கப்பட்டியிலிருந்து PDF பதிவேற்றி செயலாக்கவும்",
        "searching": "🔍 ஆவணங்களை தேடுகிறோம்...",
        "max_pdf_warning": "அதிகபட்சம் 10 PDF. முதல் 10 மட்டுமே பயன்படுத்தப்படும்.",
        "indexed_success": "பகுதிகள் குறியிடப்பட்டன!",
        "starting": "தொடங்குகிறது...",
        "reading": "படிக்கிறோம்",
        "generating_embeddings": "உட்பொதிப்புகளை உருவாக்குகிறோம்...",
        "building_index": "குறியீட்டை உருவாக்குகிறோம்...",
        "done": "முடிந்தது!",
        "quick_actions_list": [
            ("🔑 முக்கிய கருத்துகள்",  "இந்த ஆவணங்களில் முக்கிய கருத்துகள் என்ன?"),
            ("📝 சுருக்கம்",            "அனைத்து ஆவணங்களின் விரிவான சுருக்கம் தரவும்."),
            ("❓ முக்கிய கேள்வி-பதில்", "இந்த ஆவணங்களிலிருந்து முக்கிய கேள்விகளும் பதில்களும் உருவாக்கவும்."),
            ("📋 MCQ வினாடி வினா",      "ஆவணங்களிலிருந்து 4 விருப்பங்களுடன் 5 பல தேர்வு கேள்விகள் உருவாக்கவும்."),
            ("📌 முக்கிய சொற்கள்",     "அனைத்து முக்கிய சொற்களையும் அவற்றின் வரையறைகளுடன் பட்டியலிடவும்."),
            ("🗂️ தலைப்பு பட்டியல்",   "இந்த ஆவணங்களில் என்ன முக்கிய தலைப்புகள் உள்ளன?"),
        ],
        "login_subtitle": "உங்கள் புத்திசாலி PDF உதவியாளர்",
        "username_label": "பயனர் பெயர்",
        "password_label": "கடவுச்சொல்",
        "username_placeholder": "எ.கா. admin",
        "password_placeholder": "உங்கள் கடவுச்சொல்லை உள்ளிடவும்",
        "sign_in_btn": "உள்நுழைக →",
        "wrong_credentials": "❌ தவறான பயனர் பெயர் அல்லது கடவுச்சொல். முயற்சிக்கவும்: admin / admin123",
    },
}

def t(key):
    """Get UI text for the current language, fallback to English."""
    lang = st.session_state.get("language", "English")
    lang_dict = UI_TEXT.get(lang, UI_TEXT["English"])
    return lang_dict.get(key, UI_TEXT["English"].get(key, key))

# ── Session state ─────────────────────────────────────────────
if "logged_in"    not in st.session_state: st.session_state.logged_in    = False
if "username"     not in st.session_state: st.session_state.username     = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "language"     not in st.session_state: st.session_state.language     = "English"
# ════════════════════════════════════════════════
# LOGIN PAGE
# ════════════════════════════════════════════════
def show_login():
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        # Logo + title
        st.markdown(f"""
        <div style='text-align:center; margin-bottom:2rem;'>
            <div style='font-size:52px; margin-bottom:12px;'>🧠</div>
            <h1 style='color:white; font-size:2rem; font-weight:700; margin:0;'>DocuMind AI</h1>
            <p style='color:#718096; font-size:14px; margin-top:8px;'>{t('login_subtitle')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Login form
        with st.form("login_form", clear_on_submit=False):
            st.markdown(f"<p style='color:#a0aec0; font-size:13px; margin:0 0 6px;'>{t('username_label')}</p>",
                        unsafe_allow_html=True)
            username = st.text_input(
                "username_field",
                placeholder=t("username_placeholder"),
                label_visibility="collapsed"
            )

            st.markdown(f"<p style='color:#a0aec0; font-size:13px; margin:16px 0 6px;'>{t('password_label')}</p>",
                        unsafe_allow_html=True)
            password = st.text_input(
                "password_field",
                placeholder=t("password_placeholder"),
                type="password",
                label_visibility="collapsed"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(t("sign_in_btn"))

            if submitted:
                if username.strip() in USERS and USERS[username.strip()] == password:
                    st.session_state.logged_in = True
                    st.session_state.username  = username.strip()
                    st.rerun()
                else:
                    st.error(t("wrong_credentials"))

        st.markdown("""
        <div style='text-align:center; margin-top:1.5rem;
                    padding:12px; background:rgba(255,255,255,0.04);
                    border-radius:12px; border:1px solid rgba(255,255,255,0.08);'>
            <p style='color:#718096; font-size:12px; margin:0;'>
                Demo → <span style='color:#a3bffa;'>admin</span> /
                <span style='color:#a3bffa;'>admin123</span>
                &nbsp;|&nbsp;
                <span style='color:#a3bffa;'>demo</span> /
                <span style='color:#a3bffa;'>demo123</span>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════
def show_app():

    # ── Sidebar ──────────────────────────────────
    with st.sidebar:

        # Branding
        st.markdown(f"""
        <div style='padding:0.5rem 0 1.5rem;'>
            <div style='font-size:32px; margin-bottom:8px;'>🧠</div>
            <div style='font-size:1.2rem; font-weight:700;
                        color:white; margin-bottom:4px;'>DocuMind AI</div>
            <div style='font-size:12px; color:#718096;'>
                {t('signed_in_as')}&nbsp;
                <span style='color:#a3bffa; font-weight:600;'>
                    {st.session_state.username}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Language selector
        st.markdown(f"""
        <p style='color:#a0aec0; font-size:11px; font-weight:600;
                  letter-spacing:1.5px; margin-bottom:8px;'>
            {t('response_language')}
        </p>
        """, unsafe_allow_html=True)

        LANGUAGES = {
            "English":    "🇬🇧 English",
            "Hindi":      "🇮🇳 Hindi",
            "Tamil":      "🇮🇳 Tamil",
            "Telugu":     "🇮🇳 Telugu",
            "Kannada":    "🇮🇳 Kannada",
            "Malayalam":  "🇮🇳 Malayalam",
            "Bengali":    "🇮🇳 Bengali",
            "Marathi":    "🇮🇳 Marathi",
            "French":     "🇫🇷 French",
            "Spanish":    "🇪🇸 Spanish",
            "German":     "🇩🇪 German",
            "Arabic":     "🇸🇦 Arabic",
            "Chinese":    "🇨🇳 Chinese",
            "Japanese":   "🇯🇵 Japanese",
        }

        selected_lang = st.selectbox(
            "language_select",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x],
            index=list(LANGUAGES.keys()).index(st.session_state.language),
            label_visibility="collapsed"
        )
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang

        st.markdown("---")

        # Upload section
        st.markdown(f"""
        <p style='color:#a0aec0; font-size:11px; font-weight:600;
                  letter-spacing:1.5px; margin-bottom:10px;'>
            {t('upload_documents')}
        </p>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            if len(uploaded_files) > 10:
                st.warning(t("max_pdf_warning"))
                uploaded_files = uploaded_files[:10]

            st.markdown(f"""
            <div style='background:rgba(102,126,234,0.15); border:1px solid rgba(102,126,234,0.3);
                        border-radius:10px; padding:8px 12px; margin:8px 0;
                        font-size:13px; color:#a3bffa;'>
                📄 {len(uploaded_files)} {t('file_ready')}
            </div>
            """, unsafe_allow_html=True)

            if st.button(t("process_pdfs")):
                all_chunks = []
                bar = st.progress(0, text=t("starting"))

                for i, file in enumerate(uploaded_files):
                    bar.progress(
                        int((i / len(uploaded_files)) * 80),
                        text=f"{t('reading')} {file.name}..."
                    )
                    pages  = extract_text_from_pdf(file, file.name)
                    chunks = chunk_pages(pages)
                    all_chunks.extend(chunks)

                bar.progress(85, text=t("generating_embeddings"))
                texts      = [c["text"] for c in all_chunks]
                embeddings = get_embeddings(texts)

                bar.progress(95, text=t("building_index"))
                index, chunks = build_index(embeddings, all_chunks)
                st.session_state["index"]  = index
                st.session_state["chunks"] = chunks

                bar.progress(100, text=t("done"))
                st.success(f"✅ {len(all_chunks)} {t('indexed_success')}")

        # Stats panel
        if "chunks" in st.session_state:
            st.markdown("---")
            c = st.session_state["chunks"]
            files = list(set(x["source"] for x in c))
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.04); border-radius:12px;
                        padding:14px; border:1px solid rgba(255,255,255,0.07);'>
                <p style='color:#718096; font-size:10px; letter-spacing:1.5px; margin:0 0 10px;'>
                    {t('index_stats')}
                </p>
                <div style='display:flex; gap:12px;'>
                    <div style='flex:1; text-align:center;'>
                        <div style='font-size:24px; font-weight:700; color:white;'>{len(c)}</div>
                        <div style='font-size:11px; color:#718096;'>{t('chunks')}</div>
                    </div>
                    <div style='width:1px; background:rgba(255,255,255,0.08);'></div>
                    <div style='flex:1; text-align:center;'>
                        <div style='font-size:24px; font-weight:700; color:white;'>{len(files)}</div>
                        <div style='font-size:11px; color:#718096;'>{t('docs')}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Quick Action Buttons ──────────────────────────
            st.markdown(f"""
            <p style='color:#a0aec0; font-size:11px; font-weight:600;
                      letter-spacing:1.5px; margin:12px 0 8px;'>
                {t('quick_actions')}
            </p>
            """, unsafe_allow_html=True)

            for label, query in t("quick_actions_list"):
                if st.button(label, key=f"quick_{label}", use_container_width=True):
                    st.session_state["quick_query"] = query
                    st.rerun()

        st.markdown("---")
        if st.button(t("sign_out")):
            for key in ["logged_in", "username", "chat_history", "index", "chunks"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # ── Main area ─────────────────────────────────
    st.markdown(f"""
    <h1 style='color:white; font-size:2rem; font-weight:700; margin:0 0 6px;'>
        {t('page_title')}
    </h1>
    <p style='color:#718096; font-size:14px; margin:0 0 2rem;'>
        {t('page_subtitle')}
    </p>
    """, unsafe_allow_html=True)

    # Chat history display
    for chat in st.session_state.chat_history:
        # User bubble
        st.markdown(f"""
        <div style='display:flex; justify-content:flex-end; margin-bottom:10px;'>
            <div style='background:linear-gradient(135deg,#667eea,#764ba2);
                        color:white; border-radius:18px 18px 4px 18px;
                        padding:12px 18px; max-width:72%;
                        font-size:14px; line-height:1.6;'>
                {chat["question"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Sources tags
        src_html = " ".join([
            f"<span style='display:inline-block; background:rgba(118,75,162,0.3);"
            f"border:1px solid rgba(118,75,162,0.5); border-radius:20px;"
            f"padding:3px 12px; font-size:11px; color:#d6bcfa; margin:2px;'>"
            f"📄 {s['file']} · p.{s['page']}</span>"
            for s in chat["sources"]
        ])

        # Language badge
        lang = chat.get("language", "English")
        lang_badge = f"<span style='display:inline-block; background:rgba(56,189,150,0.15); border:1px solid rgba(56,189,150,0.4); border-radius:20px; padding:2px 10px; font-size:10px; color:#68d9b3; margin-bottom:6px;'>🌐 {lang}</span>"

        # AI bubble
        st.markdown(f"""
        <div style='display:flex; justify-content:flex-start; margin-bottom:20px;'>
            <div style='background:rgba(255,255,255,0.07);
                        border:1px solid rgba(255,255,255,0.1);
                        color:#e2e8f0; border-radius:18px 18px 18px 4px;
                        padding:14px 18px; max-width:78%;
                        font-size:14px; line-height:1.8;'>
                <div style='margin-bottom:8px;'>{lang_badge}</div>
                {chat["answer"]}
                <div style='margin-top:10px;'>{src_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Input area ──
    index  = st.session_state.get("index")
    chunks = st.session_state.get("chunks")
    if index is None:
        index, chunks = load_index()

    if index is None:
        st.markdown("""
        <div style='text-align:center; padding:5rem 2rem;
                    background:rgba(255,255,255,0.03);
                    border:2px dashed rgba(255,255,255,0.08);
                    border-radius:20px; margin-top:2rem;'>
            <div style='font-size:48px; margin-bottom:16px;'>📂</div>
            <p style='color:#718096; font-size:16px; margin:0;'>
                {t('no_docs')}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Check if a quick action button was clicked
        quick_query = st.session_state.pop("quick_query", None)

        col_input, col_btn = st.columns([5, 1])
        with col_input:
            question = st.text_input(
                "question",
                placeholder=t("ask_placeholder"),
                label_visibility="collapsed",
                key="q_input",
                value=quick_query if quick_query else ""
            )
        with col_btn:
            ask = st.button(t("ask_btn"), key="ask_btn")

        # Auto-run if quick action triggered, or manual ask
        run_query = quick_query or (ask and question.strip())
        final_question = quick_query if quick_query else question.strip()

        if run_query and final_question:
            with st.spinner(t("searching")):
                q_emb   = get_embeddings([final_question])[0]
                ctx     = search(q_emb, index, chunks, top_k=5)
                result  = answer_question(final_question, ctx, language=st.session_state.language)

            st.session_state.chat_history.append({
                "question": final_question,
                "answer":   result["answer"],
                "sources":  result["sources"],
                "language": st.session_state.language
            })
            st.rerun()

        if st.session_state.chat_history:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(t("clear_chat")):
                st.session_state.chat_history = []
                st.rerun()


# ── Router ────────────────────────────────────────────────────
if not st.session_state.logged_in:
    show_login()
else:
    show_app()