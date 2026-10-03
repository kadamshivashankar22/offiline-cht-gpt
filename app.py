import os
import tempfile
import uuid
import html
import hashlib
import re
import shutil
import sys
from pathlib import Path

import streamlit as st
from PIL import Image
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import GPT4All
from langchain_community.vectorstores import Chroma

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Offline Intelligent AI Assistant", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --app-bg: #0f172a;
        --panel-bg: #111827;
        --message-user: #1d4ed8;
        --message-bot: #1f2937;
        --text-main: #e5e7eb;
        --text-muted: #9ca3af;
        --border: #374151;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 15%, #1e293b 0%, transparent 40%),
            radial-gradient(circle at 85% 10%, #0b3b66 0%, transparent 30%),
            var(--app-bg);
        color: var(--text-main);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1050px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
        border-right: 1px solid var(--border);
    }

    .app-header {
        margin-bottom: 0.75rem;
        padding: 1rem 1.1rem;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: linear-gradient(120deg, #0f1f3a 0%, #102a43 50%, #143b5f 100%);
    }

    .app-title {
        margin: 0;
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .app-subtitle {
        margin: 0.25rem 0 0 0;
        font-size: 0.92rem;
        color: #cbd5e1;
    }

    .chat-wrap {
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(4px);
    }

    .msg-row {
        display: flex;
        margin: 0.55rem 0;
    }

    .msg-row.user {
        justify-content: flex-end;
    }

    .msg-row.assistant {
        justify-content: flex-start;
    }

    .msg-bubble {
        max-width: 78%;
        padding: 0.65rem 0.85rem;
        border-radius: 14px;
        line-height: 1.45;
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    .msg-bubble.user {
        background: linear-gradient(120deg, #1e3a8a 0%, #1d4ed8 100%);
        border: 1px solid #3b82f6;
        color: #f8fafc;
        border-bottom-right-radius: 6px;
    }

    .msg-bubble.assistant {
        background: var(--message-bot);
        border: 1px solid #4b5563;
        color: #e5e7eb;
        border-bottom-left-radius: 6px;
    }

    .composer-wrap {
        margin-top: 0.8rem;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.7rem 0.8rem 0.1rem 0.8rem;
        background: rgba(17, 24, 39, 0.88);
    }

    .stChatInput {
        margin-top: 0.25rem;
    }

    .upload-note {
        font-size: 0.83rem;
        color: var(--text-muted);
        margin-bottom: 0.45rem;
    }

    .side-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        margin-bottom: 0.85rem;
        border-radius: 14px;
        border: 1px solid #334155;
        background: linear-gradient(135deg, #0f223d 0%, #123056 55%, #164e7a 100%);
        box-shadow: 0 10px 24px rgba(2, 12, 27, 0.35);
    }

    .side-brand-logo {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        border: 1px solid #3b82f6;
        background: radial-gradient(circle at 30% 30%, #1d4ed8, #1e3a8a 75%);
    }

    .side-brand-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.1;
    }

    .side-brand-sub {
        font-size: 0.78rem;
        color: #bfdbfe;
        margin-top: 0.18rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">Offline Intelligent AI Assistant</h1>
        <p class="app-subtitle">Chat naturally with your local model and uploaded files.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Keep performance configuration fixed and always-on for demo stability.
turbo_mode = True
ocr_enabled = True
instant_code_mode = True
use_uploaded_context = True
judge_mode = True


# ---------------- LLM LOADING ----------------
@st.cache_resource
def setup_cuda_runtime_paths():
    """Add NVIDIA pip runtime DLL folders to PATH for GPT4All CUDA backend."""
    try:
        site_packages = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
        nvidia_root = site_packages / "nvidia"
        if not nvidia_root.exists():
            return False

        updated = False
        for bin_dir in nvidia_root.glob("*/bin"):
            p = str(bin_dir)
            if not bin_dir.exists():
                continue
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
                updated = True
        return updated
    except Exception:
        return False


@st.cache_resource
def load_llm():
    setup_cuda_runtime_paths()
    model_path = "C:/Users/munde/OneDrive/Desktop/minichatgpt/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    cpu_threads = os.cpu_count() or 8
    tuned_threads = max(4, min(8, cpu_threads))
    params = dict(
        model=model_path,
        n_threads=tuned_threads,
        n_batch=256,
        temp=0.08,
        top_k=20,
        top_p=0.9,
        max_tokens=160,
        n_predict=160,
        verbose=False,
    )
    try:
        return GPT4All(device="cuda", **params)
    except Exception:
        return GPT4All(device="cpu", **params)


@st.cache_resource
def load_llm_full():
    return load_llm()


@st.cache_resource
def load_llm_ultra():
    return load_llm()


@st.cache_resource
def load_llm_micro():
    return load_llm()


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


llm = load_llm()
llm_full = load_llm_full()
llm_ultra = load_llm_ultra()
llm_micro = load_llm_micro()
PROMPT_PROFILE_VERSION = "M7B_LANG_SYNC_V7"


@st.cache_data(show_spinner=False, ttl=1800, max_entries=400)
def cached_infer(prompt, mode):
    if mode == "full":
        return load_llm_full().invoke(prompt).strip()
    if mode == "micro":
        return load_llm_micro().invoke(prompt).strip()
    if mode == "ultra":
        return load_llm_ultra().invoke(prompt).strip()
    return load_llm().invoke(prompt).strip()


@st.cache_resource
def warmup_llm():
    try:
        load_llm_micro().invoke("User: hi\nAssistant:")
    except Exception:
        pass
    return True


_ = warmup_llm()


@st.cache_resource
def load_offline_image_generator(model_source):
    try:
        import torch
        from diffusers import DiffusionPipeline
    except Exception as e:
        raise RuntimeError(
            "Image generation dependencies are missing. "
            f'Install with: "{sys.executable}" -m pip install diffusers accelerate safetensors'
        ) from e

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_source,
            torch_dtype=dtype,
            local_files_only=True,
        )
    except Exception as e:
        raise RuntimeError(
            "Offline image model is not available locally.\n"
            "Download once (with internet), then run fully offline.\n"
            f"Model source tried: {model_source}\n"
            "Tip: pre-download `stabilityai/sd-turbo` or set a local folder path.\n"
            f"Loader details: {type(e).__name__}: {e}"
        ) from e

    try:
        pipe = pipe.to(device)
    except Exception:
        pipe = pipe.to("cpu")
        device = "cpu"

    # Speed-focused options
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    try:
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
    except Exception:
        pass

    if device == "cuda":
        # Throughput-oriented GPU settings.
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    else:
        # CPU fallback stability.
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    return pipe, device


def generate_image_offline(prompt, model_source, steps=2, guidance=0.0, width=512, height=512):
    import torch

    pipe, device = load_offline_image_generator(model_source)
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "width": int(width),
        "height": int(height),
    }
    if device == "cuda":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            image = pipe(**kwargs).images[0]
    else:
        with torch.inference_mode():
            image = pipe(**kwargs).images[0]
    return image, device


@st.cache_resource
def load_image_captioner():
    try:
        from transformers import pipeline
    except Exception:
        return None

    try:
        # Force offline usage from local cache.
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    except Exception:
        return None


# ---------------- HELPER FUNCTION ----------------
SCRIPT_RANGES = {
    "devanagari": [(0x0900, 0x097F)],
    "bengali": [(0x0980, 0x09FF)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "gujarati": [(0x0A80, 0x0AFF)],
    "odia": [(0x0B00, 0x0B7F)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "cyrillic": [(0x0400, 0x04FF)],
    "thai": [(0x0E00, 0x0E7F)],
    "hiragana": [(0x3040, 0x309F)],
    "katakana": [(0x30A0, 0x30FF)],
    "hangul": [(0xAC00, 0xD7AF)],
    "cjk": [(0x4E00, 0x9FFF)],
}

ROMANIZED_LANGUAGE_HINTS = {
    "hinglish": {
        "aap",
        "aapka",
        "apka",
        "apki",
        "tumhara",
        "tumhari",
        "tera",
        "teri",
        "naam",
        "kaise",
        "kese",
        "kya",
        "mera",
        "mujhe",
        "hum",
        "tum",
        "hai",
        "ho",
        "nahi",
        "haan",
        "madad",
        "chahiye",
        "karna",
        "krna",
        "karo",
        "kro",
        "batao",
        "bolo",
        "samjhao",
        "kaun",
        "kyun",
        "kab",
        "kahan",
        "theek",
        "sahi",
    },
    "tamil_romanized": {
        "enna",
        "epdi",
        "eppadi",
        "nan",
        "ungal",
        "unga",
        "irukku",
        "venum",
        "illai",
        "solunga",
        "saptiya",
        "vanakkam",
        "badhil",
    },
    "telugu_romanized": {
        "nenu",
        "meeru",
        "ela",
        "bagunnara",
        "kavali",
        "ledu",
        "cheppu",
        "em",
        "enduku",
        "chesi",
        "andi",
        "samadhanam",
        "namaskaram",
    },
    "kannada_romanized": {
        "nanu",
        "neevu",
        "hegide",
        "beku",
        "illa",
        "heli",
        "dayavittu",
        "enu",
        "uttara",
    },
    "malayalam_romanized": {
        "njaan",
        "ningal",
        "entha",
        "sukhamano",
        "venam",
        "illa",
        "parayu",
        "cheyyu",
        "utharam",
    },
    "bengali_romanized": {
        "ami",
        "tumi",
        "apni",
        "kemon",
        "bhalo",
        "kore",
        "korbo",
        "amar",
        "tomar",
        "uttor",
        "bolo",
        "dhonnobad",
    },
    "gujarati_romanized": {
        "hu",
        "tame",
        "kem",
        "majama",
        "chhe",
        "mane",
        "joie",
        "nathi",
        "shu",
        "javab",
    },
    "punjabi_romanized": {
        "tusi",
        "kida",
        "ki",
        "mainu",
        "chahida",
        "nahi",
        "haanji",
        "vich",
        "jawab",
    },
    "marathi_romanized": {
        "mi",
        "tumhi",
        "kasa",
        "kaay",
        "aahe",
        "nahi",
        "mala",
        "hava",
        "sanga",
        "uttar",
    },
}

ROMANIZED_LABELS = {
    "hinglish": "Hinglish (Hindi/Urdu in English letters)",
    "tamil_romanized": "romanized Tamil",
    "telugu_romanized": "romanized Telugu",
    "kannada_romanized": "romanized Kannada",
    "malayalam_romanized": "romanized Malayalam",
    "bengali_romanized": "romanized Bengali",
    "gujarati_romanized": "romanized Gujarati",
    "punjabi_romanized": "romanized Punjabi",
    "marathi_romanized": "romanized Marathi",
}


def detect_script_key(text):
    counts = {k: 0 for k in SCRIPT_RANGES}
    latin_count = 0

    for ch in text:
        cp = ord(ch)
        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            latin_count += 1
            continue
        for key, ranges in SCRIPT_RANGES.items():
            if any(start <= cp <= end for start, end in ranges):
                counts[key] += 1
                break

    top_script = max(counts, key=counts.get)
    if counts[top_script] > 0:
        return top_script
    if latin_count > 0:
        return "latin"
    return "unknown"


def detect_romanized_language(text):
    if detect_script_key(text) != "latin":
        return None
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return None

    scores = {
        lang: sum(1 for t in tokens if t in hints)
        for lang, hints in ROMANIZED_LANGUAGE_HINTS.items()
    }
    best_lang = max(scores, key=scores.get)
    best_score = scores[best_lang]
    short_query = len(tokens) <= 6
    if best_score >= 2:
        return best_lang
    if short_query and best_score >= 1:
        return best_lang
    return None


def language_instruction_for(user_input):
    script = detect_script_key(user_input)
    if script == "devanagari":
        return (
            "Reply in the same language and script as the user's latest message "
            "(Devanagari, for example Hindi/Marathi/Nepali)."
        )
    if script in {
        "bengali",
        "gurmukhi",
        "gujarati",
        "odia",
        "tamil",
        "telugu",
        "kannada",
        "malayalam",
    }:
        return "Reply in the same language and Indian script as the user's latest message."
    if script == "latin":
        roman_lang = detect_romanized_language(user_input)
        if roman_lang:
            return (
                f"Reply in {ROMANIZED_LABELS[roman_lang]} using English letters only. "
                "Do not switch to native script and do not switch to pure English."
            )
        return (
            "Reply in the same language as the user's latest message. "
            "If the user writes a non-English language in English letters (transliteration), "
            "reply in that same language style using English letters only."
        )
    return "Reply in the same language as the user's latest message."


def localized_fixed_text(user_input, key):
    script = detect_script_key(user_input)
    roman_lang = detect_romanized_language(user_input) if script == "latin" else None
    table = {
        "doc_not_found": {
            "default": "The answer is not found in the uploaded document.",
            "hinglish": "Uploaded document me iska jawab nahi mila.",
            "tamil_romanized": "Upload panna document la indha kelviku badhil kidaikkala.",
            "telugu_romanized": "Upload chesina document lo ee prashnaki samadhanam dorakaledu.",
            "kannada_romanized": "Upload madida document alli ee prashnege uttara sigalilla.",
            "malayalam_romanized": "Upload cheytha document il ee chodyathinu utharam kittiyilla.",
            "bengali_romanized": "Upload kora document e ei proshner uttor pawa jayni.",
            "gujarati_romanized": "Upload karel document ma aa prashnanu javab malyo nathi.",
            "punjabi_romanized": "Upload kite document vich is sawal da jawab nahi milya.",
            "marathi_romanized": "Upload kelelya document madhe ya prashnache uttar sapadla nahi.",
            "devanagari": "अपलोड किए गए दस्तावेज़ में उत्तर नहीं मिला।",
            "bengali": "আপলোড করা ডকুমেন্টে উত্তরটি পাওয়া যায়নি।",
            "gurmukhi": "ਅਪਲੋਡ ਕੀਤੇ ਦਸਤਾਵੇਜ਼ ਵਿੱਚ ਜਵਾਬ ਨਹੀਂ ਮਿਲਿਆ।",
            "gujarati": "અપલોડ કરેલા દસ્તાવેજમાં જવાબ મળ્યો નથી.",
            "odia": "ଅପଲୋଡ୍ କରାଯାଇଥିବା ଦଲିଲରେ ଉତ୍ତର ମିଳିଲା ନାହିଁ।",
            "tamil": "பதிவேற்றிய ஆவணத்தில் பதில் கிடைக்கவில்லை.",
            "telugu": "అప్‌లోడ్ చేసిన పత్రంలో సమాధానం కనబడలేదు.",
            "kannada": "ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ದಸ್ತಾವೇಜಿನಲ್ಲಿ ಉತ್ತರ ಸಿಗಲಿಲ್ಲ.",
            "malayalam": "അപ്‌ലോഡ് ചെയ്ത ഡോക്യുമെന്റിൽ ഉത്തരം കണ്ടെത്താനായില്ല.",
        },
        "no_relevant_file_info": {
            "default": "I could not find relevant information in the uploaded file.",
            "hinglish": "Uploaded file me mujhe relevant jankari nahi mili.",
            "tamil_romanized": "Upload panna file la poruthamana vivaram kidaikkala.",
            "telugu_romanized": "Upload chesina file lo sambandhita samacharam dorakaledu.",
            "kannada_romanized": "Upload madida file alli sambandhita mahiti sigalilla.",
            "malayalam_romanized": "Upload cheytha file il bandhapetta vivaram kittiyilla.",
            "bengali_romanized": "Upload kora file e prashongik totho pawa jayni.",
            "gujarati_romanized": "Upload karel file ma sambandhit mahiti mali nathi.",
            "punjabi_romanized": "Upload kiti file vich sambandhit jankari nahi mili.",
            "marathi_romanized": "Upload kelelya file madhe sambandhit mahiti sapadli nahi.",
            "devanagari": "अपलोड की गई फ़ाइल में मुझे संबंधित जानकारी नहीं मिली।",
            "bengali": "আপলোড করা ফাইলে প্রাসঙ্গিক তথ্য খুঁজে পাইনি।",
            "gurmukhi": "ਅਪਲੋਡ ਕੀਤੀ ਫਾਈਲ ਵਿੱਚ ਮੈਨੂੰ ਸਬੰਧਤ ਜਾਣਕਾਰੀ ਨਹੀਂ ਮਿਲੀ।",
            "gujarati": "અપલોડ કરેલી ફાઇલમાં મને સંબંધિત માહિતી મળી નથી.",
            "odia": "ଅପଲୋଡ୍ ହୋଇଥିବା ଫାଇଲରେ ସମ୍ବନ୍ଧିତ ସୂଚନା ମିଳିଲା ନାହିଁ।",
            "tamil": "பதிவேற்றிய கோப்பில் தொடர்புடைய தகவலை கண்டுபிடிக்க முடியவில்லை.",
            "telugu": "అప్‌లోడ్ చేసిన ఫైల్‌లో సంబంధిత సమాచారం కనుగొనలేకపోయాను.",
            "kannada": "ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಕಡತದಲ್ಲಿ ಸಂಬಂಧಿತ ಮಾಹಿತಿಯನ್ನು ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ.",
            "malayalam": "അപ്‌ലോഡ് ചെയ്ത ഫയലിൽ ബന്ധപ്പെട്ട വിവരം കണ്ടെത്താനായില്ല.",
        },
        "image_description_prefix": {
            "default": "Image description: ",
            "hinglish": "Image ka description: ",
            "tamil_romanized": "Image vivaram: ",
            "telugu_romanized": "Image vivaranam: ",
            "kannada_romanized": "Image vivarane: ",
            "malayalam_romanized": "Image vivaranam: ",
            "bengali_romanized": "Image er biboron: ",
            "gujarati_romanized": "Image nu varnan: ",
            "punjabi_romanized": "Image da varnan: ",
            "marathi_romanized": "Image che varnan: ",
            "devanagari": "छवि विवरण: ",
            "bengali": "ছবির বিবরণ: ",
            "gurmukhi": "ਚਿੱਤਰ ਵੇਰਵਾ: ",
            "gujarati": "છબી વર્ણન: ",
            "odia": "ଛବି ବିବରଣୀ: ",
            "tamil": "பட விளக்கம்: ",
            "telugu": "చిత్ర వివరణ: ",
            "kannada": "ಚಿತ್ರ ವಿವರಣೆ: ",
            "malayalam": "ചിത്ര വിവരണം: ",
        },
    }
    values = table.get(key, {})
    if roman_lang and roman_lang in values:
        return values[roman_lang]
    return values.get(script, values.get("default", ""))


def instant_smalltalk_reply(user_input):
    q = re.sub(r"\s+", " ", user_input.lower()).strip(" \t\r\n?.!,")
    if not q:
        return None

    script = detect_script_key(user_input)
    roman_lang = detect_romanized_language(user_input) if script == "latin" else None

    greetings = {
        "hi",
        "hii",
        "hiii",
        "hiiii",
        "hello",
        "hey",
        "namaste",
        "namaskaram",
        "vanakkam",
        "salaam",
    }
    if q in greetings:
        if roman_lang == "hinglish":
            return "Hi! Bolo, kis cheez me help chahiye?"
        if script == "devanagari":
            return "नमस्ते! बताइए, आपको किस चीज़ में मदद चाहिए?"
        return "Hi! Tell me what you need help with."

    name_signals = [
        "what is your name",
        "your name",
        "who are you",
        "aapka naam",
        "apka naam",
        "tumhara naam",
        "tera naam",
    ]
    if any(s in q for s in name_signals):
        if roman_lang == "hinglish":
            return "Mera naam Assistant hai. Aap mujhe Assistant bula sakte ho."
        if script == "devanagari":
            return "मेरा नाम Assistant है। आप मुझे Assistant बुला सकते हैं।"
        return "My name is Assistant. You can call me Assistant."

    status_signals = ["how are you", "kaise ho", "kese ho", "kaisa hai", "how r u"]
    if any(s in q for s in status_signals):
        if roman_lang == "hinglish":
            return "Main theek hoon. Aapka next question kya hai?"
        if script == "devanagari":
            return "मैं ठीक हूँ। आपका अगला सवाल क्या है?"
        return "I am doing well. What can I help you with next?"

    thanks_signals = ["thanks", "thank you", "shukriya", "dhanyavaad", "thankyou"]
    if any(s in q for s in thanks_signals):
        if roman_lang == "hinglish":
            return "Welcome! Aur kuch chahiye ho to poochho."
        if script == "devanagari":
            return "स्वागत है! और कुछ चाहिए हो तो पूछिए।"
        return "You're welcome. Ask anything else you need."

    return None


def is_code_request(user_input):
    q = re.sub(r"\s+", " ", user_input.lower()).strip()
    keywords = [
        "code",
        "program",
        "prgm",
        "prog",
        "in c",
        "in c++",
        "in java",
        "in python",
        "function",
        "algorithm",
        "write",
        "implement",
        "कोड",
        "प्रोग्राम",
        "कार्यक्रम",
        "কোড",
        "প্রোগ্রাম",
        "ਕੋਡ",
        "ਪਰੋਗਰਾਮ",
        "કોડ",
        "પ્રોગ્રામ",
        "କୋଡ",
        "ப்ரோகிராம்",
        "கோட்",
        "కోడ్",
        "ప్రోగ్రామ్",
        "ಕೋಡ್",
        "ಪ್ರೋಗ್ರಾಂ",
        "കോഡ്",
        "പ്രോഗ്രാം",
    ]
    return any(k in q for k in keywords)


def ask_model(user_input, history=None, full_response=False, fast_mode=False):
    if history is None:
        history = []

    if fast_mode:
        system_prompt = (
            "You are a helpful offline AI assistant.\n"
            "Answer the latest user message directly.\n"
            "Keep the answer accurate, specific, and relevant.\n"
            "Avoid generic template lines.\n"
            "Keep it concise unless user asks for details."
        )
    else:
        system_prompt = (
            "You are a helpful offline AI assistant.\n"
            "Answer ONLY the latest user message.\n"
            "Do not generate fake future dialogue.\n"
            "Keep the reply concise and relevant.\n"
            "Default to short answers (max 8 lines) unless user asks for details."
        )

    system_prompt += f"\n{language_instruction_for(user_input)}"

    if full_response:
        system_prompt += (
            "\nFor code requests, provide a complete answer."
            "\nIf you return code, include the full compilable program with no truncation."
        )

    conversation = f"[{PROMPT_PROFILE_VERSION}]\n" + system_prompt + "\n\n"

    if fast_mode:
        recent_history = []
    else:
        recent_history = history[-1:] if (turbo_mode or judge_mode) else history[-4:]
    for turn in recent_history:
        conversation += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"

    conversation += f"User: {user_input}\nAssistant:"

    if full_response and fast_mode:
        mode = "ultra"
    elif full_response:
        mode = "full"
    elif fast_mode:
        mode = "micro"
    else:
        mode = "normal"

    # Cache repeated prompt evaluations to avoid recomputing common demo questions.
    raw = cached_infer(conversation, mode)

    # Prevent model spillover into fabricated next turns.
    for marker in ["\nUser:", "User:", "\nAssistant:"]:
        if marker in raw:
            raw = raw.split(marker)[0].strip()

    if raw.count("```") % 2 != 0:
        raw = raw + "\n```"

    return raw


def generate_title(user_message):
    words = user_message.strip().split()
    if not words:
        return "New Chat"
    return " ".join(words[:5]).strip().replace(":", "").replace('"', "")


def render_bubble(role, text):
    safe_text = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="msg-row {role}">
            <div class="msg-bubble {role}">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def instant_code_snippet(user_input):
    query = re.sub(r"\s+", " ", user_input.lower().strip())
    query = query.replace("prgm", "program").replace("prog", "program")

    is_c_request = (
        " in c" in query
        or query.endswith(" c")
        or " c program" in query
        or "language c" in query
    )

    if "prime" in query and is_c_request:
        return """```c
#include <stdio.h>

int isPrime(int n) {
    if (n < 2) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main() {
    int n;
    printf("Enter n: ");
    scanf("%d", &n);

    printf("Prime numbers from 2 to %d are:\\n", n);
    for (int i = 2; i <= n; i++) {
        if (isPrime(i)) printf("%d ", i);
    }
    printf("\\n");
    return 0;
}
```"""

    if "java" in query and "prime" in query:
        return """```java
import java.util.Scanner;

public class PrimeNumbers {
    static boolean isPrime(int n) {
        if (n < 2) return false;
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(); // prints primes from 2..n
        for (int i = 2; i <= n; i++) {
            if (isPrime(i)) System.out.print(i + " ");
        }
        sc.close();
    }
}
```"""
    if "palindrome" in query and is_c_request:
        return """```c
#include <stdio.h>
#include <string.h>

int isPalindrome(const char *s) {
    int left = 0;
    int right = (int)strlen(s) - 1;
    while (left < right) {
        if (s[left] != s[right]) return 0;
        left++;
        right--;
    }
    return 1;
}

int main() {
    char str[200];
    printf("Enter a string: ");
    scanf("%199s", str);

    if (isPalindrome(str)) {
        printf("Palindrome\\n");
    } else {
        printf("Not a palindrome\\n");
    }
    return 0;
}
```"""
    return None


def find_tesseract_cmd():
    found = shutil.which("tesseract")
    if found:
        return found

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def find_poppler_bin():
    found = shutil.which("pdftoppm")
    if found:
        return str(Path(found).parent)

    winget_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget_root.exists():
        for pkg_dir in winget_root.glob("oschwartz10612.Poppler_*"):
            for bin_dir in pkg_dir.glob("poppler-*\\Library\\bin"):
                if (bin_dir / "pdftoppm.exe").exists():
                    return str(bin_dir)

    return None


def extract_text_with_ocr(file_path, max_pages=8):
    try:
        import pytesseract
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "OCR dependency missing in active runtime.\n"
            f"Python: {sys.executable}\n"
            f"Install with: \"{sys.executable}\" -m pip install pytesseract pillow"
        ) from e
    except Exception as e:
        raise RuntimeError(f"OCR import error (pytesseract): {type(e).__name__}: {e}") from e

    try:
        from pdf2image import convert_from_path
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "OCR dependency missing in active runtime.\n"
            f"Python: {sys.executable}\n"
            f"Install with: \"{sys.executable}\" -m pip install pdf2image"
        ) from e
    except Exception as e:
        raise RuntimeError(f"OCR import error (pdf2image): {type(e).__name__}: {e}") from e

    tesseract_cmd = find_tesseract_cmd()
    if not tesseract_cmd:
        raise RuntimeError(
            "Tesseract executable not found. Install Tesseract OCR, or add tesseract.exe to PATH.\n"
            "Expected: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        )
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    poppler_bin = find_poppler_bin()
    try:
        convert_kwargs = {"dpi": 220, "first_page": 1, "last_page": max_pages}
        if poppler_bin:
            convert_kwargs["poppler_path"] = poppler_bin
        images = convert_from_path(file_path, **convert_kwargs)
    except Exception as e:
        raise RuntimeError(
            "OCR setup issue while converting PDF pages.\n"
            f"Poppler path used: {poppler_bin}\n"
            f"Details: {type(e).__name__}: {e}"
        ) from e

    ocr_docs = []
    for page_num, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image).strip()
        if text:
            ocr_docs.append(Document(page_content=text, metadata={"page": page_num, "source": "ocr"}))
    return ocr_docs


def describe_uploaded_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError("Could not open the uploaded image.") from e

    # Preferred offline caption model (if cached locally)
    captioner = load_image_captioner()
    if captioner is not None:
        try:
            output = captioner(image)
            if output and "generated_text" in output[0]:
                caption = output[0]["generated_text"].strip()
                if caption:
                    return caption
        except Exception:
            pass

    # Offline fallback: summarize image properties + OCR text if available.
    width, height = image.size
    orientation = "landscape" if width > height else "portrait" if height > width else "square"

    sample = image.resize((64, 64))
    colors = sample.getcolors(64 * 64)
    tone = "mixed"
    if colors:
        _, (r, g, b) = max(colors, key=lambda x: x[0])
        luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
        tone = "bright" if luminance > 170 else "dark" if luminance < 85 else "neutral"

    ocr_text = ""
    try:
        import pytesseract

        extracted = pytesseract.image_to_string(image).strip()
        if extracted:
            ocr_text = " ".join(extracted.split())
    except Exception:
        pass

    summary = (
        f"{orientation.title()} image ({width}x{height}px) with {tone} overall tones. "
        "Detailed caption model is not available locally."
    )
    if ocr_text:
        snippet = (ocr_text[:180] + "...") if len(ocr_text) > 180 else ocr_text
        summary += f" Detected text: \"{snippet}\"."
    else:
        summary += " No readable text detected in the image."

    return summary


def answer_from_image_context(user_input, image_caption):
    query = user_input.lower().strip()
    quick_describe_phrases = [
        "describe",
        "what is in",
        "what's in",
        "what can you see",
        "what is there",
        "tell me about the image",
        "describe image",
    ]
    if any(phrase in query for phrase in quick_describe_phrases):
        return f"{localized_fixed_text(user_input, 'image_description_prefix')}{image_caption}"

    prompt = (
        "You are an assistant answering questions about an uploaded image.\n"
        "Use only the provided image description.\n"
        "If the description is insufficient, say so clearly.\n"
        f"{language_instruction_for(user_input)}\n"
        f"Image description: {image_caption}\n"
        f"User: {user_input}\n"
        "Assistant:"
    )
    raw = llm.invoke(prompt).strip()
    for marker in ["\nUser:", "User:", "\nAssistant:"]:
        if marker in raw:
            raw = raw.split(marker)[0].strip()
    return raw


def is_image_related_query(user_input):
    q = re.sub(r"\s+", " ", user_input.lower()).strip()
    # Strict intent detection to avoid hijacking unrelated questions.
    explicit_terms = [
        "image",
        "photo",
        "picture",
        "pic",
        "screenshot",
        "uploaded image",
        "this image",
        "the image",
        "this photo",
        "the photo",
        "this picture",
        "the picture",
        "छवि",
        "चित्र",
        "तस्वीर",
        "इमेज",
        "ছবি",
        "ਚਿੱਤਰ",
        "ਤਸਵੀਰ",
        "છબી",
        "ଫଟୋ",
        "படம்",
        "புகைப்படம்",
        "చిత్రం",
        "ఫోటో",
        "ಚಿತ್ರ",
        "ಚಿತ್ರದ",
        "ഫോട്ടോ",
        "ചിത്രം",
    ]
    if any(t in q for t in explicit_terms):
        return True

    patterns = [
        r"\bdescribe\s+(this|the)?\s*(image|photo|picture|pic|screenshot)\b",
        r"\bwhat('s| is)\s+in\s+(this|the)\s+(image|photo|picture|pic)\b",
        r"\bwhat\s+can\s+you\s+see\s+in\s+(this|the)\s+(image|photo|picture|pic)\b",
        r"\bidentify\s+(what|objects|text)\s+in\s+(this|the)\s+(image|photo|picture|pic)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def answer_from_image_context_fast(user_input, image_caption):
    query = user_input.lower().strip()
    if any(p in query for p in ["describe", "what is in", "what's in", "what can you see", "what is there"]):
        return f"{localized_fixed_text(user_input, 'image_description_prefix')}{image_caption}"

    prompt = (
        "You are answering a question about an uploaded image.\n"
        "Use only the image description below.\n"
        "If the question is not answerable from it, say that clearly.\n"
        "Keep answer concise.\n\n"
        f"{language_instruction_for(user_input)}\n"
        f"Image description: {image_caption}\n"
        f"Question: {user_input}\n"
        "Answer:"
    )
    raw = cached_infer(prompt, "micro")
    for marker in ["\nUser:", "User:", "\nAssistant:"]:
        if marker in raw:
            raw = raw.split(marker)[0].strip()
    return raw.strip()


def fast_doc_answer(query, retriever):
    try:
        docs = retriever.get_relevant_documents(query)
    except Exception:
        return "I could not retrieve document context in fast mode. Disable Judge Mode for full QA."

    if not docs:
        return localized_fixed_text(query, "no_relevant_file_info")

    query_terms = [
        t
        for t in re.findall(r"[^\W_]+", query.lower(), flags=re.UNICODE)
        if len(t) > 2 and t not in {"what", "which", "when", "where", "this", "that", "from", "with"}
    ]

    best_sentences = []
    for d in docs[:2]:
        text = " ".join((d.page_content or "").split())
        if not text:
            continue
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        if not parts:
            parts = [text]

        for p in parts[:30]:
            s = " ".join(p.split())
            if len(s) < 24:
                continue
            lower_s = s.lower()
            score = sum(1 for t in query_terms if t in lower_s)
            if score > 0:
                best_sentences.append((score, s[:260]))

    if best_sentences:
        best_sentences.sort(key=lambda x: x[0], reverse=True)
        top = [best_sentences[0][1]]
        if len(best_sentences) > 1 and best_sentences[1][0] >= best_sentences[0][0] - 1:
            top.append(best_sentences[1][1])
        return "From your uploaded file:\n- " + "\n- ".join(top)

    fallback = " ".join((docs[0].page_content or "").split())[:280]
    if fallback:
        return "From your uploaded file:\n- " + fallback
    return localized_fixed_text(query, "doc_not_found")


def should_ocr_pdf(documents):
    if not documents:
        return True
    texts = [doc.page_content or "" for doc in documents]
    total_chars = sum(len(t.strip()) for t in texts)
    page_count = max(1, len(texts))
    avg_chars = total_chars / page_count
    alpha_chars = sum(sum(ch.isalpha() for ch in t) for t in texts)
    alpha_ratio = alpha_chars / max(1, total_chars)
    # Low extracted text density usually means scan/image-based PDF.
    return total_chars < 300 or avg_chars < 90 or alpha_ratio < 0.30


def limit_docs_for_indexing(docs, max_docs=120):
    if len(docs) <= max_docs:
        return docs
    # Keep broad coverage across the file instead of taking only the beginning.
    step = len(docs) / max_docs
    return [docs[int(i * step)] for i in range(max_docs)]


def answer_from_retrieved_context(query, retriever):
    if turbo_mode:
        return fast_doc_answer(query, retriever)

    try:
        docs = retriever.get_relevant_documents(query)
    except Exception as e:
        return f"I could not retrieve document context: {e}"

    if not docs:
        return localized_fixed_text(query, "no_relevant_file_info")

    context_parts = []
    for d in docs[:2]:
        text = " ".join((d.page_content or "").split())
        if text:
            context_parts.append(text[:420])
    context = "\n\n".join(context_parts)
    if not context:
        return "Relevant sections were found but had no readable text."

    not_found_text = localized_fixed_text(query, "doc_not_found")

    prompt = (
        "You are an offline document QA assistant.\n"
        "Answer ONLY using the context below.\n"
        f"If the answer is not present, say: {not_found_text}\n"
        f"{language_instruction_for(query)}\n"
        "Keep answer clear and factual.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    raw = cached_infer(prompt, "ultra")
    for marker in ["\nUser:", "User:", "\nAssistant:"]:
        if marker in raw:
            raw = raw.split(marker)[0].strip()
    return raw.strip()


def is_document_overview_query(query):
    q = re.sub(r"\s+", " ", query.lower()).strip()
    signals = [
        "what is my file says",
        "what does my file say",
        "what does it say",
        "what is it about",
        "what is this file about",
        "what is this document about",
        "summarize",
        "summary",
        "detailed analysis",
        "analyze this file",
        "explain this document",
        "give me analysis",
    ]
    return any(s in q for s in signals)


def has_explicit_document_intent(query):
    q = re.sub(r"\s+", " ", query.lower()).strip()
    signals = [
        "document",
        "doc",
        "file",
        "pdf",
        "docx",
        "txt",
        "uploaded file",
        "uploaded document",
        "my file",
        "my document",
        "this file",
        "this document",
        "from the file",
        "from the document",
        "in the file",
        "in the document",
    ]
    return any(s in q for s in signals)


def is_document_related_query(query, history=None):
    q = re.sub(r"\s+", " ", query.lower()).strip()

    if has_explicit_document_intent(q) or is_document_overview_query(q):
        return True

    # Short follow-up after a document-focused question should stay on document context.
    if history:
        last_user = (history[-1].get("user", "") if history else "").lower().strip()
        if last_user and (has_explicit_document_intent(last_user) or is_document_overview_query(last_user)):
            followups = {
                "why",
                "how",
                "more",
                "details",
                "continue",
                "explain",
                "explain more",
                "what else",
            }
            if q in followups or (len(q.split()) <= 6 and any(f in q for f in followups)):
                return True

    return False


def fast_document_overview(docs, query):
    if not docs:
        return localized_fixed_text(query, "no_relevant_file_info")

    points = []
    for d in docs[:4]:
        text = " ".join((d.page_content or "").split())
        if not text:
            continue
        points.append(text[:220])
        if len(points) >= 3:
            break

    if not points:
        return "The uploaded file has no readable text to analyze."

    return "Quick file analysis:\n- " + "\n- ".join(points)


def answer_from_full_document(query, docs):
    if not docs:
        return "I could not find document content to analyze."

    if turbo_mode:
        return fast_document_overview(docs, query)

    # Use the opening part of the document for broad analysis requests.
    # This is usually where title, context, scope, and key structure appear.
    selected_docs = docs[: min(6, len(docs))]
    context_parts = []
    used_chars = 0
    max_chars = 5000

    for d in selected_docs:
        text = " ".join(d.page_content.split())
        if not text:
            continue
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        piece = text[:remaining]
        context_parts.append(piece)
        used_chars += len(piece)

    context = "\n\n".join(context_parts)
    if not context:
        return "The uploaded file has no readable text to analyze."

    prompt = (
        "You are an offline document analyst.\n"
        "Use ONLY the provided document context.\n"
        "Provide a detailed, structured analysis with:\n"
        "1) Main topic\n"
        "2) Key points\n"
        "3) Important details/data\n"
        "4) Short conclusion\n"
        "If something is unclear, explicitly say it is not clearly stated in the document.\n\n"
        f"{language_instruction_for(query)}\n"
        f"Document Context:\n{context}\n\n"
        f"User Question: {query}\n"
        "Answer:"
    )

    raw = cached_infer(prompt, "ultra")
    for marker in ["\nUser:", "User:", "\nAssistant:"]:
        if marker in raw:
            raw = raw.split(marker)[0].strip()
    return raw.strip()


# ---------------- SESSION STATE ----------------
if "conversations" not in st.session_state:
    st.session_state["conversations"] = {}

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

if "active_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())[:8]
    st.session_state["conversations"][chat_id] = []
    st.session_state["chat_titles"][chat_id] = f"Chat {chat_id}"
    st.session_state["active_chat"] = chat_id

if "qa" not in st.session_state:
    st.session_state["qa"] = None

if "doc_retriever" not in st.session_state:
    st.session_state["doc_retriever"] = None

if "uploaded_docs" not in st.session_state:
    st.session_state["uploaded_docs"] = None

if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None

if "uploaded_file_id" not in st.session_state:
    st.session_state["uploaded_file_id"] = None

if "uploaded_image_caption" not in st.session_state:
    st.session_state["uploaded_image_caption"] = None

if "doc_answer_cache" not in st.session_state:
    st.session_state["doc_answer_cache"] = {}

if "generated_image" not in st.session_state:
    st.session_state["generated_image"] = None

if "generated_image_meta" not in st.session_state:
    st.session_state["generated_image_meta"] = None


# ---------------- SIDEBAR ----------------
st.sidebar.markdown(
    """
    <div class="side-brand">
        <div class="side-brand-logo">🤖</div>
        <div>
            <div class="side-brand-title">MiniChatGPT</div>
            <div class="side-brand-sub">Offline • Fast • Smart</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
try:
    backend_name = getattr(llm.client.model, "backend", "unknown")
    backend_device = getattr(llm.client.model, "device", None) or "cpu"
    st.sidebar.caption(f"Engine: {backend_name} ({backend_device})")
except Exception:
    st.sidebar.caption("Engine: unknown")

with st.sidebar.expander("Offline Image Studio", expanded=False):
    st.caption("Generate images from prompts (fully offline after model is cached).")
    local_image_model_dir = Path(__file__).resolve().parent / "models" / "sd-turbo"
    default_image_model = (
        str(local_image_model_dir)
        if local_image_model_dir.exists()
        else os.environ.get("OFFLINE_IMAGE_MODEL", "stabilityai/sd-turbo")
    )
    model_source = st.text_input(
        "Model source",
        value=default_image_model,
        help="Use a local folder path or a cached Hugging Face model id.",
    )
    img_prompt = st.text_area(
        "Image prompt",
        placeholder="A cinematic futuristic city at sunrise, ultra detailed",
        height=90,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        img_steps = st.slider("Steps", 1, 6, 1)
    with col_b:
        img_size = st.selectbox("Size", ["384x384", "512x512", "640x384", "384x640"], index=0)

    width, height = map(int, img_size.split("x"))

    if st.button("Generate Image", use_container_width=True):
        if not img_prompt.strip():
            st.warning("Enter an image prompt.")
        else:
            with st.spinner("Generating image offline..."):
                try:
                    image, img_device = generate_image_offline(
                        prompt=img_prompt.strip(),
                        model_source=model_source.strip(),
                        steps=img_steps,
                        guidance=0.0,
                        width=width,
                        height=height,
                    )
                    st.session_state["generated_image"] = image
                    st.session_state["generated_image_meta"] = {
                        "prompt": img_prompt.strip(),
                        "model": model_source.strip(),
                        "device": img_device,
                        "size": f"{width}x{height}",
                    }
                    st.toast("Image generated")
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Image generation failed: {type(e).__name__}: {e}")

    if st.session_state.get("generated_image") is not None:
        st.image(st.session_state["generated_image"], caption="Generated image", use_container_width=True)
        meta = st.session_state.get("generated_image_meta") or {}
        st.caption(
            f"Device: {meta.get('device', 'unknown')} | Size: {meta.get('size', 'n/a')}"
        )

        try:
            import io

            buf = io.BytesIO()
            st.session_state["generated_image"].save(buf, format="PNG")
            st.download_button(
                "Download PNG",
                data=buf.getvalue(),
                file_name="offline_generated.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception:
            pass
st.sidebar.markdown("### History")

if st.sidebar.button("+ New Chat", use_container_width=True):
    new_id = str(uuid.uuid4())[:8]
    st.session_state["conversations"][new_id] = []
    st.session_state["chat_titles"][new_id] = f"Chat {new_id}"
    st.session_state["active_chat"] = new_id

for cid, title in st.session_state["chat_titles"].items():
    if st.sidebar.button(title, key=cid, use_container_width=True):
        st.session_state["active_chat"] = cid

if st.session_state.get("uploaded_image_caption"):
    st.sidebar.markdown("### Image Context")
    st.sidebar.caption(st.session_state["uploaded_image_caption"])

active_chat = st.session_state["active_chat"]


# ---------------- FILE UPLOAD (IN COMPOSER AREA) ----------------
qa = None
uploaded_file = None

with st.container():
    st.markdown('<div class="composer-wrap">', unsafe_allow_html=True)
    st.markdown(
        '<div class="upload-note">Attach PDF/DOCX/TXT for document Q&A or image (PNG/JPG/WEBP) for image description.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload file",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    uploaded_bytes = uploaded_file.getvalue()
    uploaded_file_id = f"{uploaded_file.name}:{hashlib.sha256(uploaded_bytes).hexdigest()}"
    is_image_file = uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))

    # Process the file only when a new file is selected.
    if st.session_state["uploaded_file_id"] != uploaded_file_id:
        # Reset previous retriever immediately so stale context is never used.
        st.session_state["qa"] = None
        st.session_state["doc_retriever"] = None
        st.session_state["uploaded_docs"] = None
        st.session_state["uploaded_file_name"] = None
        st.session_state["uploaded_image_caption"] = None
        st.session_state["doc_answer_cache"] = {}

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_bytes)
            file_path = tmp.name

        if is_image_file:
            try:
                image_caption = describe_uploaded_image(file_path)
                st.session_state["uploaded_image_caption"] = image_caption
                st.session_state["uploaded_file_name"] = uploaded_file.name
                st.session_state["uploaded_file_id"] = uploaded_file_id
                st.toast("Image uploaded and analyzed successfully")
            except RuntimeError as e:
                st.error(str(e))
                st.session_state["uploaded_file_id"] = uploaded_file_id
            finally:
                try:
                    os.remove(file_path)
                except Exception:
                    pass
        elif uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            st.error("Unsupported file type")
            st.session_state["uploaded_file_id"] = uploaded_file_id
            st.stop()

        if not is_image_file:
            try:
                documents = loader.load()
            except Exception as e:
                st.error(f"Error reading document: {e}")
                st.session_state["uploaded_file_id"] = uploaded_file_id
                st.stop()

            if documents:
                extracted_text = " ".join(doc.page_content for doc in documents).strip()

                if uploaded_file.name.endswith(".pdf") and ocr_enabled and should_ocr_pdf(documents):
                    with st.spinner("Running OCR on PDF pages for better accuracy..."):
                        try:
                            # Keep OCR bounded for speed on long PDFs.
                            ocr_max_pages = min(6, max(3, len(documents) // 4))
                            ocr_docs = extract_text_with_ocr(file_path, max_pages=ocr_max_pages)
                        except RuntimeError as e:
                            st.warning(str(e))
                            ocr_docs = []

                    ocr_text = " ".join(doc.page_content for doc in ocr_docs).strip() if ocr_docs else ""
                    if ocr_text and len(ocr_text) > max(200, int(len(extracted_text) * 1.2)):
                        documents = ocr_docs
                        st.toast("OCR text selected for this PDF (better readable content).")
                    elif not extracted_text and ocr_text:
                        documents = ocr_docs
                        st.toast("OCR completed. Using extracted text from scanned PDF.")
                    elif not extracted_text and not ocr_text:
                        st.warning(
                            "Could not read text from this PDF (including OCR). Try a clearer file."
                        )
                        st.session_state["uploaded_file_name"] = uploaded_file.name
                        st.session_state["uploaded_file_id"] = uploaded_file_id
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
                        st.stop()

                if turbo_mode:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=80)
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=180)
                docs = text_splitter.split_documents(documents)
                if turbo_mode:
                    docs = limit_docs_for_indexing(docs, max_docs=70)
                st.session_state["uploaded_docs"] = docs

                if docs:
                    embeddings = load_embeddings()
                    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
                    retriever_k = min(1, len(docs))
                    retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": retriever_k},
                    )

                    strict_prompt = PromptTemplate(
                        input_variables=["context", "question"],
                        template="""
You are an AI assistant.

Use ONLY the information provided in the context below to answer the question.

If the answer is not present in the context, say so clearly in the same language as the question.

Do NOT use your general knowledge.
If user asks for analysis/summary, provide a detailed response from the context.
Always answer in the same language as the question.
If the question uses English letters to write another language (like Hinglish/Tanglish),
answer in that same language using English letters only.

Context:
{context}

Question:
{question}

Answer:
""",
                    )

                    if not judge_mode:
                        st.session_state["qa"] = RetrievalQA.from_chain_type(
                            llm=llm_full,
                            retriever=retriever,
                            chain_type="stuff",
                            chain_type_kwargs={"prompt": strict_prompt},
                        )
                    else:
                        st.session_state["qa"] = None
                    st.session_state["doc_retriever"] = retriever
                    st.session_state["uploaded_file_name"] = uploaded_file.name
                    st.session_state["uploaded_file_id"] = uploaded_file_id
                    st.toast("File uploaded and processed successfully")
                else:
                    st.warning("No readable content found in document.")
                    st.session_state["uploaded_file_name"] = uploaded_file.name
                    st.session_state["uploaded_file_id"] = uploaded_file_id
            else:
                st.warning("Document appears empty.")
                st.session_state["uploaded_file_name"] = uploaded_file.name
                st.session_state["uploaded_file_id"] = uploaded_file_id

            try:
                os.remove(file_path)
            except Exception:
                pass

qa = st.session_state["qa"]
doc_retriever = st.session_state["doc_retriever"]
uploaded_docs = st.session_state["uploaded_docs"]
image_caption = st.session_state["uploaded_image_caption"]


# ---------------- CHAT AREA ----------------
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

if not st.session_state["conversations"][active_chat]:
    if image_caption:
        render_bubble("assistant", "Image is ready. Ask me to describe it or ask questions about it.")
    else:
        render_bubble("assistant", "Ask me anything, or upload a file and ask questions from it.")

for chat in st.session_state["conversations"][active_chat]:
    render_bubble("user", chat["user"])
    render_bubble("assistant", chat["bot"])

typed_query = st.chat_input("Type your message and press Enter")
query = typed_query

if query:
    # Show user message immediately instead of waiting for model response.
    render_bubble("user", query)
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown(
        """
        <div class="msg-row assistant">
            <div class="msg-bubble assistant">Thinking...</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    code_request = is_code_request(query)
    image_query = is_image_related_query(query)
    doc_query = is_document_related_query(query, st.session_state["conversations"][active_chat])
    answer = instant_code_snippet(query) if instant_code_mode else None

    if answer is None:
        if judge_mode:
            try:
                if image_caption and image_query and not code_request:
                    answer = answer_from_image_context_fast(query, image_caption)
                elif doc_retriever and use_uploaded_context and not code_request and doc_query:
                    cache_key = f"{st.session_state.get('uploaded_file_id','')}::{query.strip().lower()}"
                    cached_doc_answer = st.session_state["doc_answer_cache"].get(cache_key)
                    if cached_doc_answer:
                        answer = cached_doc_answer
                    else:
                        if is_document_overview_query(query):
                            answer = answer_from_full_document(query, uploaded_docs)
                        else:
                            answer = answer_from_retrieved_context(query, doc_retriever)
                        st.session_state["doc_answer_cache"][cache_key] = answer
                else:
                    quick_chat = instant_smalltalk_reply(query)
                    if quick_chat is not None:
                        answer = quick_chat
                    else:
                        answer = ask_model(
                            query,
                            st.session_state["conversations"][active_chat],
                            full_response=code_request,
                            fast_mode=True,
                        )
            except Exception as e:
                answer = f"Fast mode error: {e}"
        else:
            with st.spinner("Thinking..."):
                try:
                    if qa and use_uploaded_context and not code_request and doc_query:
                        result = qa.invoke({"query": query})
                        answer = result["result"]
                    elif image_caption and image_query and not code_request:
                        answer = answer_from_image_context(query, image_caption)
                    else:
                        answer = ask_model(
                            query,
                            st.session_state["conversations"][active_chat],
                            full_response=code_request,
                        )
                except Exception as e:
                    answer = f"Error while generating response: {e}"

    if turbo_mode and len(answer) > 500 and not code_request and not answer.strip().startswith("```"):
        answer = answer[:500].rsplit(" ", 1)[0] + "..."

    thinking_placeholder.empty()
    render_bubble("assistant", answer)

    st.session_state["conversations"][active_chat].append({"user": query, "bot": answer})

    if st.session_state["chat_titles"][active_chat].startswith("Chat"):
        try:
            st.session_state["chat_titles"][active_chat] = generate_title(query)
        except Exception:
            st.session_state["chat_titles"][active_chat] = "New Chat"

    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
