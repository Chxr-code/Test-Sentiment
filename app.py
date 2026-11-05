import html
import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from pythainlp.tokenize import word_tokenize

from src.models.bilstm import BiLSTM


# --------------------------------------------------
# Paths and settings
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"
CHECKPOINT_PATH = ARTIFACT_DIR / "checkpoints" / "bilstm_best.pt"
VOCAB_PATH = ARTIFACT_DIR / "vocab.json"
CONFIG_PATH = ARTIFACT_DIR / "preprocessing_config.json"

LABEL_DISPLAY = {
    "pos": ("Positive", "ความรู้สึกเชิงบวก"),
    "neu": ("Neutral", "ความรู้สึกเป็นกลาง"),
    "neg": ("Negative", "ความรู้สึกเชิงลบ"),
    "q": ("Question", "ข้อความคำถาม"),
}

# Neo-brutalism palette
INK = "#1A1A1A"
CREAM = "#FBF4EC"
CORAL = "#FF6B6B"
PURPLE = "#9B87F5"
TEAL = "#4ECDC4"
YELLOW = "#FFD93D"
ORANGE = "#FF9F5A"
PINK = "#FFC9C9"

LABEL_COLORS = {
    "pos": TEAL,
    "neu": YELLOW,
    "neg": CORAL,
    "q": PURPLE,
}

LABEL_ICONS = {
    "pos": "😊",
    "neu": "😐",
    "neg": "😠",
    "q": "❓",
}


# --------------------------------------------------
# Load the exact vocabulary, preprocessing config,
# and selected BiLSTM checkpoint.
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    required_files = [CHECKPOINT_PATH, VOCAB_PATH, CONFIG_PATH]
    missing_files = [str(path.relative_to(ROOT_DIR)) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "ไม่พบไฟล์ที่จำเป็น: " + ", ".join(missing_files)
        )

    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        vocab = json.load(file)

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        preprocessing_config = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # weights_only=False is required because the checkpoint also stores
    # training metadata/configuration. Only load this trusted project file.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    model_config = checkpoint["config"]

    vocab_size = len(vocab)
    max_len = int(preprocessing_config["max_len"])

    if vocab_size != int(preprocessing_config["vocab_size"]):
        raise ValueError("ขนาด Vocabulary ไม่ตรงกับ preprocessing_config.json")

    if vocab_size != int(model_config["vocab_size"]):
        raise ValueError("ขนาด Vocabulary ไม่ตรงกับ checkpoint")

    if max_len != int(model_config["max_len"]):
        raise ValueError("MAX_LEN ไม่ตรงกับ checkpoint")

    if preprocessing_config["tokenizer"] != "pythainlp.newmm":
        raise ValueError("Tokenizer ใน config ไม่ตรงกับ pythainlp.newmm")

    model = BiLSTM(
        vocab_size=vocab_size,
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        num_classes=len(preprocessing_config["label_mapping"]),
        dropout=float(model_config["dropout"]),
        padding_idx=int(preprocessing_config["padding_index"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab, preprocessing_config, device


# --------------------------------------------------
# Inference preprocessing: intentionally mirrors
# 11_encode_dataset.py
# --------------------------------------------------
def encode_text(text, vocab, config):
    tokens = word_tokenize(
        text,
        engine="newmm",
        keep_whitespace=False,
    )

    if not tokens:
        return None, tokens

    max_len = int(config["max_len"])
    unk_id = int(config["unknown_index"])
    pad_id = int(config["padding_index"])

    token_ids = [vocab.get(token, unk_id) for token in tokens]
    token_ids = token_ids[:max_len]

    attention_mask = [1] * len(token_ids)
    padding_length = max_len - len(token_ids)

    token_ids += [pad_id] * padding_length
    attention_mask += [0] * padding_length

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    mask = torch.tensor([attention_mask], dtype=torch.long)

    return (input_ids, mask), tokens


# --------------------------------------------------
# UI helpers
# --------------------------------------------------
def render_html(markup: str):
    """Render HTML without indentation/blank lines (which Markdown would
    otherwise turn into code blocks or break apart)."""
    cleaned = "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def inject_styles():
    render_html(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap');

:root {{
  --ink: {INK}; --cream: {CREAM}; --coral: {CORAL}; --purple: {PURPLE};
  --teal: {TEAL}; --yellow: {YELLOW}; --orange: {ORANGE}; --pink: {PINK};
  --border: 3px solid var(--ink);
  --shadow: 5px 5px 0 var(--ink);
  --radius: 14px;
}}

/* ---------- Base ---------- */
.stApp, [data-testid="stAppViewContainer"] {{
  background: var(--cream) !important;
  color: var(--ink);
}}
.stApp, .stApp p, .stApp label, .stApp span, .stApp div, .stApp textarea, .stApp button {{
  font-family: 'Space Grotesk', 'IBM Plex Sans Thai', sans-serif;
}}
[data-testid="stIconMaterial"], .material-symbols-rounded {{
  font-family: 'Material Symbols Rounded' !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
/* Course badge at top-left, same bar as Deploy */
[data-testid="stHeader"]::before {{
  content: "CMT 6702 (209-365)";
  position: absolute; left: 1rem; top: 50%; transform: translateY(-50%);
  z-index: 2; pointer-events: none;
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: .9rem;
  color: var(--ink); background: var(--yellow);
  border: 2.5px solid var(--ink); border-radius: 10px;
  padding: 4px 12px; box-shadow: 3px 3px 0 var(--ink);
}}
[data-testid="stMain"] {{ position: relative; z-index: 1; }}
.block-container {{ padding-top: 3rem; padding-bottom: 4rem; }}

/* Decorative corner blocks like the reference */
.stApp::before, .stApp::after {{
  content: ""; position: fixed; z-index: 0; pointer-events: none;
  border: var(--border);
}}
.stApp::before {{
  top: -40px; right: -40px; width: 300px; height: 340px;
  background: var(--yellow); border-radius: 0 0 0 70px;
}}
.stApp::after {{
  bottom: -40px; left: -40px; width: 260px; height: 300px;
  background: var(--pink); border-radius: 0 70px 0 0;
}}
@media (max-width: 1100px) {{
  .stApp::before, .stApp::after {{ display: none; }}
}}

/* ---------- Hero ---------- */
.nb-hero {{
  background: #fff; border: var(--border); border-radius: 22px;
  box-shadow: 8px 8px 0 var(--ink); padding: 2rem 1.5rem 1.6rem;
  text-align: center; margin-bottom: 1.6rem;
}}
.nb-dots {{ display: flex; justify-content: center; gap: 8px; margin-bottom: 1rem; }}
.nb-dots span {{ width: 18px; height: 18px; border-radius: 50%; border: 2.5px solid var(--ink); }}
.nb-icon {{
  width: 68px; height: 68px; margin: 0 auto 0.9rem; display: grid; place-items: center;
  font-size: 2rem; background: var(--coral); border: var(--border);
  border-radius: 16px; box-shadow: var(--shadow);
}}
.nb-title {{
  font-family: 'Space Grotesk', sans-serif !important; font-weight: 700;
  font-size: clamp(2rem, 6vw, 3.1rem); line-height: 1.05; letter-spacing: -1px;
  color: var(--ink); margin: 0;
}}
.nb-sub {{ font-size: 1.15rem; font-weight: 600; color: #555; margin: .35rem 0 1rem; }}
.nb-pills {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-bottom: 1rem; }}
.nb-pill {{
  padding: 5px 16px; border: 2.5px solid var(--ink); border-radius: 999px;
  font-weight: 600; font-size: .9rem; box-shadow: 3px 3px 0 var(--ink);
}}
.nb-tagline {{ color: #444; margin: 0; font-size: .98rem; }}

/* ---------- Class legend ---------- */
.nb-legend {{
  background: #fff; border: var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: 1rem 1.1rem; margin-bottom: 1.6rem;
}}
.nb-legend-title {{ font-weight: 700; margin-bottom: .7rem; }}
.nb-legend-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; }}
.nb-chip {{
  border: 2.5px solid var(--ink); border-radius: 10px; padding: 8px 10px;
  font-weight: 700; box-shadow: 3px 3px 0 var(--ink); font-size: .92rem;
}}
.nb-chip small {{ display: block; font-weight: 500; font-size: .78rem; }}

/* ---------- Text area ---------- */
.stTextArea label p {{ font-weight: 700 !important; font-size: 1.05rem !important; color: var(--ink) !important; }}
.stTextArea [data-baseweb="textarea"], .stTextArea [data-baseweb="base-input"] {{
  border: var(--border) !important; border-radius: var(--radius) !important;
  background: #fff !important; box-shadow: var(--shadow);
}}
.stTextArea textarea {{
  background: #fff !important; color: var(--ink) !important;
  font-size: 1.05rem !important; caret-color: var(--ink);
}}
.stTextArea textarea::placeholder {{ color: #8a8a8a !important; }}

/* ---------- Button ---------- */
.stButton > button {{
  background: var(--coral) !important; color: var(--ink) !important;
  border: var(--border) !important; border-radius: var(--radius) !important;
  box-shadow: var(--shadow) !important; padding: .7rem 1rem !important;
  transition: transform .08s ease, box-shadow .08s ease;
  margin-top: .4rem;
}}
.stButton > button p {{ font-weight: 700 !important; font-size: 1.1rem !important; color: var(--ink) !important; }}
.stButton > button:hover {{ transform: translate(2px, 2px); box-shadow: 3px 3px 0 var(--ink) !important; }}
.stButton > button:active {{ transform: translate(5px, 5px); box-shadow: 0 0 0 var(--ink) !important; }}
.stButton > button:focus:not(:active) {{ border-color: var(--ink) !important; }}

/* ---------- Alerts (warning / error) ---------- */
[data-testid="stAlert"] {{
  border: var(--border); border-radius: var(--radius); box-shadow: var(--shadow);
}}
[data-testid="stAlertContainer"] {{ background: var(--yellow) !important; color: var(--ink) !important; }}
[data-testid="stAlertContainer"] p {{ color: var(--ink) !important; font-weight: 600; }}

/* ---------- Section heading ---------- */
.nb-heading {{
  display: inline-block; font-weight: 700; font-size: 1.25rem;
  background: var(--ink); color: #fff; padding: 4px 14px; border-radius: 8px;
  margin: 1.8rem 0 1rem; transform: rotate(-1deg);
}}

/* ---------- Result card ---------- */
.nb-result {{
  border: var(--border); border-radius: 20px; box-shadow: 8px 8px 0 var(--ink);
  padding: 1.4rem 1.5rem; display: flex; align-items: center; gap: 1.2rem; flex-wrap: wrap;
}}
.nb-result-icon {{
  width: 72px; height: 72px; flex-shrink: 0; display: grid; place-items: center;
  font-size: 2.2rem; background: #fff; border: var(--border); border-radius: 16px;
}}
.nb-result-body {{ flex: 1; min-width: 160px; }}
.nb-result-kicker {{ font-weight: 600; font-size: .85rem; text-transform: uppercase; letter-spacing: 1px; }}
.nb-result-label {{ font-size: 2.1rem; font-weight: 700; line-height: 1.1; }}
.nb-result-thai {{ font-weight: 500; }}
.nb-conf {{
  background: #fff; border: var(--border); border-radius: 14px;
  padding: .6rem 1rem; text-align: center; box-shadow: 4px 4px 0 var(--ink);
}}
.nb-conf-value {{ font-size: 1.8rem; font-weight: 700; line-height: 1.1; }}
.nb-conf-label {{ font-size: .78rem; font-weight: 600; }}

/* ---------- Probability bars ---------- */
.nb-bars {{
  background: #fff; border: var(--border); border-radius: 20px;
  box-shadow: var(--shadow); padding: 1.2rem 1.3rem;
}}
.nb-bar-row + .nb-bar-row {{ margin-top: 1rem; }}
.nb-bar-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; gap: 8px; }}
.nb-bar-name {{ font-weight: 700; }}
.nb-bar-name small {{ font-weight: 500; color: #555; margin-left: 6px; }}
.nb-bar-pct {{ font-weight: 700; font-variant-numeric: tabular-nums; }}
.nb-track {{
  height: 26px; background: #EFE8DF; border: var(--border);
  border-radius: 999px; overflow: hidden;
}}
.nb-fill {{ height: 100%; border-right: var(--border); border-radius: 999px 0 0 999px; }}
.nb-fill.full {{ border-right: none; border-radius: 999px; }}
.nb-fill.empty {{ border-right: none; }}

/* ---------- Expander ---------- */
[data-testid="stExpander"] details {{
  border: var(--border) !important; border-radius: var(--radius) !important;
  background: #fff; box-shadow: var(--shadow);
}}
[data-testid="stExpander"] summary p {{ font-weight: 700; color: var(--ink); }}
[data-testid="stExpander"] {{ margin-top: 1.6rem; }}
.nb-tokens {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.nb-token {{
  border: 2px solid var(--ink); border-radius: 8px; padding: 2px 9px;
  background: var(--cream); font-size: .92rem;
}}

/* ---------- Small text ---------- */
.nb-note {{ font-size: .85rem; color: #555; margin-top: 1.2rem; }}
.nb-footer {{
  margin-top: 3rem; text-align: center; font-size: .85rem; font-weight: 600;
  border-top: var(--border); padding-top: 1rem; color: var(--ink);
}}
[data-testid="stCaptionContainer"] p {{ color: #555 !important; }}
</style>
"""
    )


def render_hero(device):
    dots = "".join(
        f'<span style="background:{color}"></span>'
        for color in (CORAL, PURPLE, TEAL, YELLOW, TEAL, ORANGE)
    )
    render_html(
        f"""
<div class="nb-hero">
  <div class="nb-dots">{dots}</div>
  <div class="nb-icon">💬</div>
  <div class="nb-title">THAI SENTIMENT.AI</div>
  <div class="nb-sub">Thai Social Media Sentiment Classification</div>
  <div class="nb-pills">
    <span class="nb-pill" style="background:#fff">BiLSTM</span>
    <span class="nb-pill" style="background:{PURPLE}">Wisesight Corpus</span>
    <span class="nb-pill" style="background:{TEAL}">4 Classes</span>
    <span class="nb-pill" style="background:{YELLOW}">Device: {html.escape(str(device)).upper()}</span>
  </div>
  <p class="nb-tagline">ทดลองวิเคราะห์ข้อความภาษาไทยด้วยโมเดล Deep Learning ที่ฝึกจาก Wisesight Sentiment Corpus</p>
</div>
"""
    )


def render_legend():
    chips = "".join(
        f'<div class="nb-chip" style="background:{LABEL_COLORS[key]}">'
        f"{LABEL_ICONS[key]} {english}<small>{thai}</small></div>"
        for key, (english, thai) in LABEL_DISPLAY.items()
    )
    render_html(
        f"""
<div class="nb-legend">
  <div class="nb-legend-title">ระบบจำแนกข้อความเป็น 4 คลาส</div>
  <div class="nb-legend-grid">{chips}</div>
</div>
"""
    )


def render_result(label, confidence):
    english, thai = LABEL_DISPLAY[label]
    render_html(
        f"""
<div class="nb-heading">ผลการวิเคราะห์</div>
<div class="nb-result" style="background:{LABEL_COLORS[label]}">
  <div class="nb-result-icon">{LABEL_ICONS[label]}</div>
  <div class="nb-result-body">
    <div class="nb-result-kicker">Prediction</div>
    <div class="nb-result-label">{english}</div>
    <div class="nb-result-thai">{thai}</div>
  </div>
  <div class="nb-conf">
    <div class="nb-conf-value">{confidence:.2%}</div>
    <div class="nb-conf-label">Confidence</div>
  </div>
</div>
"""
    )


def render_probability_bars(probability_rows):
    rows_html = ""
    for row in probability_rows:
        pct = row["Probability"] * 100
        fill_class = "full" if pct >= 99.5 else ("empty" if pct < 0.5 else "")
        rows_html += f"""
<div class="nb-bar-row">
  <div class="nb-bar-head">
    <div class="nb-bar-name">{row["english"]} ({row["label"]})<small>{row["thai"]}</small></div>
    <div class="nb-bar-pct">{pct:.2f}%</div>
  </div>
  <div class="nb-track">
    <div class="nb-fill {fill_class}" style="width:{pct:.2f}%;background:{LABEL_COLORS[row["label"]]}"></div>
  </div>
</div>
"""
    render_html(
        f"""
<div class="nb-heading">Probability ของแต่ละคลาส</div>
<div class="nb-bars">{rows_html}</div>
"""
    )


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------
st.set_page_config(
    page_title="THAI SENTIMENT.AI",
    page_icon="💬",
    layout="centered",
)

inject_styles()

try:
    model, vocab, config, device = load_artifacts()
except Exception as error:
    render_hero("—")
    st.error(f"โหลดโมเดลหรือไฟล์ประกอบไม่สำเร็จ: {error}")
    st.stop()

render_hero(device)
render_legend()

user_text = st.text_area(
    "ป้อนข้อความภาษาไทยที่ต้องการวิเคราะห์",
    height=150,
    placeholder="ตัวอย่าง: ร้านนี้บริการดีมาก ประทับใจสุด ๆ",
    max_chars=2000,
)

analyze = st.button("✦ วิเคราะห์ความรู้สึก", type="primary", use_container_width=True)

if analyze:
    if not user_text.strip():
        st.warning("กรุณาป้อนข้อความก่อนวิเคราะห์")
    else:
        encoded, tokens = encode_text(user_text, vocab, config)

        if encoded is None:
            st.warning("ไม่พบ Token จากข้อความนี้ กรุณาลองป้อนข้อความอื่น")
        else:
            input_ids, attention_mask = encoded
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.inference_mode():
                logits = model(input_ids, attention_mask)
                probabilities = F.softmax(logits, dim=1)[0].cpu()

            label_mapping = config["label_mapping"]
            index_to_label = {
                int(index): label for label, index in label_mapping.items()
            }

            predicted_index = int(torch.argmax(probabilities).item())
            predicted_label = index_to_label[predicted_index]
            confidence = float(probabilities[predicted_index].item())

            render_result(predicted_label, confidence)

            probability_rows = []
            for label, index in sorted(label_mapping.items(), key=lambda item: item[1]):
                english, thai = LABEL_DISPLAY[label]
                probability_rows.append(
                    {
                        "label": label,
                        "english": english,
                        "thai": thai,
                        "Probability": float(probabilities[int(index)].item()),
                    }
                )

            render_probability_bars(probability_rows)

            with st.expander("ดูรายละเอียดการตัดคำ"):
                token_chips = "".join(
                    f'<span class="nb-token">{html.escape(token)}</span>' for token in tokens
                )
                render_html(f'<div class="nb-tokens">{token_chips}</div>')
                st.caption(
                    f"จำนวน Token ก่อนตัดความยาว: {len(tokens)} | "
                    f"MAX_LEN: {config['max_len']} | "
                    f"Vocabulary size: {len(vocab)}"
                )

            render_html(
                '<p class="nb-note">หมายเหตุ: Probability เป็นค่าที่โมเดลคำนวณได้ '
                "ไม่ใช่การรับประกันความถูกต้องของผลวิเคราะห์</p>"
            )

render_html(
    """
<div class="nb-footer">
THAI SENTIMENT.AI · Research Demo · Dataset: Wisesight Sentiment Corpus · Model: BiLSTM
</div>
"""
)