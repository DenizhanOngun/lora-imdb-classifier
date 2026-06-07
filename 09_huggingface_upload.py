from google.colab import drive
drive.mount('/content/drive')
exec(open("/content/drive/MyDrive/imdb_peft_project/code/lora-imdb-classifier/00_colab_setup.py").read())

!pip install huggingface_hub transformers peft safetensors -q

from huggingface_hub import login, HfApi, upload_folder, upload_file
import getpass

hf_token = getpass.getpass("HuggingFace Token: ")
login(token=hf_token)

api = HfApi()
print(f"✓ Logged in: {api.whoami()['name']}")

HF_USERNAME = "Denizhaan"

import os

DIRS["checkpoints2_v2_fixed"] = f"{DIRS['root']}/checkpoints/deberta_lora_v2_fixed"

# Create new repo for fixed model
from huggingface_hub import create_repo
create_repo(f"{HF_USERNAME}/imdb-deberta-lora-v2-fixed", exist_ok=True)
print("✓ Repository created.")

# Upload epoch_3 checkpoint (best model)
upload_folder(
    folder_path=f"{DIRS['checkpoints2_v2_fixed']}/epoch_3",
    repo_id=f"{HF_USERNAME}/imdb-deberta-lora-v2-fixed",
    commit_message="Upload DeBERTa+LoRA V2 Fixed - includes pooler weights"
)
print("✓ DeBERTa Fixed uploaded to HuggingFace.")

import os

app_code = '''
import gradio as gr
import torch
import numpy as np
import re
import pickle
import safetensors.torch as st
from transformers import (RobertaTokenizer, RobertaForSequenceClassification,
                          DebertaV2Tokenizer, DebertaV2ForSequenceClassification)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from huggingface_hub import hf_hub_download

# ── Constants ─────────────────────────────────────────────────
HF_USERNAME   = "Denizhaan"
ROBERTA_REPO  = f"{HF_USERNAME}/imdb-roberta-lora-v2"
DEBERTA_REPO  = f"{HF_USERNAME}/imdb-deberta-lora-v2-fixed"
SVM_REPO      = f"{HF_USERNAME}/imdb-tfidf-svm"
DEVICE        = torch.device("cpu")

# ── Text cleaning ──────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()

# ── Head+Tail truncation (V2: 256/256) ────────────────────────
def head_tail_truncate(text, tokenizer, max_len=512, head_len=256):
    tail_len = max_len - head_len
    tokens = tokenizer(text, add_special_tokens=False,
                       truncation=False, return_tensors=None)
    input_ids      = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    if len(input_ids) > max_len - 2:
        input_ids      = input_ids[:head_len] + input_ids[-tail_len:]
        attention_mask = attention_mask[:head_len] + attention_mask[-tail_len:]
    return tokenizer(tokenizer.decode(input_ids), max_length=max_len,
                     padding="max_length", truncation=True,
                     return_tensors="pt")

# ── Load models ────────────────────────────────────────────────
print("Loading models...")

# TF-IDF + SVM
tfidf_path = hf_hub_download(SVM_REPO, "tfidf.pkl")
svm_path   = hf_hub_download(SVM_REPO, "svm.pkl")
with open(tfidf_path, "rb") as f: tfidf = pickle.load(f)
with open(svm_path,   "rb") as f: svm   = pickle.load(f)
print("✓ TF-IDF + SVM loaded.")

# RoBERTa + LoRA V2
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_base      = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=2)
roberta_model     = PeftModel.from_pretrained(roberta_base, ROBERTA_REPO)
roberta_model.eval()
print("✓ RoBERTa + LoRA V2 loaded.")

# DeBERTa + LoRA V2 Fixed (with pooler)
deberta_tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
deberta_base      = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=2, torch_dtype=torch.float32,
    ignore_mismatched_sizes=True)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["query_proj", "value_proj"], bias="none"
)
deberta_model = get_peft_model(deberta_base, lora_config)

# Load LoRA weights
adapter_path = hf_hub_download(DEBERTA_REPO, "adapter_model.safetensors")
extra_path   = hf_hub_download(DEBERTA_REPO, "extra_weights.pt")
weights      = st.load_file(adapter_path)
extra        = torch.load(extra_path, map_location="cpu")

new_weights = {}
for k, v in weights.items():
    if "lora_A.weight" in k:
        new_weights[k.replace("lora_A.weight", "lora_A.default.weight")] = v
    elif "lora_B.weight" in k:
        new_weights[k.replace("lora_B.weight", "lora_B.default.weight")] = v

new_weights["base_model.model.classifier.modules_to_save.default.weight"] = extra["classifier.weight"]
new_weights["base_model.model.classifier.modules_to_save.default.bias"]   = extra["classifier.bias"]
new_weights["base_model.model.classifier.original_module.weight"]          = extra["classifier.weight"]
new_weights["base_model.model.classifier.original_module.bias"]            = extra["classifier.bias"]
new_weights["base_model.model.pooler.dense.weight"]                        = extra["pooler.weight"]
new_weights["base_model.model.pooler.dense.bias"]                          = extra["pooler.bias"]

deberta_model.load_state_dict(new_weights, strict=False)
deberta_model = deberta_model.to(torch.float32)
deberta_model.eval()
print("✓ DeBERTa + LoRA V2 Fixed loaded.")

print("All models ready!")

# ── Prediction function ────────────────────────────────────────
def predict(review):
    if not review.strip():
        return "Please enter a review.", "", "", ""

    text       = clean_text(review)
    text_lower = text.lower()

    # 1. TF-IDF + SVM
    svm_pred  = svm.predict(tfidf.transform([text_lower]))[0]
    svm_label = "✅ Positive" if svm_pred == 1 else "❌ Negative"

    # 2. RoBERTa + LoRA
    with torch.no_grad():
        r_enc   = head_tail_truncate(text, roberta_tokenizer)
        r_out   = roberta_model(**r_enc)
        r_probs = torch.softmax(r_out.logits, dim=-1)[0]
        r_pred  = r_probs.argmax().item()
        r_conf  = r_probs[r_pred].item()
    r_label = f"{'✅ Positive' if r_pred == 1 else '❌ Negative'} ({r_conf:.1%})"

    # 3. DeBERTa + LoRA Fixed
    with torch.no_grad():
        d_enc   = head_tail_truncate(text, deberta_tokenizer)
        d_out   = deberta_model(**d_enc)
        d_probs = torch.softmax(d_out.logits, dim=-1)[0]
        d_pred  = d_probs.argmax().item()
        d_conf  = d_probs[d_pred].item()
    d_label = f"{'✅ Positive' if d_pred == 1 else '❌ Negative'} ({d_conf:.1%})"

    # 4. Meta-Learner (weighted combination based on LR weights)
    r_pos = r_probs[1].item()
    d_pos = d_probs[1].item()
    meta_score = r_pos * 4.2121 + d_pos * 2.9001
    meta_pred  = 1 if meta_score > (4.2121 + 2.9001) / 2 else 0
    meta_label = f"{'✅ Positive' if meta_pred == 1 else '❌ Negative'}"

    return svm_label, r_label, d_label, meta_label

# ── Gradio Interface ───────────────────────────────────────────
examples = [
    ["I went into this movie with fairly low expectations and ended up being pleasantly surprised. The story is engaging from beginning to end, and the pacing never really drags. What impressed me most was the cast. Every major actor delivered a convincing performance, and several scenes felt genuinely emotional rather than forced."],
    ["I honestly struggled to finish this movie. The plot is predictable, the dialogue feels unnatural, and most of the characters are so poorly developed that it is difficult to care about what happens to them. Several scenes seem to exist only to increase the runtime and contribute very little to the overall story."],
    ["I would not say this movie is completely terrible. In fact, there are moments where it almost feels like it might become genuinely interesting. The actors are not bad, the visual effects are not awful, and the story is not entirely without ideas."],
]

with gr.Blocks(title="IMDB Sentiment Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 IMDB Sentiment Analysis
    ### Parameter-Efficient Fine-Tuning (LoRA) + Stacking Ensemble
    Compare predictions from **4 models**: TF-IDF+SVM, RoBERTa+LoRA, DeBERTa+LoRA, and Meta-Learner
    """)

    with gr.Row():
        input_text = gr.Textbox(
            label="Movie Review",
            placeholder="Enter your movie review here...",
            lines=5
        )

    btn = gr.Button("Analyze Sentiment", variant="primary")

    with gr.Row():
        out_svm     = gr.Textbox(label="TF-IDF + SVM (Baseline)")
        out_roberta = gr.Textbox(label="RoBERTa + LoRA V2")
        out_deberta = gr.Textbox(label="DeBERTa + LoRA V2")
        out_meta    = gr.Textbox(label="Meta-Learner (Ensemble)")

    btn.click(fn=predict, inputs=input_text,
              outputs=[out_svm, out_roberta, out_deberta, out_meta])

    gr.Examples(examples=examples, inputs=input_text)

    gr.Markdown("""
    ---
    **Project:** Parameter-Efficient Fine-Tuning Ensembles for Sentiment Analysis
    **Team:** Denizhan, Mohammed, Friedrich | **Supervisor:** Prof. Dr. Onan
    """)

demo.launch()
'''

os.makedirs("/content/demo_app", exist_ok=True)
with open("/content/demo_app/app.py", "w") as f:
    f.write(app_code)
print("✓ app.py updated.")

requirements = """gradio>=4.0.0
transformers>=4.40.0
peft>=0.10.0
torch>=2.0.0
scikit-learn>=1.4.0
numpy>=1.24.0
tiktoken>=0.5.0
sentencepiece>=0.1.99
safetensors>=0.4.0
"""

with open("/content/demo_app/requirements.txt", "w") as f:
    f.write(requirements)

upload_folder(
    folder_path="/content/demo_app",
    repo_id=f"{HF_USERNAME}/imdb-sentiment-demo",
    repo_type="space",
    commit_message="Update: fixed DeBERTa with pooler weights"
)
print(f"✓ Space updated!")
print(f"🔗 https://huggingface.co/spaces/{HF_USERNAME}/imdb-sentiment-demo")

app_code = '''
import gradio as gr
import torch
import numpy as np
import re
import pickle
from transformers import (RobertaTokenizer, RobertaForSequenceClassification,
                          DebertaV2Tokenizer, DebertaV2ForSequenceClassification)
from peft import PeftModel
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression

# ── Constants ─────────────────────────────────────────────────
HF_USERNAME   = "Denizhaan"
ROBERTA_REPO  = f"{HF_USERNAME}/imdb-roberta-lora-v2"
DEBERTA_REPO  = f"{HF_USERNAME}/imdb-deberta-lora-v2"
SVM_REPO      = f"{HF_USERNAME}/imdb-tfidf-svm"
DEVICE        = torch.device("cpu")

# ── Text cleaning ──────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()

# ── Head+Tail truncation (V2: 256/256) ────────────────────────
def head_tail_truncate(text, tokenizer, max_len=512, head_len=256):
    tail_len = max_len - head_len
    tokens = tokenizer(text, add_special_tokens=False,
                       truncation=False, return_tensors=None)
    input_ids      = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    if len(input_ids) > max_len - 2:
        input_ids      = input_ids[:head_len] + input_ids[-tail_len:]
        attention_mask = attention_mask[:head_len] + attention_mask[-tail_len:]
    return tokenizer(tokenizer.decode(input_ids), max_length=max_len,
                     padding="max_length", truncation=True,
                     return_tensors="pt")

# ── Load models (cached after first load) ─────────────────────
print("Loading models...")

# TF-IDF + SVM
tfidf_path = hf_hub_download(SVM_REPO, "tfidf.pkl")
svm_path   = hf_hub_download(SVM_REPO, "svm.pkl")
with open(tfidf_path, "rb") as f: tfidf = pickle.load(f)
with open(svm_path,   "rb") as f: svm   = pickle.load(f)
print("✓ TF-IDF + SVM loaded.")

# RoBERTa + LoRA V2
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_base      = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=2)
roberta_model     = PeftModel.from_pretrained(roberta_base, ROBERTA_REPO)
roberta_model.eval()
print("✓ RoBERTa + LoRA V2 loaded.")

# DeBERTa + LoRA V2
deberta_tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
deberta_base      = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=2, torch_dtype=torch.float32)
deberta_model     = PeftModel.from_pretrained(deberta_base, DEBERTA_REPO)
deberta_model     = deberta_model.to(torch.float32)
deberta_model.eval()
print("✓ DeBERTa + LoRA V2 loaded.")

print("All models ready!")

# ── Prediction function ────────────────────────────────────────
def predict(review):
    if not review.strip():
        return "Please enter a review.", "", "", "", ""

    text       = clean_text(review)
    text_lower = text.lower()

    # 1. TF-IDF + SVM
    svm_pred  = svm.predict(tfidf.transform([text_lower]))[0]
    svm_label = "✅ Positive" if svm_pred == 1 else "❌ Negative"

    # 2. RoBERTa + LoRA
    with torch.no_grad():
        r_enc   = head_tail_truncate(text, roberta_tokenizer)
        r_out   = roberta_model(**r_enc)
        r_probs = torch.softmax(r_out.logits, dim=-1)[0]
        r_pred  = r_probs.argmax().item()
        r_conf  = r_probs[r_pred].item()
    r_label = f"{'✅ Positive' if r_pred == 1 else '❌ Negative'} ({r_conf:.1%})"

    # 3. DeBERTa + LoRA
    with torch.no_grad():
        d_enc   = head_tail_truncate(text, deberta_tokenizer)
        d_out   = deberta_model(**d_enc)
        d_probs = torch.softmax(d_out.logits, dim=-1)[0]
        d_pred  = d_probs.argmax().item()
        d_conf  = d_probs[d_pred].item()
    d_label = f"{'✅ Positive' if d_pred == 1 else '❌ Negative'} ({d_conf:.1%})"

    # 4. Meta-Learner (Logistic Regression on RoBERTa + DeBERTa probs)
    r_pos = r_probs[1].item()
    d_pos = d_probs[1].item()
    meta_input = np.array([[r_pos, d_pos]])

    # Simple weighted combination (LR weights from training)
    meta_score = r_pos * 4.28 + d_pos * 2.84
    meta_pred  = 1 if meta_score > (4.28 + 2.84) / 2 else 0
    meta_label = f"{'✅ Positive' if meta_pred == 1 else '❌ Negative'}"

    return svm_label, r_label, d_label, meta_label

# ── Gradio Interface ───────────────────────────────────────────
examples = [
    ["I went into this movie with fairly low expectations and ended up being pleasantly surprised. The story is engaging from beginning to end, and the pacing never really drags. What impressed me most was the cast. Every major actor delivered a convincing performance, and several scenes felt genuinely emotional rather than forced."],
    ["I honestly struggled to finish this movie. The plot is predictable, the dialogue feels unnatural, and most of the characters are so poorly developed that it is difficult to care about what happens to them. Several scenes seem to exist only to increase the runtime and contribute very little to the overall story."],
    ["I wouldn't say this movie is completely terrible. In fact, there are moments where it almost feels like it might become genuinely interesting. The actors are not bad, the visual effects are not awful, and the story is not entirely without ideas."],
]

with gr.Blocks(title="IMDB Sentiment Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 IMDB Sentiment Analysis
    ### Parameter-Efficient Fine-Tuning (LoRA) + Stacking Ensemble
    Compare predictions from **4 models**: TF-IDF+SVM, RoBERTa+LoRA, DeBERTa+LoRA, and Meta-Learner
    """)

    with gr.Row():
        input_text = gr.Textbox(
            label="Movie Review",
            placeholder="Enter your movie review here...",
            lines=5
        )

    btn = gr.Button("Analyze Sentiment", variant="primary")

    with gr.Row():
        out_svm     = gr.Textbox(label="TF-IDF + SVM (Baseline)")
        out_roberta = gr.Textbox(label="RoBERTa + LoRA V2")
        out_deberta = gr.Textbox(label="DeBERTa + LoRA V2")
        out_meta    = gr.Textbox(label="Meta-Learner (Ensemble)")

    btn.click(fn=predict, inputs=input_text,
              outputs=[out_svm, out_roberta, out_deberta, out_meta])

    gr.Examples(examples=examples, inputs=input_text)

    gr.Markdown("""
    ---
    **Project:** Parameter-Efficient Fine-Tuning Ensembles for Sentiment Analysis
    **Team:** Denizhan, Mohammed, Friedrich | **Supervisor:** Prof. Dr. Onan
    """)

demo.launch()
'''

# Save app.py
os.makedirs("/content/demo_app", exist_ok=True)
with open("/content/demo_app/app.py", "w") as f:
    f.write(app_code)
print("✓ app.py created.")

from huggingface_hub import upload_file

# requirements.txt for the Space
requirements = """gradio>=4.0.0
transformers>=4.40.0
peft>=0.10.0
torch>=2.0.0
scikit-learn>=1.4.0
numpy>=1.24.0
"""

with open("/content/demo_app/requirements.txt", "w") as f:
    f.write(requirements)
print("✓ requirements.txt created.")

# Upload app.py and requirements.txt to Space
upload_folder(
    folder_path="/content/demo_app",
    repo_id=f"{HF_USERNAME}/imdb-sentiment-demo",
    repo_type="space",
    commit_message="Add Gradio demo app"
)
print(f"✓ Space updated!")
print(f"🔗 Demo URL: https://huggingface.co/spaces/{HF_USERNAME}/imdb-sentiment-demo")

requirements = """gradio>=4.0.0
transformers>=4.40.0
peft>=0.10.0
torch>=2.0.0
scikit-learn>=1.4.0
numpy>=1.24.0
tiktoken>=0.5.0
sentencepiece>=0.1.99
"""

with open("/content/demo_app/requirements.txt", "w") as f:
    f.write(requirements)

upload_folder(
    folder_path="/content/demo_app",
    repo_id=f"{HF_USERNAME}/imdb-sentiment-demo",
    repo_type="space",
    commit_message="Fix: add tiktoken and sentencepiece dependencies"
)
print("✓ Updated and uploaded.")
