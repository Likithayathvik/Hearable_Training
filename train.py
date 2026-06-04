"""
train.py — Delta training for boAt Review Analyzer BERT model
Converted from Review_Analysis_delta_training notebook
Runs on GitHub Actions (CPU) automatically
"""

import os
import io
import json
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
import nltk
import gspread

from collections import Counter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

warnings.filterwarnings("ignore")
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# ── CONFIG ──────────────────────────────────────────────────────────────────
GDRIVE_SHEET_ID   = "1uHBQs85jZVoYP5h9lLRQuBxtEoG9Db3CCWZ5n4Dz7xU"
GDRIVE_SHEET_TAB  = "Sheet2"
DRIVE_PARENT_NAME = "RA_model"

RA_ROOT        = "./RA_model"
OLD_MODEL_PATH = f"{RA_ROOT}/_previous"
TIMESTAMP      = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME       = f"RA_model_deltatraining_{TIMESTAMP}"
NEW_MODEL_PATH = f"{RA_ROOT}/{RUN_NAME}"

# ── HYPERPARAMETERS ─────────────────────────────────────────────────────────
EPOCHS                   = 7
BATCH_SIZE               = 8
LR_ENCODER               = 1e-5
LR_HEAD                  = 3e-5
MAX_LEN                  = 160
MIN_SAMPLES              = 25
PATIENCE                 = 3
NEW_DATA_BOOST           = 3
WEIGHT_DECAY             = 0.01
WARMUP_RATIO             = 0.1
MIN_NEW_ROWS             = 10
QUALITY_GUARD_DROP       = 0.03
DOMINANT_LABEL_THRESHOLD = 0.4
MAX_DELTA_RUNS           = 4
HIST_VAL_SAMPLE_N        = 50

# ── GOOGLE DRIVE HELPERS ────────────────────────────────────────────────────
def get_drive_service():
    creds_json = json.loads(os.environ['GOOGLE_CREDENTIALS_JSON'])
    creds = Credentials.from_service_account_info(
        creds_json,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build('drive', 'v3', credentials=creds)

def _find_folder_id(service, name, parent_id=None):
    q = (f"name='{name}' and mimeType='application/vnd.google-apps.folder' "
         f"and trashed=false")
    if parent_id:
        q += f" and '{parent_id}' in parents"
    res = service.files().list(q=q, fields="files(id,name)").execute()
    files = res.get('files', [])
    return files[0]['id'] if files else None

def _create_folder(service, name, parent_id=None):
    body = {'name': name, 'mimeType': 'application/vnd.google-apps.folder'}
    if parent_id:
        body['parents'] = [parent_id]
    return service.files().create(body=body, fields='id').execute()['id']

def _list_subfolders(service, parent_id):
    res = service.files().list(
        q=(f"'{parent_id}' in parents and "
           f"mimeType='application/vnd.google-apps.folder' and trashed=false"),
        fields="files(id,name,createdTime)",
        orderBy="createdTime desc",
    ).execute()
    return res.get('files', [])

def download_drive_folder(service, folder_id, local_path):
    os.makedirs(local_path, exist_ok=True)
    files = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id,name)"
    ).execute().get('files', [])
    for f in files:
        req  = service.files().get_media(fileId=f['id'])
        buf  = io.BytesIO()
        dl   = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        with open(os.path.join(local_path, f['name']), 'wb') as out:
            out.write(buf.getvalue())
        print(f"    ⬇  {f['name']}")

# ── UPDATED: upload with retry + chunked upload ──────────────────────────────
def upload_folder_to_drive(service, local_path, drive_folder_name, parent_id=None):
    folder_id = _find_folder_id(service, drive_folder_name, parent_id=parent_id)
    if folder_id is None:
        folder_id = _create_folder(service, drive_folder_name, parent_id=parent_id)
    for filename in os.listdir(local_path):
        filepath = os.path.join(local_path, filename)
        if not os.path.isfile(filepath):
            continue
        media = MediaFileUpload(filepath, resumable=True, chunksize=1024*1024)
        q = (f"name='{filename}' and '{folder_id}' in parents "
             f"and trashed=false")
        existing = service.files().list(q=q, fields="files(id)").execute().get('files', [])

        # Retry up to 3 times
        for attempt in range(3):
            try:
                if existing:
                    service.files().update(
                        fileId=existing[0]['id'], media_body=media,
                        supportsAllDrives=True
                    ).execute()
                else:
                    service.files().create(
                        body={'name': filename, 'parents': [folder_id]},
                        media_body=media, fields='id',
                        supportsAllDrives=True
                    ).execute()
                print(f"    ⬆  {filename}")
                break
            except Exception as e:
                print(f"    ⚠️ Attempt {attempt+1} failed for {filename}: {e}")
                if attempt == 2:
                    raise
    return folder_id

# ── DATASET ──────────────────────────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts, max_length=max_len,
            padding=True, truncation=True, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx],
        }

# ── CLEAN DATA ───────────────────────────────────────────────────────────────
def _clean(d):
    d = d.copy()
    for col in d.select_dtypes(include="object").columns:
        d[col] = d[col].astype(str).str.strip()
    d = d.dropna(subset=["Reviews", "Consolidated Reason"])
    d = d[(d["Reviews"] != "") & (d["Consolidated Reason"] != "")].reset_index(drop=True)
    return d

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("boAt Review Analyzer — Delta Training")
    print("=" * 60)

    # ── STEP 1: Connect to Google Drive ──────────────────────────────────────
    print("\nSTEP 1 — Connecting to Google Drive...")
    drive_service = get_drive_service()
    print("  ✅ Drive connected")

    # ── STEP 2: Load data from Google Sheet ──────────────────────────────────
    print("\nSTEP 2 — Loading data from Google Sheet...")
    creds_json = json.loads(os.environ['GOOGLE_CREDENTIALS_JSON'])
    creds = Credentials.from_service_account_info(
        creds_json,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
    )
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(GDRIVE_SHEET_ID).worksheet(GDRIVE_SHEET_TAB)
    df = pd.DataFrame(ws.get_all_records())
    print(f"  ✅ Loaded {df.shape[0]} rows from Google Sheet")

    # ── STEP 3: Download latest model from Drive ──────────────────────────────
    print("\nSTEP 3 — Loading previous training run from Drive...")
    os.makedirs(RA_ROOT, exist_ok=True)

    parent_id = _find_folder_id(drive_service, DRIVE_PARENT_NAME)
    if parent_id is None:
        raise SystemExit(f"❌ Drive folder '{DRIVE_PARENT_NAME}' not found.")

    subs = _list_subfolders(drive_service, parent_id)
    run_subs = [
        s for s in subs
        if s['name'].startswith("RA_model_fulltraining_")
        or s['name'].startswith("RA_model_deltatraining_")
    ]
    if not run_subs:
        raise SystemExit("❌ No training run folders found in Drive.")

    latest_run    = run_subs[0]
    prev_run_name = latest_run['name']
    prev_run_id   = latest_run['id']
    print(f"  Latest run : {prev_run_name}")

    if os.path.isdir(OLD_MODEL_PATH):
        shutil.rmtree(OLD_MODEL_PATH)
    print(f"  Downloading → {OLD_MODEL_PATH}")
    download_drive_folder(drive_service, prev_run_id, OLD_MODEL_PATH)
    print("  ✅ Restored from Drive")

    tokenizer = BertTokenizerFast.from_pretrained(OLD_MODEL_PATH, local_files_only=True)
    old_model = BertForSequenceClassification.from_pretrained(OLD_MODEL_PATH, local_files_only=True)

    old_label2id   = old_model.config.label2id
    OLD_NUM_LABELS = len(old_label2id)
    print(f"  Old model loaded with {OLD_NUM_LABELS} classes")

    # ── STEP 4: Split historical + new rows ──────────────────────────────────
    print("\nSTEP 4 — Splitting historical + new rows...")
    state_path = os.path.join(OLD_MODEL_PATH, "training_state.json")
    if not os.path.exists(state_path):
        raise SystemExit("❌ training_state.json not found.")

    with open(state_path) as f:
        prior_state = json.load(f)

    last_hist_rows  = int(prior_state["last_trained_row"])
    delta_run_count = int(prior_state.get("delta_run_count", 0))
    print(f"  last_trained_row : {last_hist_rows}")
    print(f"  delta_run_count  : {delta_run_count}")

    raw_df        = df.copy()
    raw_hist_df   = raw_df.iloc[:last_hist_rows].reset_index(drop=True)
    raw_weekly_df = raw_df.iloc[last_hist_rows:].reset_index(drop=True)

    hist_df   = _clean(raw_hist_df)
    weekly_df = _clean(raw_weekly_df)

    print(f"  Historical (cleaned) : {len(hist_df)}")
    print(f"  New rows   (cleaned) : {len(weekly_df)}")

    full_df = pd.concat([hist_df, weekly_df], ignore_index=True)

    if len(weekly_df) < MIN_NEW_ROWS:
        raise SystemExit(
            f"\n❌ ABORT: only {len(weekly_df)} new rows. "
            f"Need at least {MIN_NEW_ROWS}."
        )

    # ── STEP 5: Reconcile label set ───────────────────────────────────────────
    print("\nSTEP 5 — Reconciling label set...")
    all_labels_seen = set(hist_df["Consolidated Reason"]).union(
        set(weekly_df["Consolidated Reason"])
    )
    new_labels = sorted(l for l in all_labels_seen if l not in old_label2id)

    label2id = dict(old_label2id)
    for lbl in new_labels:
        label2id[lbl] = len(label2id)
    id2label   = {i: l for l, i in label2id.items()}
    num_labels = len(label2id)

    if new_labels:
        print(f"  Found {len(new_labels)} new label(s): {new_labels}")
    else:
        print("  No new labels.")
    print(f"  Total labels : {num_labels}")

    # ── STEP 6: Prepare model ─────────────────────────────────────────────────
    print("\nSTEP 6 — Preparing model...")
    if num_labels == OLD_NUM_LABELS:
        model = old_model
        model.config.id2label = id2label
        model.config.label2id = label2id
        print("  No head expansion needed.")
    else:
        model = BertForSequenceClassification.from_pretrained(
            OLD_MODEL_PATH,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        with torch.no_grad():
            old_W = old_model.classifier.weight.data
            old_b = old_model.classifier.bias.data
            model.classifier.weight.data[:OLD_NUM_LABELS, :] = old_W
            model.classifier.bias.data[:OLD_NUM_LABELS]      = old_b
        print(f"  Classifier expanded {OLD_NUM_LABELS} → {num_labels}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device : {device}")

    # ── STEP 7: Prepare training data ─────────────────────────────────────────
    print("\nSTEP 7 — Preparing training data...")

    boosted_weekly = pd.concat([weekly_df] * NEW_DATA_BOOST, ignore_index=True)
    train_df = pd.concat([hist_df, boosted_weekly], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    texts  = train_df["Reviews"].tolist()
    labels = [label2id[l] for l in train_df["Consolidated Reason"]]

    t_texts, v_texts, t_labels, v_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )

    train_dataset = ReviewDataset(t_texts, t_labels, tokenizer, MAX_LEN)
    val_dataset   = ReviewDataset(v_texts, v_labels, tokenizer, MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    print(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    class_weights = compute_class_weight(
        'balanced', classes=np.unique(labels), y=labels
    )
    weight_tensor = torch.tensor(
        [class_weights[i] if i < len(class_weights) else 1.0
         for i in range(num_labels)],
        dtype=torch.float
    ).to(device)

    # ── STEP 8: Train ─────────────────────────────────────────────────────────
    print("\nSTEP 8 — Training...")
    optimizer = AdamW([
        {'params': model.bert.parameters(),       'lr': LR_ENCODER},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    best_val_loss  = float('inf')
    patience_count = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels   = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_labels   = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = loss_fn(outputs.logits, batch_labels)
                val_loss += loss.item()
                preds    = outputs.logits.argmax(dim=-1)
                correct += (preds == batch_labels).sum().item()
                total   += batch_labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = correct / total

        print(f"  Epoch {epoch+1}/{EPOCHS} — "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  "
              f"Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss - QUALITY_GUARD_DROP:
            best_val_loss  = avg_val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # ── STEP 9: Save model ────────────────────────────────────────────────────
    print("\nSTEP 9 — Saving model...")
    os.makedirs(NEW_MODEL_PATH, exist_ok=True)
    model.save_pretrained(NEW_MODEL_PATH)
    tokenizer.save_pretrained(NEW_MODEL_PATH)

    training_state = {
        "last_trained_row": len(raw_df),
        "delta_run_count":  delta_run_count + 1,
        "trained_on":       TIMESTAMP,
        "training_type":    "delta",
        "val_loss":         best_val_loss,
        "num_labels":       num_labels,
    }
    with open(os.path.join(NEW_MODEL_PATH, "training_state.json"), "w") as f:
        json.dump(training_state, f, indent=2)

    print(f"  ✅ Model saved to {NEW_MODEL_PATH}")

    # ── STEP 10: Upload to Google Drive ───────────────────────────────────────
    print("\nSTEP 10 — Uploading model to Google Drive...")
    parent_id = _find_folder_id(drive_service, DRIVE_PARENT_NAME)
    if parent_id is None:
        parent_id = _create_folder(drive_service, DRIVE_PARENT_NAME)

    upload_folder_to_drive(
        drive_service, NEW_MODEL_PATH, RUN_NAME, parent_id=parent_id
    )
    print(f"  ✅ Uploaded {RUN_NAME} to Drive/{DRIVE_PARENT_NAME}/")
    print("\n" + "=" * 60)
    print("Delta training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
