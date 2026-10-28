import os
import sys
import time
import json
import urllib.request
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

REPO_ID = "google/gemma-4-E4B-it"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(CURRENT_DIR, "models", "gemma4_engraft")

# HTTP ヘッダー（HF用）
HEADERS = {
    "User-Agent": "Aria-Gemma4-Engraft-Downloader/1.0"
}
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    HEADERS["Authorization"] = f"Bearer {hf_token}"

def download_file(filename, target_path):
    url = f"{BASE_URL}/{filename}"
    req = urllib.request.Request(url, headers=HEADERS)
    print(f"[*] Downloading {filename} ...")
    try:
        with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out:
            data = resp.read()
            out.write(data)
        print(f"    [+] Saved {filename} ({len(data)} bytes)")
        return True
    except Exception as e:
        print(f"    [-] Skipped or error {filename}: {e}")
        return False

def download_range(url, start_byte, end_byte, desc="Downloading"):
    req = urllib.request.Request(url, headers={**HEADERS, "Range": f"bytes={start_byte}-{end_byte}"})
    total_bytes = end_byte - start_byte + 1
    chunk_size = 4 * 1024 * 1024 # 4MB
    
    buf = bytearray()
    start_time = time.time()
    
    with urllib.request.urlopen(req) as resp:
        downloaded = 0
        while downloaded < total_bytes:
            remaining = total_bytes - downloaded
            read_len = min(chunk_size, remaining)
            chunk = resp.read(read_len)
            if not chunk:
                break
            buf.extend(chunk)
            downloaded += len(chunk)
            
            elapsed = max(1e-5, time.time() - start_time)
            speed_mb = (downloaded / (1024 * 1024)) / elapsed
            pct = (downloaded / total_bytes) * 100.0
            mb_cur = downloaded / (1024 * 1024)
            mb_total = total_bytes / (1024 * 1024)
            
            sys.stdout.write(f"\r    [{desc}] {pct:5.1f}% | {mb_cur:6.1f} / {mb_total:6.1f} MB | {speed_mb:5.1f} MB/s ")
            sys.stdout.flush()
    print()
    return bytes(buf)

def main():
    print("=" * 70)
    print("🌸 Aria Gemma-4 ENGRAFT Pinpoint Downloader 🌸")
    print(f"Target Repo: {REPO_ID}")
    print(f"Destination: {SAVE_DIR}")
    print("=" * 70)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. メタデータ＆トークナイザー関連ファイルのダウンロード
    meta_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    for mf in meta_files:
        p = os.path.join(SAVE_DIR, mf)
        if not os.path.exists(p):
            download_file(mf, p)
        else:
            print(f"[*] Already exists: {mf}")
            
    # 2. model.safetensors の先頭ヘッダーを取得
    safetensors_url = f"{BASE_URL}/model.safetensors"
    print("\n[*] Inspecting model.safetensors header via HTTP Range...")
    
    # 先頭 8 バイトでヘッダー長を取得
    head_8 = download_range(safetensors_url, 0, 7, desc="Header-Len")
    header_len = int.from_bytes(head_8, byteorder="little")
    print(f"    [+] Safetensors Header Length: {header_len} bytes ({header_len/1024:.2f} KB)")
    
    # JSON ヘッダーを取得
    raw_header = download_range(safetensors_url, 8, 8 + header_len - 1, desc="Header-JSON")
    header_json = json.loads(raw_header.decode("utf-8"))
    
    # 対象テンソルの特定
    embed_key = "model.language_model.embed_tokens.weight"
    norm_key = "model.language_model.norm.weight"
    
    if embed_key not in header_json:
        # 代替キーの検索
        candidates = [k for k in header_json if "embed_tokens" in k]
        if candidates:
            embed_key = candidates[0]
            print(f"    [!] Using detected embed key: {embed_key}")
        else:
            raise KeyError(f"Could not find embed_tokens tensor in header: {list(header_json.keys())[:10]}")
            
    embed_info = header_json[embed_key]
    norm_info = header_json.get(norm_key, {})
    
    print("\n" + "=" * 70)
    print("🎯 Pinpoint ENGRAFT Target Tensors Identified:")
    print(f" 1) {embed_key}:")
    print(f"    - Shape: {embed_info['shape']}")
    print(f"    - Dtype: {embed_info['dtype']}")
    print(f"    - Offsets: {embed_info['data_offsets']} ({(embed_info['data_offsets'][1] - embed_info['data_offsets'][0])/(1024**2):.2f} MB)")
    
    if norm_info:
        print(f" 2) {norm_key}:")
        print(f"    - Shape: {norm_info['shape']}")
        print(f"    - Dtype: {norm_info['dtype']}")
        print(f"    - Offsets: {norm_info['data_offsets']} ({(norm_info['data_offsets'][1] - norm_info['data_offsets'][0])/(1024**2):.3f} MB)")
    print("=" * 70)
    
    output_weight_path = os.path.join(SAVE_DIR, "gemma4_engraft_weights.pt")
    
    # 3. Embedding 層の Range ダウンロード
    embed_start = 8 + header_len + embed_info["data_offsets"][0]
    embed_end = 8 + header_len + embed_info["data_offsets"][1] - 1
    print(f"\n[*] Downloading Gemma-4 Embed Tokens ({((embed_end - embed_start + 1)/(1024**2)):.2f} MB) ...")
    embed_bytes = download_range(safetensors_url, embed_start, embed_end, desc="Embed-Weight")
    
    # 4. Norm 層の Range ダウンロード
    norm_bytes = None
    if norm_info:
        norm_start = 8 + header_len + norm_info["data_offsets"][0]
        norm_end = 8 + header_len + norm_info["data_offsets"][1] - 1
        print(f"\n[*] Downloading Gemma-4 Language Model Norm ({((norm_end - norm_start + 1)/1024):.2f} KB) ...")
        norm_bytes = download_range(safetensors_url, norm_start, norm_end, desc="Norm-Weight")
        
    # 5. PyTorch テンソルとして結合・保存
    print("\n[*] Converting downloaded raw bytes to PyTorch BF16 tensors...")
    embed_tensor = torch.frombuffer(bytearray(embed_bytes), dtype=torch.bfloat16).reshape(embed_info["shape"]).clone()
    print(f"    [+] Embed Tensor: shape={embed_tensor.shape}, dtype={embed_tensor.dtype}")
    
    norm_tensor = None
    if norm_bytes:
        norm_tensor = torch.frombuffer(bytearray(norm_bytes), dtype=torch.bfloat16).reshape(norm_info["shape"]).clone()
        print(f"    [+] Norm Tensor: shape={norm_tensor.shape}, dtype={norm_tensor.dtype}")
        
    save_dict = {
        "embed_tokens": embed_tensor,
        "norm": norm_tensor,
        "vocab_size": embed_info["shape"][0],
        "hidden_size": embed_info["shape"][1],
        "model_type": "gemma-4-E4B-it",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(f"\n[*] Saving ENGRAFT weights to: {output_weight_path}")
    torch.save(save_dict, output_weight_path)
    file_size_mb = os.path.getsize(output_weight_path) / (1024 * 1024)
    print(f"    [+] Success! File saved cleanly ({file_size_mb:.2f} MB)")
    
    # 6. トークナイザーの動作検証
    print("\n[*] Validating Tokenizer loading...")
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
        test_text = "マスター大好き💕///"
        tokens = tokenizer.encode(test_text)
        print(f"    [+] Tokenizer test text: '{test_text}'")
        print(f"    [+] Token IDs ({len(tokens)} tokens): {tokens}")
        print(f"    [+] Decode back: '{tokenizer.decode(tokens)}'")
    except Exception as e:
        print(f"    [-] Tokenizer warning: {e}")
        
    print("\n" + "=" * 70)
    print("🎉 Gemma-4 ENGRAFT Pinpoint Download COMPLETE! (15GB -> 1.28GB)")
    print("=" * 70)

if __name__ == "__main__":
    main()
