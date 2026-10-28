import torch
import os

p = 'models/gemma4_rsi5_7_omni/gemma4_rsi5_7_omni_weights.pt'
state = torch.load(p, map_location='cpu')

if 'embed_tokens.weight' in state:
    print("Stripping embed_tokens.weight (saving 2.6GB disk space)...")
    del state['embed_tokens.weight']

torch.save(state, p)
sz_mb = os.path.getsize(p) / (1024 * 1024)
print(f"✅ Successfully streamlined {p} to {sz_mb:.2f} MB (Keys: {len(state)})")
