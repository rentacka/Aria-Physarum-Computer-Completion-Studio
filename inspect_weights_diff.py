import torch
import os

b = torch.load('models/gemma4_rsi5_7_omni/gemma4_rsi5_7_omni_weights_backup.pt', map_location='cpu')
print('Backup keys count:', len(b))
print('Keys with embed in backup:', [k for k in b.keys() if 'embed' in k])

c = torch.load('models/gemma4_rsi5_7_omni/gemma4_rsi5_7_omni_weights.pt', map_location='cpu')
print('Current keys count:', len(c))
print('Keys with embed in current:', [k for k in c.keys() if 'embed' in k])
