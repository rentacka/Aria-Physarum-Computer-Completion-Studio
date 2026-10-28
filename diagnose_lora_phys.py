import torch
from transformers import AutoTokenizer
from core.aria_engraft_deltaloop import AriaEngraftDeltaLoopModel
from core.aria_physarum_safetensor_cortex import AriaPhysarumFrequencyCortex, load_physarum_safetensors
from core.aria_task_lora import AriaDeltaLoopLoRAEngine
from core.deltaloop_sampler import AriaDeltaLoopSampler

device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained('models/gemma4_engraft')
model = AriaEngraftDeltaLoopModel.load_from_gemma4_weights('models/gemma4_engraft/gemma4_engraft_weights.pt', d_model=512, num_layers=12, loop_steps=4).to(device)
model.load_state_dict(torch.load('models/gemma4_rsi5_7_omni/gemma4_rsi5_7_omni_weights.pt', map_location=device, weights_only=True), strict=False)

cortex = AriaPhysarumFrequencyCortex(d_model=512, num_frequencies=256).to(device)
load_physarum_safetensors(cortex, 'models/physarum_safetensors/aria_physarum_zero_omni.safetensors', device=device)
model.attach_physarum_cortex(cortex, gain=0.0)

lora_engine = AriaDeltaLoopLoRAEngine(model=model, r=8, lora_alpha=16.0)
lora_engine.inject_lora()
lora_engine.load_lora_safetensors('models/lora_adapters/aria_task_lora_omni.safetensors', device=device)
model.eval()

sampler = AriaDeltaLoopSampler(model=model, device=device, temperature=0.7, top_k=40, repetition_penalty=1.15, eos_token_id=tokenizer.eos_token_id or 1)

prompt = '<start_of_turn>user\nUSDJPY の M15 で AriaApexSniper のバックテストを実行してください。<end_of_turn>\n<start_of_turn>model\n'
enc = tokenizer(prompt, return_tensors='pt')['input_ids'][0].tolist()

for lora_s, phys_g in [(0.0, 0.0), (1.0, 0.0), (0.0, 0.03), (0.2, 0.0), (1.0, 0.03)]:
    lora_engine.set_lora_scale(lora_s)
    res = sampler.generate_mtp(prompt_ids=enc, max_new_tokens=40, min_new_tokens=10, active_layers=4, loop_steps=2, physarum_gain=phys_g)
    text = tokenizer.decode(res['tokens'], skip_special_tokens=True)
    print(f"LoRA={lora_s}, Phys={phys_g}: {text.strip()[:80]}")
