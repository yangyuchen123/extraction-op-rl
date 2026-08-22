import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agent_system.environments.env_package.extraction_ops import ExtractionOpsWorld
from agent_system.environments.env_package.extraction_ops.projection import extraction_ops_projection
from agent_system.environments.prompts.extraction_ops import EXTRACTION_OPS_TEMPLATE

MODEL = "/root/autodl-tmp/models/Qwen3-0.6B-extraction-ops-random-sft-fixed"
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

for seed in (0, 1, 2):
    w = ExtractionOpsWorld(seed=seed, random_maps=True)
    obs, _ = w.reset(seed)
    print(f"=== seed {seed} ===")
    print(f"key@{w.mission['key_location']} target@{w.mission['target_location']} extract@{w.mission['extraction_point']} spawn@{w.spawn}")
    for i in range(20):
        p = tok.apply_chat_template(
            [{"role": "user", "content": EXTRACTION_OPS_TEMPLATE.format(current_observation=obs)}],
            add_generation_prompt=True, tokenize=False,
        )
        b = tok(p, return_tensors="pt").to(m.device)
        with torch.inference_mode():
            g = m.generate(**b, max_new_tokens=64, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        resp = tok.batch_decode(g[:, b["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        acts, _ = extraction_ops_projection([resp])
        a = acts[0]
        r = w.step(a)
        print(f"  step{i}: {a:<30} loc={r.info['location']:<10} inv={r.info['inventory']} reward={r.reward} term={r.info['terminal_reason']}")
        obs = r.observation
        if w.done:
            break
    print()
