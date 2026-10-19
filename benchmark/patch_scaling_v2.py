import re

patch_content = '''CHUNK_1_OVERVIEW = (
    "Zephyr Dynamics Inc. was founded in 2011 by Dr. Priya Nandakumar and Marcus Löbell "
    "in Reykjavik, Iceland. The company specializes in atmospheric data harvesting systems "
    "and micro-drone swarm technology. Zephyr Dynamics has 847 full-time employees across "
    "6 global offices. Annual revenue for 2023 was $412 million USD."
)

CHUNK_2_LEADERSHIP = (
    "Dr. Priya Nandakumar serves as Chief Executive Officer (CEO) and Co-Founder in Reykjavik. "
    "Marcus Löbell serves as Chief Technology Officer (CTO). "
    "Saoirse Hennessey serves as Chief Financial Officer (CFO). "
    "Jorge Fonseca serves as Chief Legal Officer (CLO), having joined the executive team in 2019."
)

CHUNK_3_AEROSENSE = (
    "The AeroSense X7 is Zephyr Dynamics flagship atmospheric sensor array, launched in Q2 2022. "
    "It measures temperature, humidity, particulate matter, and wind shear at altitudes up to 12,000 meters. "
    "The X7 uses a proprietary QuantumCell battery that lasts 72 hours on a single charge. Unit price: $8,400 USD."
)

CHUNK_4_SWARMKIT = (
    "SwarmKit Pro is a micro-drone coordination platform released in January 2023. "
    "It supports simultaneous control of up to 2,048 drones using the ZephyrMesh v3 protocol. "
    "Licensing is subscription-based: $1,200/month for standard capacity, "
    "and $3,800/month for the full 2,048-drone capacity tier."
)'''

# Read benchmark/run_hacer_scaling_matrix.py
with open("benchmark/run_hacer_scaling_matrix.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace chunks 1-4
pattern = r"CHUNK_1_OVERVIEW = \([\s\S]*?CHUNK_4_SWARMKIT = \([\s\S]*?\)\n"
text = re.sub(pattern, patch_content + "\n", text, count=1)

# Update session file pattern and log paths
text = text.replace("hacer_scaling_k", "hacer_v2_k")
text = text.replace("smoke_test_trials.jsonl", "smoke_test_trials_v2.jsonl")
text = text.replace("smoke_test_summary.json", "smoke_test_summary_v2.json")

# Add telemetry fields in execute_single_trial return dict
old_return = '''        "tokens_predicted": tok_pred,
        "tokens_loaded": tok_loaded,
        "tokens_cached": tok_cached,
        "response_text": resp_text,
        "prompt_sha256": prompt_sha256,
        "active_n_ctx": dev_res.get("active_n_ctx") or p1.get("active_n_ctx"),'''

new_return = '''        "tokens_predicted": tok_pred,
        "tokens_loaded": tok_loaded,
        "tokens_cached": tok_cached,
        "truncated": p1.get("truncated", False),
        "stopped_eos": p1.get("stopped_eos", False),
        "stopped_limit": p1.get("stopped_limit", False),
        "full_prompt_tokens_native": p1.get("full_prompt_token_count"),
        "prefix_tokens_native": p1.get("prefix_token_count"),
        "prompt_n_evaluated": timings.get("prompt_n"),
        "context_headroom": p1.get("context_headroom"),
        "response_text": resp_text,
        "prompt_sha256": prompt_sha256,
        "active_n_ctx": dev_res.get("active_n_ctx") or p1.get("active_n_ctx"),'''

text = text.replace(old_return, new_return)

with open("benchmark/run_hacer_scaling_matrix.py", "w", encoding="utf-8") as f:
    f.write(text)

print("[+] Patched benchmark/run_hacer_scaling_matrix.py successfully!")

# Also patch benchmark/test_scaling_matrix.py
with open("benchmark/test_scaling_matrix.py", "r", encoding="utf-8") as f:
    test_text = f.read()

test_text = re.sub(pattern, patch_content + "\n", test_text, count=1)

# Update prompt token estimation in test
test_text = test_text.replace("prompt_tokens = 365", "prompt_tokens = 386")
test_text = test_text.replace("reconcile_tokens(365, 85)", "reconcile_tokens(386, 81)")
test_text = test_text.replace("self.assertEqual(reconcile_tokens(365, 85), 281)", "self.assertEqual(reconcile_tokens(386, 81), 306)")
test_text = test_text.replace("self.assertEqual(reconcile_tokens(365, 175), 191)", "self.assertEqual(reconcile_tokens(386, 155), 232)")
test_text = test_text.replace("self.assertEqual(reconcile_tokens(365, 330), 36)", "self.assertEqual(reconcile_tokens(386, 332), 55)")

with open("benchmark/test_scaling_matrix.py", "w", encoding="utf-8") as f:
    f.write(test_text)

print("[+] Patched benchmark/test_scaling_matrix.py successfully!")
