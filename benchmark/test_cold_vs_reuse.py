import json
import base64
import time
import subprocess
import uuid
from pathlib import Path

def adb(cmd):
    return subprocess.run(['adb', '-s', '36fef3202145'] + cmd, capture_output=True, text=True, check=True).stdout.strip()

trigger_path = '/sdcard/Android/data/com.chat/files/benchmark_trigger.json'
result_path = '/sdcard/Android/data/com.chat/files/benchmark_result.json'
output_path = '/sdcard/Android/data/com.chat/files/hacer_phase1_result.json'
session_path = '/sdcard/Android/data/com.chat/files/hacer_phase1_session.bin'

prefix = "Excerpts:\n[1] AeroSense X7 is Zephyr Dynamics flagship atmospheric sensor array launched in Q2 2022. It measures temperature, humidity, particulate matter, ozone levels, and wind shear at altitudes up to 12,000 meters. The X7 uses a proprietary QuantumCell battery that lasts 72 hours on a single charge. It weighs 2.3 kg and has an operating temperature range of -60C to +85C. Unit price: $1,400 USD. Volume discount of 15% applies for orders above 50 units.\n\n"
suffix = "Question: What is the battery duration and unit price of the AeroSense X7?"

def send_trigger(t):
    encoded = base64.b64encode(json.dumps(t).encode('utf-8')).decode('ascii')
    adb(['shell', f'rm -f {result_path} {trigger_path} {output_path}'])
    adb(['shell', f'echo {encoded} | base64 -d > {trigger_path}'])
    deadline = time.time() + 180
    while time.time() < deadline:
        if adb(['shell', f'test -f {result_path} && echo ready || true']) == 'ready':
            return json.loads(adb(['shell', f'cat {result_path}']))
        time.sleep(1.5)
    raise TimeoutError('Timeout')

print('Step 1: Flushing context with unrelated query...')
flush_res = send_trigger({
    'requestId': uuid.uuid4().hex,
    'resultPath': result_path,
    'query': 'Explain photosynthesis in plants and chlorophyll absorption spectra.',
    'chatMode': 'general',
    'inferenceOptions': {'mode': 'baseline', 'nPredict': 1}
})
print('Context flushed successfully.')

print('Step 2: Running true cold baseline (no session loaded, context was flushed)...')
baseline_res = send_trigger({
    'requestId': uuid.uuid4().hex,
    'resultPath': result_path,
    'query': 'HACER Cold Baseline',
    'chatMode': 'general',
    'phase1': 'baseline',
    'phase1Prefix': prefix,
    'phase1Suffix': suffix,
    'phase1SessionPath': session_path,
    'phase1OutputPath': output_path,
    'phase1NPredict': 1,
})
p1 = baseline_res.get('phase1', {})
timings = p1.get('timings', {})
print('Cold baseline result:')
print('prompt_n:', timings.get('prompt_n'))
print('prompt_ms:', timings.get('prompt_ms'))
print('prompt_tok/s:', timings.get('prompt_per_second'))
print('elapsed_wall_ms:', p1.get('elapsed_wall_ms'))

print('\nStep 3: Flushing context again before testing Flash-loaded reuse...')
send_trigger({
    'requestId': uuid.uuid4().hex,
    'resultPath': result_path,
    'query': 'Describe plate tectonics and continental drift theory in geology.',
    'chatMode': 'general',
    'inferenceOptions': {'mode': 'baseline', 'nPredict': 1}
})
print('Context flushed again.')

print('Step 4: Running Flash session reuse (context was flushed, session loaded from disk)...')
reuse_res = send_trigger({
    'requestId': uuid.uuid4().hex,
    'resultPath': result_path,
    'query': 'HACER Flash Reuse',
    'chatMode': 'general',
    'phase1': 'reuse',
    'phase1Prefix': prefix,
    'phase1Suffix': suffix,
    'phase1SessionPath': session_path,
    'phase1OutputPath': output_path,
    'phase1NPredict': 1,
})
p1_r = reuse_res.get('phase1', {})
timings_r = p1_r.get('timings', {})
print('Flash session reuse result:')
print('tokens_loaded:', p1_r.get('tokens_loaded'))
print('prompt_n:', timings_r.get('prompt_n'))
print('prompt_ms:', timings_r.get('prompt_ms'))
print('prompt_tok/s:', timings_r.get('prompt_per_second'))
print('elapsed_wall_ms:', p1_r.get('elapsed_wall_ms'))

cold_ms = timings.get('prompt_ms', 0)
reuse_ms = timings_r.get('prompt_ms', 0)
savings_ms = cold_ms - reuse_ms
pct = (savings_ms / cold_ms * 100) if cold_ms > 0 else 0
print(f'\n=== NET BENEFIT OF EXACT-PREFIX CACHE REUSE FROM FLASH ===')
print(f'Cold prompt prefill: {cold_ms} ms')
print(f'Flash-loaded prefill: {reuse_ms} ms')
print(f'Prefill latency saved: {savings_ms} ms ({pct:.1f}% reduction)')
