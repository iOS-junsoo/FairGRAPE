import subprocess
import requests
import time

SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/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"
BUFFER_SECONDS = 5

def send_to_slack(text):
    if not text.strip():
        return
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": f"```{text}```"}, timeout=5)
    except Exception as e:
        print(f"[slack_logger] Slack 전송 실패 (무시): {e}")

command = [
    "/workspace/fairgrape_env_gpu/bin/python", "main_test.py",
    "--checkpoint", "trained_model/unpruned/CelebA_unpruned_classes_bygender_mobilenetv2_0.pt",
    "--dataset", "CelebA", "--network", "mobilenetv2",
    "--prune_type", "FairGRAPE", "--loss_type", "classes",
    "--sensitive_group", "gender", "--prune_rate", "0.9",
    "--no_init_train", "--batch", "256", "--retrain_iter", "3",
    "--stop_batch", "10000", "--impt", "2", "--no_grl",
    "--skip_readable_check", "--seed", "100",
    "--keep_per_iter", "0.95",
]

process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

send_to_slack("🚀 실험 시작")

buffer = []
last_sent = time.time()

for line in process.stdout:
    print(line, end="")
    buffer.append(line)

    if time.time() - last_sent >= BUFFER_SECONDS or len(buffer) > 20:
        send_to_slack("".join(buffer))
        buffer = []
        last_sent = time.time()

if buffer:
    send_to_slack("".join(buffer))

send_to_slack("✅ 실험 완료")