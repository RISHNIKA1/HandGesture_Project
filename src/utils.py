import subprocess

def perform_action(action, action_label=None):
    if action_label:
        action_label.config(text=f"Action: {action}")

    if action == "open YouTube":
        subprocess.run(["open", "https://www.youtube.com"])

    elif action == "open WhatsApp":
        subprocess.run(["open", "https://web.whatsapp.com"])

    elif action == "open Notepad":
        subprocess.run(["open", "-a", "TextEdit"])

    elif action == "open Calculator":
        subprocess.run(["open", "-a", "Calculator"])

    elif action == "open Instagram":
        subprocess.run(["open", "https://www.instagram.com"])
