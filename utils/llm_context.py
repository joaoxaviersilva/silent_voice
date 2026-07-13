import json

from utils.groq_client import client

from config import LLM_MODEL


def get_theme_and_vocab(text):
    prompt = f"""
    Contexto: "{text}"
    Gere tema e até 15 palavras possíveis.
    JSON: {{"theme":"","vocab":[]}}
    """

    r = client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2
    )

    try:
        data = json.loads(r.choices[0].message.content)
        return data["theme"], [w.upper() for w in data["vocab"]]
    except Exception:
        return "", []
