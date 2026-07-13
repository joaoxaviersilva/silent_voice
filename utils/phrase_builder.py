import json

from utils.groq_client import client

from config import LLM_MODEL


def build_phrase(context, words):
    prompt = f"""
    Contexto: "{context}"
    Palavras: {words}
    Gere frase natural.
    JSON: {{"phrase":""}}
    """

    r = client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2
    )

    try:
        return json.loads(r.choices[0].message.content)["phrase"]
    except Exception:
        return " ".join(words)
