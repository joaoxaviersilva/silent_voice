import json

from utils.groq_client import client

from config import LLM_MODEL


def rerank(context, candidates):
    prompt = f"""
    Contexto: {context}
    Escolha melhor:
    {candidates}
    JSON: {{"best":""}}
    """

    r = client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.1
    )

    try:
        return json.loads(r.choices[0].message.content)["best"].upper()
    except Exception:
        return candidates[0]
