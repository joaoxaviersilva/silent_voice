import json

from config import LLM_MODEL
from utils.groq_client import client


def extract_json(content):
    content = content.strip()

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("A resposta não contém um JSON válido.")

    return json.loads(content[start : end + 1])


def build_phrase(context, words):
    normalized_words = [
        str(word).strip()
        for word in words
        if str(word).strip()
    ]

    normalized_words = list(dict.fromkeys(normalized_words))

    if not normalized_words:
        return ""

    context = str(context).strip()

    fallback_phrase = " ".join(normalized_words)

    prompt = f"""
Construa uma frase natural em português utilizando as palavras
confirmadas pelo sistema.

Contexto da conversa:
{context or "Nenhum contexto disponível."}

Palavras confirmadas:
{json.dumps(normalized_words, ensure_ascii=False)}

Regras:
- Use as palavras confirmadas como base principal da frase.
- Considere o contexto somente para organizar a frase naturalmente.
- Não invente nomes, fatos ou intenções que não estejam presentes.
- Retorne apenas um JSON válido.
- Use exatamente o formato:
{{"phrase": "FRASE NATURAL"}}
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você transforma palavras confirmadas em uma "
                        "frase curta e natural em português."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content or ""
        data = extract_json(content)

        phrase = str(data.get("phrase", "")).strip()

        if phrase:
            return phrase

    except Exception as error:
        print(f"Aviso: falha ao construir a frase: {error}")

    return fallback_phrase