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


def normalize_vocab(vocab):
    if not isinstance(vocab, list):
        return []

    normalized_vocab = []

    for word in vocab:
        normalized_word = str(word).strip().upper()

        if not normalized_word:
            continue

        if normalized_word in normalized_vocab:
            continue

        normalized_vocab.append(normalized_word)

        if len(normalized_vocab) == 15:
            break

    return normalized_vocab


def get_theme_and_vocab(text):
    text = str(text).strip()

    if not text:
        return "", []

    prompt = f"""
Analise o contexto transcrito e gere um tema e possíveis palavras
que poderiam ser respondidas ou pronunciadas nesse contexto.

Contexto:
{text}

Regras:
- Gere no máximo 15 palavras ou expressões curtas.
- Use palavras em português.
- Não repita palavras.
- Retorne somente um JSON válido.
- Use exatamente o formato:
{{"theme": "TEMA", "vocab": ["PALAVRA 1", "PALAVRA 2"]}}
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você gera vocabulários contextuais em formato JSON."
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

        theme = str(data.get("theme", "")).strip()
        vocab = normalize_vocab(data.get("vocab", []))

        return theme, vocab

    except Exception as error:
        print(f"Aviso: falha ao gerar contexto e vocabulário: {error}")
        return "", []