import json

from config import LLM_MODEL
from utils.groq_client import client


def extract_json(content):
    content = content.strip()

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("A resposta não contém um JSON válido.")

    return json.loads(content[start : end + 1])


def rerank(context, candidates):
    if not candidates:
        return None

    normalized_candidates = [
        str(candidate).strip().upper()
        for candidate in candidates
        if str(candidate).strip()
    ]

    normalized_candidates = list(dict.fromkeys(normalized_candidates))

    if not normalized_candidates:
        return None

    prompt = f"""
Use o contexto da conversa para escolher a palavra mais adequada.

Contexto:
{context}

Candidatos:
{json.dumps(normalized_candidates, ensure_ascii=False)}

Regras:
- Escolha somente uma palavra presente na lista de candidatos.
- Não crie palavras novas.
- Retorne somente um JSON válido.
- Use exatamente este formato: {{"best": "PALAVRA"}}
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um reranqueador. Escolha apenas uma "
                        "das palavras candidatas fornecidas."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        result = extract_json(content)

        best = str(result.get("best", "")).strip().upper()

        if best in normalized_candidates:
            return best

    except Exception as error:
        print(f"Aviso: falha no reranking pelo LLM: {error}")

    return normalized_candidates[0]