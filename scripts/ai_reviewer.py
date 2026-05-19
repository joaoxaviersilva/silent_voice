import os
import sys
import subprocess
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Erro: GEMINI_API_KEY não encontrada.")
    sys.exit(1)

client = genai.Client(api_key=api_key)

def obter_git_diff():
    """Captura o diff do commit atual ou do Pull Request."""
    try:
       
        base_ref = os.getenv("GITHUB_BASE_REF")
        
        if base_ref:
            comando = ["git", "diff", f"origin/{base_ref}...HEAD", "--", "*.py"]
        else:
            comando = ["git", "diff", "HEAD~1", "HEAD", "--", "*.py"]
        
        resultado = subprocess.run(comando, capture_output=True, text=True, check=True)
        return resultado.stdout
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar git diff: {e.stderr}")
        return None

diff_conteudo = obter_git_diff()

if not diff_conteudo or not diff_conteudo.strip():
    print("Nenhuma alteração em arquivos Python encontrada no git diff.")
    sys.exit(0)

prompt = f"""Você é um engenheiro DevOps e revisor de código sênior especialista em Python.
Analise o `git diff` abaixo das alterações feitas no repositório. Sua resposta deve ser estritamente focada nos seguintes pontos:

1. **O que mudou:** Explique brevemente o resumo das alterações.
2. **Erros e Bugs:** Identifique possíveis quebras de lógica, exceções não tratadas ou bugs sintáticos.
3. **Segurança:** Procure por vulnerabilidades (chaves expostas, injeções, falhas de pacotes).
4. **Clean Code & Otimização:** Sugira melhorias de legibilidade, padrões Pythonicos e performance.

Seja extremamente direto, pragmático e vá direto ao ponto. Use markdown para formatar.

```diff
{diff_conteudo}
```"""

print("Enviando git diff para análise da IA...\n")

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    print("=== RELATÓRIO DE CODE REVIEW (GEMINI) ===")
    print(response.text)
    print("=========================================")
except Exception as e:
    print(f"Erro ao gerar conteúdo com a API: {e}")
    sys.exit(1)