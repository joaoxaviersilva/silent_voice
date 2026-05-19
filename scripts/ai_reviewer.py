import os
import sys
import subprocess
import time
import requests  
from google import genai

MODELO_GEMINI = "gemini-2.5-flash"
MAX_TENTATIVAS = 3

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
        print(f"Erro ao executar git diff: {e.stderr}", file=sys.stderr)
        return None

def realizar_review(client, prompt):
    """Envia o prompt para a API do Gemini com sistema de retry."""
    for tentativa in range(MAX_TENTATIVAS):
        try:
            response = client.models.generate_content(
                model=MODELO_GEMINI,
                contents=prompt
            )
            return response.text
        except Exception as e:
            if "503" in str(e) and tentativa < MAX_TENTATIVAS - 1:
                print(f"Servidor instável (503). Tentando novamente em 5 segundos... (Tentativa {tentativa + 2}/{MAX_TENTATIVAS})")
                time.sleep(5)
            else:
                print(f"Erro crítico na API: {e}", file=sys.stderr)
                sys.exit(1)

def postar_comentario_no_pr(relatorio):
    """Posta o relatório gerado como um comentário no Pull Request."""
    github_token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    event_path = os.getenv("GITHUB_EVENT_PATH")
    
    if not github_token or not repo or not event_path:
        print("Ambiente fora de um Pull Request ativo. Pulando postagem de comentário.")
        return

    import json
    with open(event_path, 'r') as f:
        event_data = json.load(f)
    

    pr_number = event_data.get("pull_request", {}).get("number")
    if not pr_number:
        print("Número do PR não encontrado. O comentário não será postado.")
        return

    print(f"Postando comentário no PR #{pr_number}...")
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    body = {
        "body": f"### 🤖 Gemini Code Review\n\n{relatorio}"
    }
    
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 201:
        print("Comentário postado com sucesso no Pull Request!")
    else:
        print(f"Falha ao postar comentário: {response.status_code} - {response.text}", file=sys.stderr)

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Erro: GEMINI_API_KEY não encontrada.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
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
    relatorio = realizar_review(client, prompt)
    
    print("=== RELATÓRIO DE CODE REVIEW (GEMINI) ===")
    print(relatorio)
    print("=========================================")

    postar_comentario_no_pr(relatorio)

if __name__ == "__main__":
    main()