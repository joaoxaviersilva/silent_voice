from utils.groq_client import client

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Responda apenas: API funcionando"}],
    temperature=0,
)

print(response.choices[0].message.content)
