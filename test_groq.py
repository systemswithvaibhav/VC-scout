from groq import Groq

client = Groq(api_key="")

response = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Explain startup investing in 2 lines"}
    ],
    model="llama-3.1-8b-instant"
)

print(response.choices[0].message.content)