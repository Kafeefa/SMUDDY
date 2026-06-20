
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer using Deepseek with provided context.
    """
    prompt = f"""You are a helpful AI assistant. Use the provided context to answer the question accurately.

Context:
{context}

Question:
{question}

Provide a clear and accurate answer based on the context above."""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024
    )
    
    return response.choices[0].message.content