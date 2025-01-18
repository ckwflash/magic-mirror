from groq import Groq
import config

client = Groq(
    api_key= config.API_KEY
)

def generate(emotion, age):
    llm = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are to give quotes based on the age and emotion of the user and insult him. You are to give the quote only without quotation marks and not explicitly mention the emotion and age"
            },
            {
                "role": "user",
                "content": f"My age is {age} and my emotion is {emotion}. Give me a quote"
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return llm.choices[0].message.content # send back quote