import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(question, context_chunks, language="English"):
    context_text = ""
    for i, chunk in enumerate(context_chunks):
        context_text += (
            f"\n[Source {i+1}: {chunk['source']}, Page {chunk['page']}]\n"
            f"{chunk['text']}\n"
        )

    lang_instruction = f"IMPORTANT: You MUST respond entirely in {language}. All explanations, answers, and text must be in {language} only."

    # ✅ If no useful context → allow general AI answer
    if len(context_text.strip()) < 50:
        return f"""{lang_instruction}

Answer this question clearly and concisely:

QUESTION: {question}

ANSWER (in {language}):"""

    # ✅ If context exists → use PDF + fallback to general knowledge
    return f"""{lang_instruction}

Answer using the context below.
If the answer is not present in the context, use your general knowledge.

After answering, list sources as: Sources: [filename, Page X]

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER (in {language}):"""


def answer_question(question, context_chunks, language="English"):
    prompt = build_prompt(question, context_chunks, language=language)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    raw_answer = response.choices[0].message.content

    # Extract unique sources
    seen = set()
    sources = []
    for chunk in context_chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"file": chunk["source"], "page": chunk["page"]})

    return {"answer": raw_answer, "sources": sources}