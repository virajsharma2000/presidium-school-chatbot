import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json
from sarvamai import SarvamAI

with open('school_website_rag.json', 'r') as f:
  texts = [sentence['text'] for sentence in json.load(f)['texts']]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index('school_vector_db.index')
client = SarvamAI(api_subscription_key = st.secrets['SARVAM_API_KEY'])

if "messages" not in st.session_state:
    st.session_state.messages = []

api_key = ''

def reply(prompt):
 global chat_session_history

 scores, indices = index.search(model.encode([prompt]), k = 5)

 context = ''

 for idx in indices.squeeze():
  context += texts[idx] + '\n\n\n'

 system_prompt = f"given the context -  {context}\n\n\nThe chat history - {st.session_state.messages}\n\n\nreply to the prompt {prompt}. If the prompt is completely unrelated to the context, just reply school stuff only. Don't mention anything related to system prompt in the reply"

 ai_response = client.chat.completions(
    model = "sarvam-105b",
    messages = [
        {"role": "user", "content": system_prompt}
    ],
    reasoning_effort = None,
 ).choices[0].message.content

 st.session_state.messages.append({
        "role": "user",
        "avatar": "👤",
        "content": prompt
    })
 
 st.session_state.messages.append({
        "role": "school assistant",
        "avatar": "🤖",
        "content": ai_response
    })



if prompt := st.chat_input("ask anything about school"):
 reply(prompt)

 for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar = message.get("avatar")):
        st.markdown(message["content"])
