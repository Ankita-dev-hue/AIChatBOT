from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
import requests
import streamlit as st


# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


# Create an LLM Object for API Call
@st.cache_resource
def load_conversation():
    llm = ChatGroq(api_key=api_key, model_name="llama3-70b-8192")
    memory = ConversationBufferMemory()
    return ConversationChain(llm=llm, memory=memory, verbose=False)


conversation = load_conversation()

# Initialize our streamlit app
st.title("Hi, I'm Kiti, your AI Assistant!")
st.subheader("Powered by Groq's LLM and Streamlit")
st.markdown("How can I help you today?")
st.markdown("Ask me anything! I remember the whole conversation!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field
user_input = st.text_input("You:", key="input")

if user_input:
    response = conversation.predict(input=user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for speaker, text in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")
