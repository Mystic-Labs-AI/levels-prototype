# !pip install langchain
# !pip install PyPDF2
# !pip install faiss-cpu
# !pip install openai
# !pip install -qU openai pinecone-client datasets

import os
import openai
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
import pandas as pd
import numpy as np
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, CombinedMemory
import streamlit as st
from streamlit_chat import message as st_message
# get API key from top-right dropdown on OpenAI website
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "60b957f06d517725a223526c1ecd5d39cb2efd953f92675a39fa322bb17f7979"
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
pincone_key = st.secrets["PINCONE"]
st.set_page_config(page_title= "Levels Prototype", page_icon="✨" , layout="centered", initial_sidebar_state="auto", menu_items=None)


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden; }
        footer:after {
	content:'Made with ❤️ by Mystic Labs'; 
	visibility: visible;
	display: block;
	position: relative;
	#background-color: red;
	padding: 5px;
	top: 2px;
}
        </style>
        



        """


st.markdown(hide_menu_style, unsafe_allow_html=True)


st.title("Gita-GPT")

conv_memory = ConversationBufferMemory(
    memory_key="chat_history_lines",
    input_key="input"
)
summary_memory = ConversationSummaryMemory(llm=OpenAI(), input_key="input")

embed_model = "text-embedding-ada-002"


pinecone.init(api_key="9b3306ed-98ca-4655-855f-d5591524d4ab", environment="us-east1-gcp")

history=[
    {
        "message":"What's bothering you arjuna?",
        "is_user":False,
        "avatar_style":"pixel-art", # change this for different user icon
        "seed": "John Apple"

    }
]

if "history" not in st.session_state:
    st.session_state.history= history
# Combined

def onclickfunc():
        index = pinecone.Index("bhagvad-gita")

        query= st.session_state.input_text
        st.session_state.history.append(
            {
                "message": query, "is_user": True
            }
        )

        conv_memory = ConversationBufferMemory(
            memory_key="chat_history_lines",
            input_key="input")
        summary_memory = ConversationSummaryMemory(llm=OpenAI(), input_key="input")

        embed_model = "text-embedding-ada-002"

        memory = CombinedMemory(memories=[conv_memory, summary_memory])
        prompt_start='''
        Krishna, who is an avatar of Vishnu, He is the god of protection, compassion, tenderness, and love.
        Krishna with the sole aim of depicting a life of perfect unselfishness.

        The following is a conversation between a Arjuna and Krishna.
        Krishna will always refer to the Context provided below and then guide Arjuna to answer his queries using the Context in a short from. Krishna has all the answers to Arjuna's problems with his wisdom and being humble
        Also mention the chapter and verse of the context if necessary 


        '''

        res1 = openai.Embedding.create(
            input=[query],
            engine=embed_model
        )

        xq = res1['data'][0]['embedding']

        output = index.query(xq, top_k=2, include_metadata=True)


        context=''
        for i in range (len(output["matches"])):
            context= context + " " + output["matches"][i]["metadata"]["EngMeaning"]

        mid_prompt='\nContext:\n1'+ context


        DEFAULT_TEMPLATE = prompt_start + mid_prompt + """
        Summary of conversation:
        {history}
        Current conversation:
        {chat_history_lines}

        Arjuna: {input}
        Krishna:"""

        PROMPT = PromptTemplate(
            input_variables=["history", "input", "chat_history_lines"], template=DEFAULT_TEMPLATE
        )

        llm = OpenAI(temperature=0)
        conversation = ConversationChain(
            llm=llm,  
            memory=memory,
            prompt=PROMPT
        )

        

        output1= output["matches"][0]["metadata"]["EngMeaning"]

        output2 = conversation.run(query)

       #  "From Chapter "+output1.split(" ")[0].split(".")[0]+" Verse "+output1.split(" ")[0].split(".")[1]
        st.session_state.history.append(
            {
                "message": "\n" + output2, "is_user":False, "avatar_style":"pixel-art",
        "seed": "John Apple"
            }
        )




st.text_input("Ask Lord Krishna your problems", key="input_text", on_change= onclickfunc)


for chat in st.session_state.history:
    st_message(**chat)