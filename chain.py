import getpass
import os
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

class Chain():
    def __init__(self, retriever, session_id):
        self.store = {session_id: ChatMessageHistory()}
        self.retriever = retriever
        self.session_id = session_id
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.chat_history()
        self.create_chain()
        self.create_conversational_rag_chain()
        
    def get_session_history(self) -> BaseChatMessageHistory:
        return self.store[self.session_id]
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def chat_history(self):
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        
    def create_chain(self):
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        If you don't know the answer, just say that you don't know. \
            
        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        
        qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, qa_chain)
    
    def create_conversational_rag_chain(self):
        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
    def invoker(self, input, session_id):
        session_history = self.get_session_history()
        
        print(session_history.messages)
        
        chat_history = [
            msg for msg in session_history.messages
        ]
        
        output = self.conversational_rag_chain.invoke(
            {"input": input, "chat_history": chat_history},
            config={
                "configurable": {"session_id": session_id}
            }
        )
        
        session_history.add_user_message(input)
        session_history.add_ai_message(output['answer'])
        
        return output['answer']
        
        