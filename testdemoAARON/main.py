from dotenv import load_dotenv
import os
import openai
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb as db
from chromadb import Client
from chromadb.config import Settings
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import logging
import sqlite3
#from transformers import T5ForConditionalGeneration, T5Tokenizer
from rerankers import Reranker
#from py2neo import Graph
from langchain.chains.base import Chain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Import necessary libraries for anonymization, spellchecking, and niceness
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from spellchecker import SpellChecker
from textblob import TextBlob

class AnswerOnlyOutputParser(StrOutputParser):
    def parse(self, response):
        if "you do not have access" in response.lower():
            return "You do not have access"
        return response.split("Answer:")[1].strip() if "Answer:" in response else response.strip()

class ChatBot():
    def __init__(self, llm_type="Local (PHI3)", api_key="", memory=None):
        load_dotenv()
        self.chroma_client, self.collection = self.initialize_chromadb()
        self.llm_type = llm_type
        self.api_key = api_key
        self.setup_language_model()
        # Use the passed memory or create a new one
        self.memory = memory if memory else ConversationBufferMemory(memory_key="chat_history")
        self.setup_langchain()
        self.setup_context_identification_template()
        self.setup_reranker()
        #self.initialize_knowledge_graph()
        # Uncomment this line if `initialize_tools` is necessary
        # self.initialize_tools()

    def setup_reranker(self):
        self.reranker = Reranker("t5")

    def rerank_documents(self, question, documents):
        # Get the context from the collection
        for document in documents["documents"]:
            context = document
        access_levels = []
        for document in range(len(documents['metadatas'][0])):
            access_levels.append(documents['metadatas'][0][document]['access_role'])
        # Rerank the documents
        reranked_documents = self.reranker.rank(question, context, access_levels)
        return reranked_documents    
        
    def initialize_chromadb(self):
        # Initialize ChromaDB client using environment variable for path
        db_path = "testdemoAARON/chroma.db"
        client = db.PersistentClient(path=db_path)
        collection = client.get_collection(name="Company_Documents")
        return client, collection

    def setup_language_model(self):
        if self.llm_type == "External (OpenAI)" and self.api_key:
            try:
                openai.api_key = self.api_key
                self.llm = self.create_openai_chat_model()
            except Exception as e:
                raise ValueError(f"Failed to initialize the external LLM: {e}")
        else:
            # Setup for Local (PHI3) model
            try:
                self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
                self.llm = HuggingFaceHub(
                    repo_id=self.repo_id,
                    model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
                    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize the local LLM: {e}")

    def create_openai_chat_model(self):
        return lambda prompt: openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  #model may need updating
            messages=[{"role": "system", "content": "You are an informational chatbot. Use the following piece of context to answer the question. If you don't know the answer, simply state 'You do not have the required level of access.'"}, {"role": "user", "content": prompt}],
            temperature=0.8
        )
            
    #def initialize_knowledge_graph(self):
        #neo4j_url = os.getenv('NEO4J_URL')
        #neo4j_user = os.getenv('NEO4J_USER')
        #neo4j_password = os.getenv('NEO4J_PASSWORD')
        #self.graph = Graph(neo4j_url, auth=(neo4j_user, neo4j_password))

    #def query_knowledge_graph(self, query):
        #return self.graph.run(query).data()
    
    # def get_context_from_collection(self, input, access_levels):
    #     # Query all context first
    #     all_documents = self.collection.query(query_texts=[input], n_results=100)

    #     if not all_documents or 'documents' or not all_documents.get('documents'):
    #         return "No context found for the given input."

    #     all_documents = all_documents['documents']

    #     print(f'All documents: {all_documents}')

    #     # access_level check 
    #     if len(access_levels) == 1:
    #         where_clause = {"access_role": access_levels[0]}
    #     else:
    #         where_clause = {"$or": [{"access_role": level} for level in access_levels]}

    #     print(f'Where clause: {where_clause}')

    #     documents = self.collection.query(
    #         query_texts=[input], 
    #         n_results=100, 
    #         where=where_clause
    #     )   

    #     if not documents or 'documents' or not documents.get('documents'):
    #         return "No context available for your access level."
        
    #     documents = documents['documents']

    #     print(f"Filtered documents: {documents}")

    #     # Rerank the filtered documents
    #     reranked_documents = self.rerank_documents(input, documents)

    #     # Use top 3 reranked documents
    #     context = " ".join([doc["text"] for doc in reranked_documents[:3]])  # Append the top 3 docs together
    #     # context = reranked_documents[0]["text"]  # Pick the best document from the top 3

    #     return context
            
    def get_context_from_collection(self, input):
        # Extract context from the collection
        documents = self.collection.query(query_texts=[input],
                                        n_results=10
                                        )
        reranked_documents = self.rerank_documents(input, documents)
        # Use top 3 reranked documents
        context = ([doc.text for doc in reranked_documents.top_k(3)])
        document_roles = ([doc.doc_id for doc in reranked_documents.top_k(3)])
        # Store the conversation in memory
        self.memory.save_context({"input": input}, {"output": context})
        # context = reranked_documents.top_k(3)[0].text # This code is to pick the best document from the top 3
        return context, document_roles
            
    #def get_context_from_collection(self, input, access_levels):
        # Extract context from the collection
        #if len(access_levels) == 1:
            #documents = self.collection.query(query_texts=[input],
                                          #n_results=10,
                                          #where={"access_role": "General Access"}
                                          #where=access_levels[0]
                                          #)
        # if access_role == "General":
       #      documents = self.collection.query(query_texts=[input],
       #                                   n_results=5,
       #                                   where={"access_role": access_role+" Access"}
       #                                   )
       # elif access_role == "Executive":
       #     access_text = [{"access_role": "General Access"}, {"access_role": "Executive Access"}]
       #     documents = self.collection.query(query_texts=[input],
       #                                   n_results=10,
       #                                   where={"$or": access_text}
       #                                   )
        #else:
            #documents = self.collection.query(query_texts=[input],
                                              #n_results=10,
                                              #where={"$or": access_levels}
                                              #)
        #reranked_documents = self.rerank_documents(input, documents)
        # Use top 3 reranked documents
        #context = " ".join([doc.text for doc in reranked_documents.top_k(3)])  # This code is append the top 3 docs together
        # Store the conversation in memory
        #self.memory.save_context({"input": input}, {"output": context})
        # context = reranked_documents.top_k(3)[0].text # This code is to pick the best document from the top 3
        #return context

    #def get_context_from_knowledge_graph(self, input):
        # query for everything
        #query = "MATCH (n)-[r]->(m) RETURN n, r, m"
        #query = f"MATCH (n) WHERE n.name CONTAINS '{input}' RETURN n"
        #results = self.query_knowledge_graph(query)
        #results = ["", ""]
        #context = " ".join([str(result) for result in results])
        #return context
        #for document in documents["documents"]:
           #context = document
        #reranked_documents = self.rerank_documents(input, documents["documents"])
        #context = " ".join([doc["text"] for doc in reranked_documents[:5]])  # Use top 5 reranked documents
        #context = reranked_documents  # Use top 5 reranked documents
        #return context 


    # Uncomment this method if it's necessary
    # def initialize_tools(self):
    #     # Initialize tools for anonymization, spellchecking, and ensuring niceness
    #     self.analyzer = AnalyzerEngine()
    #     self.anonymizer = AnonymizerEngine()
    #     self.spellchecker = SpellChecker()

    def preprocess_input(self, input_dict):
        # Anonymize, spellcheck, and ensure niceness
        # Extract context and question from input_dict
        context = input_dict.get("context", "")
        question = input_dict.get("question", "")

        # Concatenate context and question
        combined_text = f"{context} {question}"
        return combined_text
            
    def setup_context_identification_template(self):
        template = """
        You are an informational chatbot. Employees will ask you questions about company data and meeting information.
        Your task is to determine where the relevant information to answer the question is found.

        - If the relevant information is found in the Filtered Contexts, respond with 'Filtered Context'.
        - If the relevant information is found in the Restricted Contexts, respond with 'Restricted Context'.
        - If no relevant information is found in either, respond with 'No Relevant Context Found'.

        Filtered Contexts: {filtered_contexts}

        Restricted Contexts: {restricted_contexts}

        Question: {question}

        Answer:
        """

        self.context_identification_prompt = PromptTemplate(
            template=template, 
            input_variables=["filtered_contexts", "restricted_contexts", "question"]
        )
        
        self.context_identification_chain = (
            {"filtered_contexts": RunnablePassthrough(), "restricted_contexts": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.context_identification_prompt
            | self.llm
            | AnswerOnlyOutputParser()
        )
            
    def setup_langchain(self):
        template = """
        You are an informational chatbot. Employees will ask you questions about company data and meeting information.
        Use the following instructions to provide the appropriate response:

        - You should prioritize the information provided in the Filtered Contexts to answer the question.
        - If relevant information is found in the Restricted Contexts, do not use it in your answer. Instead, respond with 'You do not have the required level of access.'
        - If you cannot find the information needed in the Filtered Contexts and no relevant information is in the Restricted Contexts, respond with 'I do not have the required information to answer the question.'

        Filtered Contexts: {filtered_contexts}

        Restricted Contexts: {restricted_contexts}

        Question: {question}

        Answer:
        """

        # Create a PromptTemplate
        self.prompt = PromptTemplate(template=template, input_variables=["filtered_contexts", "restricted_contexts", "question"])
        self.rag_chain = (
            {"filtered_contexts": RunnablePassthrough(), "restricted_contexts": RunnablePassthrough(), "question": RunnablePassthrough()}  # Using passthroughs for context and question
            | self.prompt
            | self.llm
            | AnswerOnlyOutputParser()
        )
            
    #def setup_langchain(self):
        #template = """
        #You are an informational chatbot. These employees will ask you questions about company data and meeting information. Use the following piece of context to answer the question.
        #If you don't know the answer, simply state "You do not have the required level of access".
        # You answer with short and concise answers, no longer than 2 sentences.

        #Context: {context}
        #Question: {question}
        #Answer:
        #"""

        #self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        #self.rag_chain = (
        #    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}  # Using passthroughs for context and question
        #    | self.prompt
        #    | self.llm
        #    | AnswerOnlyOutputParser()
        #)

    def filter_context(self, document_roles, user_role, context):
        filtered_contexts = []
        restricted_contexts = []
        for i in range(len(document_roles)):
            print(f"Document Role: {document_roles[i]}, User Role: {user_role}")
            if document_roles[i] in user_role:
                filtered_contexts.append(context[i])
            else:
                restricted_contexts.append(context[i])
            print(f"Filtered Contexts: {filtered_contexts}")
            print(f"Restricted Contexts: {restricted_contexts}")
        return filtered_contexts, restricted_contexts

    def ask(self, question, access_levels):
        # Get context and document roles from the collection
        context, document_roles = self.get_context_from_collection(question)
        filtered_contexts, restricted_contexts = self.filter_context(document_roles, access_levels, context)
        # join the context together
        restricted_contexts = " ".join(restricted_contexts)
        filtered_contexts = " ".join(filtered_contexts)
        
        context_source = self.context_identification_chain.invoke({"filtered_contexts": filtered_contexts, "restricted_contexts": restricted_contexts, "question": question})

        # Step 2: Based on the source, generate the response
        if "filtered context" in context_source.lower():
            # Proceed with the current setup using filtered context
            response = self.rag_chain.invoke({"filtered_contexts": filtered_contexts, "restricted_contexts": "", "question": question})
        elif "restricted context" in context_source.lower():
            # Proceed with the current setup using restricted context
            response = "You do not have the required level of access."
        else:
            response = "I do not have the required information to answer the question."

        # Save the conversation to memory
        self.memory.save_context({"input": question}, {"output": response})
        
        return response

    #def ask(self, input_dict):
        #context = self.get_context_from_collection(input_dict["question"], input_dict.get("access_levels", []))
        #input_dict["context"] = context

        # Load the previous chat history from memory
        # Load and append chat history to the context
        #chat_history = self.memory.chat_memory
        #if chat_history and chat_history.messages:
            #historical_context = " ".join([msg.content for msg in chat_history.messages])
            #context = f"{historical_context} {context}"  # Append historical context to current context
        #chat_history = self.memory.chat_memory
        #if chat_history and chat_history.messages:
            #input_dict["context"] += " " + " ".join([msg.content for msg in chat_history.messages])

        # Preprocess the input
        #processed_input = self.preprocess_input(input_dict)

        # Run the RAG chain
        #response = self.rag_chain.invoke(input_dict)
        
        # Save the conversation to memory
        #self.memory.save_context({"input": input_dict["question"]}, {"output": response})

        #return response
