import os
import re
import json
import asyncio
import uvicorn
import time
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool, BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from pydantic import BaseModel, Field
from typing import Type
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import constants, models

app = FastAPI(title="FuzzAgent-API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_vector_store():
    persistent_directory = "vector_db_openai_embeddings"
    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(persistent_directory):
        print("Loading existing vector store...")
        db = Chroma(persist_directory=persistent_directory, embedding_function=openai_embeddings)
        print(f"OpenAIEmbeddings Vectorstore loaded with {db._collection.count()} documents")
        return db
    else:
        print(f"Vector store {persistent_directory} does not exist.")
        return None

def parse_and_format_output(input, **kwargs):
    json_parser = JsonOutputParser()
    print(f"Raw input to the formatting function: {input}")
    parsed_json = json_parser.parse(input)

    def format_request_templates(parsed_json):
        formatted_string = "For The RTSP protocol, the client request templates are: \n"
        for i, (method, template_list) in enumerate(parsed_json.items(), start=1):
            formatted_template = f'"{method}": [\n' + ',\n'.join([f'"{line}"' for line in template_list]) + '\n]'
            formatted_string += f"{i}. {method}:\n{formatted_template}\n\n"
        return formatted_string

    return format_request_templates(parsed_json)

def initiate_rag_retriever(search_type, search_kwargs):
    vectorstore = get_vector_store()
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
def initiate_llm_and_rag_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    
    retriever = initiate_rag_retriever("mmr", {"k": 10, "fetch_k": 20, "lambda_mult": 0.5, "filter": {"doc_type": "rfc-2326-rtsp"}})
    print(f"RAG Retriever loaded successfully: {retriever}")

    # Create a prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", constants.contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a history-aware retriever
    # This uses the LLM to help reformulate the question based on chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", constants.qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a chain to combine documents for question answering
    # `create_stuff_documents_chain` feeds all retrieved context into the LLM
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create a retrieval chain that combines the history-aware retriever and the question answering chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return llm, rag_chain    

def initiate_fuzz_agent():
    llm, rag_chain = initiate_llm_and_rag_chain()
    print(f"LLM and RAG chain loaded successfully")

    
    # Set Up ReAct Agent with Document Store Retriever
    # Load the ReAct Docstore Prompt
    react_docstore_prompt = hub.pull("hwchase17/react")

    tools = [
        Tool(
            name="Answer Question",
            func=lambda input, **kwargs: rag_chain.invoke(
                {"input": input, "chat_history": kwargs.get("chat_history", [])}
            ),
            description="useful for when you need to answer questions about the context",
        ),
        Tool(
            name="Format Response",
            func=parse_and_format_output,
            description="useful for when you need to parse json, formatting it into a string"
        )
        # SimpleSearchTool()
    ]

    # Create the ReAct Agent with document store retriever
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_docstore_prompt,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, handle_parsing_errors=True, verbose=True,
    )
    
    return agent_executor

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat-llm", response_model=models.ResponseModel)
async def chat_with_llm(request: models.RequestModel):
    print("Chatting with FuzzAgent...")
    
    prompt = request.messages[-1].content
    print(f"Prompt in the coming request: {prompt}")
    
    try:
        response = await asyncio.wait_for(
            fuzzagent.ainvoke({"input": prompt, "chat_history": chat_history}),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        pass
        # raise HTTPException(status_code=504, detail="Request timeout")
        
    llm_response = response["output"]
    prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
    completion_tokens = len(llm_response.split())

    print("*"*30)
    print(f"FuzzAgent Response: {llm_response}")
    print("*"*30)
    
    chat_history.append(HumanMessage(content=prompt))
    chat_history.append(AIMessage(content=llm_response))
    
    return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": llm_response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
    }


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    
    global fuzzagent
    global chat_history
    chat_history = []
    fuzzagent = initiate_fuzz_agent()
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)