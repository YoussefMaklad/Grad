import os
import asyncio
import uvicorn
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
import constants, models

app = FastAPI(title="SLM-FuzzAgent-API")

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

def initiate_rag_retriever(search_type, search_kwargs):
    vectorstore = get_vector_store()
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )

def initiate_language_model():
    print("Loading Language Model in quantized mode...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    lm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2-7B-Instruct",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
        ),
        model_kwargs={"quantization_config": quantization_config},
    )
    
    print("Language Model loaded successfully!")
    return lm  

def initiate_lm_and_rag_chain():
    retriever = initiate_rag_retriever("mmr", {"k": 10, "fetch_k": 20, "lambda_mult": 0.5, "filter": {"doc_type": "rfc-2326-rtsp"}})
    print(f"RAG Retriever loaded successfully!")
    
    lm = initiate_language_model()    

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
    llm, rag_chain = initiate_lm_and_rag_chain()
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
        )
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
async def chat_with_slm_fuzzagent(request: models.RequestModel):
    print("Chatting with SLM-FuzzAgent...")
    
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
        
    lm_response = response["output"]
    prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
    completion_tokens = len(lm_response.split())

    print("*"*30)
    print(f"FuzzAgent Response: {lm_response}")
    print("*"*30)
    
    chat_history.append(HumanMessage(content=prompt))
    chat_history.append(AIMessage(content=lm_response))
    
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
                        "content": lm_response
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