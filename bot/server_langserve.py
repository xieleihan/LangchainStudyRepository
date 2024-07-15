#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatTongyi
from langserve import add_routes
import os

import os
from dotenv import find_dotenv, load_dotenv
# 加载 API key
load_dotenv(find_dotenv())
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatTongyi(),
    path="/openai",
)


model = ChatTongyi(
    dashscope_api_key = dashscope_api_key
)
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)