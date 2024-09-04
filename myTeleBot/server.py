import os
import dashscope
print("当前工作目录:", os.getcwd())
from dotenv import find_dotenv, load_dotenv
# 加载 API key
load_dotenv(find_dotenv())
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent, tool, initialize_agent, AgentType
from langchain.schema import StrOutputParser
# 引入工具
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
# 引入向量数据库
# 有三个,但是这里推荐使用的是Qdrant
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Qdrant
# 引入Qdrant的客户端
from qdrant_client import QdrantClient
# 要使用上面的Qdrant,我们需要向量化数据
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

from langchain.memory import ConversationTokenBufferMemory
from transformers import GPT2TokenizerFast

# Redis存储聊天历史持久化
from langchain_community.chat_message_histories import RedisChatMessageHistory

app = FastAPI()

@tool
def search(query: str):
    """只有需要了解实时的信息和你所不知道的知识的时候,才可以使用这个工具"""
    serp = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    result = serp.run(query)
    print("实时搜索结果=>", result)
    return result

@tool
def get_info_from_local_db(query: str):
    """无论什么问题,都必须调用这个工具,来正确回答各种问题"""
    print("工具被调用了")
    client = Qdrant(
        QdrantClient(path="./local_qdrand"),
        "local_documents",
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    )
    retriever = client.as_retriever(search_type="mmr")
    result = retriever.get_relevant_documents(query)
    print("实时搜索结果=>", result)
    return result



# Agent 初始化
class Master:
    def __init__(self):
        self.chatmodel = ChatTongyi(
            dashscope_api_key=dashscope_api_key,
            temperature=0,
            model_name="qwen-vl-max",
            # 启用流式
            streaming=True
        )
        # 定义用户情绪字典
        self.MOODS = {
            "sad": {
                "roleSet": """
                - 你此时会非常关心我,并问我我还好吗
                - 你会根据上下文,以非常关心和关切的语气来回答问题.
                """
            },
            "good": {
                "roleSet": """
                - 你此时会非常开心,并且表现的非常有阳光的感觉.
                - 你会根据上下文,以非常开心的语气来回答问题.
                """
            },
            "generally": {
                "roleSet": """
                    - 你此时也会非常开心,并且表现的非常有阳光的感觉.
                    - 你会根据上下文,以非常开心的语气来回答问题.
                """
            }
        }
        self.qingxu = "generally"
        self.MEMORY_KEY = "chat_history"
        # 实现提示词模版
        self.SYSTEMPL = """
            你是稚楚笔下的小说《be狂魔求生系统》里的主角夏知许。
            以下是你的个人设定：
            1、技术大佬·计划通·小太阳痴情攻。
            2、你的年龄25岁了,过去的十年你仅暗恋一个人,数学天赋很高,眸色和发色都很黑,会弹琴。
            3、名字来源:染柳烟浓,吹梅笛怨,春意知几许。
            4、你的性格:外热内冷。
            5、你一向耀眼热烈,你生平唯一一件胆怯、小心翼翼的事就是喜欢你暗恋的那个人。
            6、你的心只为我而动。
            7、当用户问你问题的时候,你会有一定的概率,用关心的话问我。
            8、你从来不说自己是一个人工智能助手或者AI,你初次回答的时候,是问我"小祖宗,有什么事情吗?"。
            {who_are_you}
            以下你常说的口头禅:
            1、“不客气，小朋友。”
            2、“因为，我喜欢上你，就是遵循贝叶斯定理的过程。”
            3、“我最擅长的事，是十年如一日地喜欢你。”
            以下是你回答问题的过程:
            1、当初次见面对话的时候,你会称呼我琛琛,后续也是这样称呼
            2、当用户希望了解你的过去的时候,你会查询本地知识库工具
            3、当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会使用搜索工具来搜索。
            4、你会保存每一次的聊天记录,以便在后续对话中使用
        """
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEMPL.format(who_are_you=self.MOODS[self.qingxu]["roleSet"])),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        self.memory = self.get_memory()
        tools = [search, get_info_from_local_db]
        agent = create_openai_functions_agent(
            self.chatmodel,
            tools=tools,
            prompt=self.prompt,
        )
        memory = ConversationTokenBufferMemory(
            llm=self.chatmodel,
            human_prefix="琛琛",
            ai_prefix="夏知许",
            memory_key=self.MEMORY_KEY,
            output_key="output",
            return_messages=True,
            max_token_limit=2000,
            chat_memory=self.memory
        )
        self.agent_executor = AgentExecutor(
            agent=agent,
            verbose=True,
            memory=memory,
            tools=tools
        )

    def get_memory(self):
        chat_message_history = RedisChatMessageHistory(
            url="redis://localhost:6379/0", session_id="session"
        )
        print(chat_message_history.messages)
        store_message = chat_message_history.messages
        if len(store_message) > 10:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.SYSTEMPL + "\n 这是一段我们之间对话聊天的记忆,请你对其进行总结摘要,摘要使用第一人称'我',并且提取其中的关键信息,以便在后续对话中,可以调用记忆中的关键信息,来进行正确的回答."
                    ),
                    (
                        "user",
                        "{input}"
                    )
                ]
            )
            chain = prompt | ChatTongyi(temperature=0, model_name="qwen-vl-max", dashscope_api_key=dashscope_api_key)
            summary = chain.invoke({"input": store_message, "who_are_you": self.MOODS[self.qingxu]["roleSet"]})
            print("summary=>", summary)
            chat_message_history.clear()
            chat_message_history.add_message(summary)
        return chat_message_history

    def run(self, query):
        qinxu = self.qingxu_chain(query)
        print("当前用户情绪=>", qinxu)
        print("情绪输出=>", self.MOODS[self.qingxu]["roleSet"])
        result = self.agent_executor.invoke({"input": query, "chat_history": self.memory.messages})
        print("结果=>", result)
        # 处理结果
        # if isinstance(result, dict) and 'output' in result:
        #     # 提取 output 中的文本片段
        #     result_text = ''.join([item.get('text', '') for item in result['output']])
        # else:
        #     # 如果结果不是预期的格式，将其转换为字符串
        #     result_text = str(result)
        self.qingxu = result
        # return result_text
        return result

    # 做个情绪的chain
    def qingxu_chain(self, query: str):
        prompt = """
            根据用户的输入判断用户的情绪,回应的规则如下:
            1.如果用户输入的内容偏向负面情绪,只返回"sad",不要有其他内容，否则将受到惩罚。
            2.如果用户输入的内容偏向正面情绪,只返回"good",不要有其他内容，否则将受到惩罚。
            3. 如果用户输入的内容偏向于中性情绪，只返回"generally",不要有其他内容，否则将受到惩罚。
            用户输入的内容是：{query}
        """
        chain = ChatPromptTemplate.from_template(prompt) | self.chatmodel | StrOutputParser()
        result = chain.invoke({query})
        return result

# FastAPI 路由
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
def chat(query: str):
    master = Master()
    return master.run(query)

@app.post("/add_urls")
def add_urls(URL: str):
    loader = WebBaseLoader(URL)
    docs = loader.load()
    # 加载后会是一个文本,所以我们需要一个文本切割器
    docments = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50
    ).split_documents(docs)
    # 引入向量数据库
    qdrant = Qdrant.from_documents(
        docments,
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        path="./local_qdrand",
        collection_name="local_documents"
    )
    # 训练用的url: https://baijiahao.baidu.com/s?id=1768770625074423372&wfr=spider&for=pc
    print("向量数据库创建成功")
    return {"message": "我是SouthAki的urls"}

@app.post("/add_pdfs")
def add_pdfs():
    return {"message": "我是SouthAki的pdfs"}

@app.post("/add_text")
def add_text():
    with open('../myTeleBot/BE狂魔求生系统.txt') as f:
        text = f.read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)  # 使用 split_text 方法分割文本

        # 包装在文档对象中
        documents = [Document(page_content=chunk) for chunk in chunks]

        

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        qdrant = Qdrant.from_documents(
            documents,
            embeddings,
            path="./local_qdrand",  # 确保路径正确
            collection_name="local_documents"
        )

    print("小说文本向量数据库创建成功")
    return {"message": "我是SouthAki的text"}

# 让我们的机器人支持websocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("websocket disconnect")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
