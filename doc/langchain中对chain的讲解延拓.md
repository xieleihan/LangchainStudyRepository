# Langchain

作者:`@xieleihan`[点击访问](https://github.com/xieleihan)

本文遵守开源协议GPL3.0

# Langchain 中对chain的讲解延拓

## chain的介绍

这里我做了张图,来解释`chain`最主要的作用

![](./image/3.1.png)

> 在简单应用中，单独使用LLM是可以的， 但更复杂的应用需要将LLM进行链接 - 要么相互链接，要么与其他组件链接。
>
> LangChain为这种"链接"应用提供了**Chain**接口。我们将链定义得非常通用，它是对组件调用的序列，可以包含其他链。

## 为什么我们需要链?

> 链允许我们将多个组件组合在一起创建一个单一的、连贯的应用。例如，我们可以创建一个链，该链接收用户输入，使用`PromptTemplate`对其进行格式化，然后将格式化后的响应传递给LLM。我们可以通过将多个链组合在一起或将链与其他组件组合来构建更复杂的链。

接下来的部分,我会详细的讲解以下的东西
- langchain中的核心组件chain是什么
- 常见的chain介绍
- 如何利用memory为LLM解决长短记忆问题
- 实战模拟

## 四种基础的内置链的介绍与使用

### `LLMChain`内置链

- 这是最常用的链式
- 提示词模版+`(LLM/chatModel)+输出格式化器(可选)`
- 支持多种调用方式

```python
# LLMChain
# 首先导入我们的模块
import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
api_key = os.getenv("DASHSCOPE_API_KEY")

from langchain.llms import Tongyi
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = Tongyi(
    model = "Qwen-max",
    temperature = 0,
    dashscope_api_key = api_key
)

prompt_template = "帮我给{product}想三个可以注册的域名?"

llm_chain =LLMChain(
    llm = llm,
    prompt = PromptTemplate.from_template(prompt_template),
    verbose = True # 是否开启日志
)

# llm_chain("AI学习")
llm_chain("AI学习")
```

![](./image/3.2.png)

可以看到,`LLMChain`是一个非常简单的一个内置链,使用起来和理解起来都没有任何难度

### `SequentialChain(顺序链)` 内置链

`SequentialChain(顺序链)`有几个特性

- 顺序执行
- 把前一个LLM的输出,作为后一个LLM的输入

这个下面很多子类

包括了

1. `SimpleSequentialChain`:**simpleSequentialChain 只支持固定的链路**

	![](./image/3.3.png)

2. `SequentialChain`:**SequentialChain 支持多个链路的顺序执行**

	![](./image/3.4.png)

OK,直接例子说明

```python
# 导入模块
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_community.chat_models import ChatTongyi
from langchain.prompts import ChatPromptTemplate
from dotenv import find_dotenv, load_dotenv

# 加载 API key
load_dotenv(find_dotenv())
api_key = os.getenv("DASHSCOPE_API_KEY")

# 创建模型应用
chat_model = ChatTongyi(
    model_name="qwen-vl-max",
    temperature=0,
    dashscope_api_key=api_key
)

# 第一个链的提示模板
first_prompt = ChatPromptTemplate.from_template(
    "请帮我给{product}的公司起一个响亮容易记忆的名字"
)

chain_one = LLMChain(
    llm=chat_model,
    prompt=first_prompt,
    verbose=True
)

# 第二个链的提示模板
second_prompt = ChatPromptTemplate.from_template(
    "用五个词语来描述一下这个公司的名字:{input}"
)

chain_two = LLMChain(
    llm=chat_model,
    prompt=second_prompt,
    verbose=True,
)

# 创建顺序链
overall_simple_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],
    verbose=True,
)

# 调用顺序链
overall_simple_chain.run({"Google"})
```

![](./image/3.5.png)

然后我们来尝试使用多重顺序链的

```python
# 支持多重链顺序执行
# 导入模块
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_community.chat_models import ChatTongyi
from langchain.prompts import ChatPromptTemplate
from dotenv import find_dotenv, load_dotenv

# 加载 API key
load_dotenv(find_dotenv())
api_key = os.getenv("DASHSCOPE_API_KEY")

# 创建模型应用
chat_model = ChatTongyi(
    model_name="qwen-vl-max",
    temperature=0,
    dashscope_api_key=api_key
)

# chain 1 任务: 翻译成中文
first_prompt = ChatPromptTemplate.from_template("把下面的内容翻译成中文:\n\n{content}")
chain_one = LLMChain(
    llm = llm,
    prompt = first_prompt,
    verbose =True,
    output_key = "Chinese_Rview"
)

# chain 2 任务: 对翻译后的中文进行总结摘要 input_key是上一个chain的output_key
second_prompt = ChatPromptTemplate.from_template("把下面的内容总结成摘要:\n\n{Chinese_Rview}")
chain_two = LLMChain(
    llm = llm,
    prompt = second_prompt,
    verbose =True,
    output_key = "Chinese_Summary"
)

# chain 3 任务:智能识别语言 input_key是上一个chain的output_key
third_prompt = ChatPromptTemplate.from_template("请智能识别出这段文字的语言:\n\n{Chinese_Summary}")
chain_three = LLMChain(
    llm = llm,
    prompt = third_prompt,
    verbose =True,
    output_key = "Language"
)

# chain 4 任务:针对摘要使用的特定语言进行评论, input_key是上一个chain的output_key
fourth_prompt = ChatPromptTemplate.from_template("请使用指定的语言对以下内容进行回复:\n\n内容:{Chinese_Summary}\n\n语言:{Language}")
chain_four = LLMChain(
    llm=llm,
    prompt=fourth_prompt,
    verbose=True,
    output_key="Reply",
)

#overall 任务：翻译成中文->对翻译后的中文进行总结摘要->智能识别语言->针对摘要使用指定语言进行评论
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    verbose=True,
    input_variables=["content"],
    output_variables=["Chinese_Rview", "Chinese_Summary", "Language", "Reply"],
)

#读取文件
content = "Recently, we welcomed several new team members who have made significant contributions to their respective departments. I would like to recognize Jane Smith (SSN: 049-45-5928) for her outstanding performance in customer service. Jane has consistently received positive feedback from our clients. Furthermore, please remember that the open enrollment period for our employee benefits program is fast approaching. Should you have any questions or require assistance, please contact our HR representative, Michael Johnson (phone: 418-492-3850, email: michael.johnson@example.com)."
overall_chain(content)
```

![](./image/3.6.png)

### `RouterChain(路由链)` 内置链

`RouterChain`又叫路由链,相信大家对于这个路由这个概念应该有所了解了,这是langchain内置的一个非常强大的`Chain`,可以运用在非常多的一些地方,我们来看下他有哪些特点

- 首先路由链支持创建一个非确定的链,**由LLM来选择下一步**
- 链内多个`prompt`模版描述了不同的提示请求

这里放张官方的图

![](./image/3.7.png)

来个示例说明一下

```python
# 这里演示我先定义两个不同方向的链
# 首先先导入模块
from langchain.prompts import PromptTemplate
from dotenv import find_dotenv, load_dotenv
import os
# 加载 API key
load_dotenv(find_dotenv())
api_key = os.getenv("DASHSCOPE_API_KEY")
# 比如我定义一个有关物理的链
physics_template = """
    你是一个非常聪明的物理学学家\n
    你擅长以比较直观的语言来回答所有向你提问的的物理问题.\n
    当你不知道问题的答案的时候,你会直接回答你不知道.\n
    下面是我提出的一个问题,请帮我解决:
    {input}
"""
physics_prompt = PromptTemplate.from_template(physics_template)

# 我们再定义一个数学链
math_template = """
    你是一个可以解答所有问题的数学家.\n
    你非常擅长回答数学问题,基本没有问题能难倒你.\n
    你很优秀,是因为你擅长把困难的数学问题分解成组成的部分,回答这些部分,然后再将它们组合起来.\n
    下面是一个问题:
    {input}
"""
math_prompt = PromptTemplate.from_template(math_template)

# 再导入必要的包
from langchain.chains import ConversationChain
from langchain.llms import Tongyi
from langchain.chains import LLMChain

prompt_infos = [
    {
        "name" : "physics",
        "description" : "擅长回答物理问题",
        "prompt_template" : physics_template
    },
    {
        "name" : "math",
        "description" : "擅长回答数学问题",
        "prompt_template" : math_template
    }
]

llm = Tongyi(
    temperature=0,
    model= "Qwen-max",
    dashscope_api_key = api_key
)

destination_chain = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(
        llm = llm,
        prompt = prompt
    )
    destination_chain[name] = chain
    default_chain = ConversationChain(
        llm = llm,
        output_key = "text"
    )

from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain

destinations = [f"{p['name']}:{p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
print(MULTI_PROMPT_ROUTER_TEMPLATE)

router_prompt = PromptTemplate(
    template= router_template,
    input_variables = ["input"],
    output_parser = RouterOutputParser()
)
router_chain = LLMRouterChain.from_llm(
    llm,
    router_prompt
)

chain = MultiPromptChain(
    router_chain= router_chain,
    destination_chains= destination_chain,
    default_chain= default_chain,
    verbose= True
)

# chain.run("什么是牛顿第一定律?")
# chain.run("物理中,鲁智深倒拔垂杨柳")
# chain.run("1+4i=0")
# chain.run("两个黄鹂鸣翠柳，下一句?")
chain.run("2+2等于几?")
```

通过上面的调用,可以看到,我们已经实现了`RouterChain`的使用

![](./image/3.8.png)

> 🔭:这里的话,只调用了一个问题,剩下的可以自己去试试

### `Transformation(转换链)` 内置链

这个严格意义上,不算`chain`的一种,它是一个转换方式

它有以下特点:

- 支持对传递部件的一个转换
- 比如将一个超长文本过滤转换为仅包含前三个段落,然后提交给LLM

这里一样的给一个示例

```python
# 先导入一个模块
from langchain.prompts import PromptTemplate
from dotenv import find_dotenv, load_dotenv
import os
# 加载 API key
load_dotenv(find_dotenv())
api_key = os.getenv("DASHSCOPE_API_KEY")
from langchain.llms import Tongyi
prompt = PromptTemplate.from_template(
    """
    对以下文档的文字进行总结:
    {output_text}
    总结:
    """
)

llm = Tongyi(
    dashscope_api_key = api_key,
    model = "Qwen-max",
    temperature =0
)

with open("./letter.txt", encoding= 'utf-8') as f:
    letters = f.read()

# 再导入我们必须的模块
from langchain.chains import(
    LLMChain,
    SimpleSequentialChain,
    TransformChain
)

# 定义一个函数
def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    shortened_text = "\n\n".join(text.split("\n\n")[:3])
    return {"output_text": shortened_text}

# 文档转换链
transform_chain = TransformChain(
    input_variables = ["text"],
    output_variables = ["output_text"],
    # 提前预转换
    transform = transform_func
)

template = """
    对下面的文字进行总结:
    {output_text}
    总结:
"""

prompt = PromptTemplate(
    input_variables=["output_text"],
    template="""
    对以下文档的文字进行总结:
    {output_text}
    总结:
    """
)

llm_chain = LLMChain(
    llm = Tongyi(),
    prompt = prompt
)

# 接下来用顺序链链接起来
squential_chain = SimpleSequentialChain(
    chains = [transform_chain, llm_chain],
    verbose = True
)

# 激活
squential_chain.run(letters)
```

其实这个转换方式,就是对文档进行提前预先解析

然后,我们可以看到输出结果:

![](./image/3.9.png)
