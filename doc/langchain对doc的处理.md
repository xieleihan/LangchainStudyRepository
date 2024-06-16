# Langchain

作者:`@xieleihan`[点击访问](https://github.com/xieleihan)

本文遵守开源协议GPL3.0

# Langchain Doc文档的分析处理

## 介绍

### 首先是基础的langchain的知识

#### 模型IO:大语言的模型交互接口

```text
简要的介绍就是三个部分:
model I/O
|
v
- Prompts # 输入的文本信息
- Language models # 语言模型的处理
- Output parsers # 输出
```

![](./image/2.1.png)

> 思维导图来源于`@xieleihan`,请勿转载,如需转载,请征求授权

#### prompts的相应知识

`prompts`模版,让这个模型输出更加的高级和灵活的提示词工程,在AI领域是非常关键的

我总结下,就是有以下四点:

1. **立角色**:引导AI进入具体场景,赋予其行家的身份
2. **述问题**:告诉AI你的困惑和问题,以及背景信息
3. **定目标**:告诉AI你的需求是什么,希望达成什么目标
4. **补要求**:告诉AI回答时需要注意什么,或者如何回复

这里可以给模版

> 1. 将提示词提炼出模版
> 2. 实现提示词的复用,版本管理,动态变化

### 试下将这个放入测试中

#### 用prompts模版教会LLM输入出

```Python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("请帮我生成一个具有{country}特色的名字")
prompt.format(country="中国")
```

Output:`'请帮我生成一个具有中国特色的名字'`

OK,上面的结果出来的,但是,刚才说了prompt的模版,就是我们希望让AI知道自己的身份,它需要帮我们做什么这样的

所以根据上面的,我们需要对输入进去的问题进行修改

```Python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("你是一个{name},请你帮我生成一个具有{country}特色的{sex}的名字")
print(prompt.format(name="算命大师", country="中国", sex="男"))
```

Output:`你是一个算命大师,请你帮我生成一个具有中国特色的男的名字`

通过传递参数,让我们的语言在询问AI的时候,能得到更有用的答案

可以看到字符模版就是上面那种,然后处理方式有LLM,还有另一个就是chatmodels,就是对话模版

```Python
# 对话模版具有结构,chatmodels
# 首先导入模块
from langchain.prompts import ChatPromptTemplate

# 一个结构模版  system,human,ai,user_input
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个起名大师,你的名字叫{name}."),
    ("human", "你好{name},你感觉怎么样"),
    ("ai","你好!我现在状态非常好"),
    ("human","{user_input}"),
])

# 如果我们按照上面的试试
# prompt.format(name="mldys",user_input="你叫什么名字呢?")
# 出来的结果就不是我们想要的
# 所以这里要换一个chatmodels里的一个,chat_template.format_messages方法
chat_template.format_messages(name="mldys",user_input="你叫什么名字呢?")
```

![](./image/2.2.png)

就是除了上面的方式,其实`langchain`还有另一个信息模版的模块

就是对应的例如

`SystemMessage`, `HumanMessage`, `AIMessage`

```Python
# 还是第一步导入模块
from langchain.schema import SystemMessage
from langchain.schema import HumanMessage
from langchain.schema import AIMessage

# 直接创建消息
SystemMessage(
    content="你是一个起名大师",
    # additional_kwargs 附加参数
    additional_kwargs={
        "大师姓名": "女大学生"
        # 这里我在写的时候看到植物大战僵尸杂交版更新了,就随便起的
    }
)

# HumanMessage(
#     content= "请问大师叫什么"
# )

# AIMessage(
#     content="我是女大学生"
# )
```

Output:`SystemMessage(content='你是一个起名大师', additional_kwargs={'大师姓名': '女大学生'})`

```Python
from langchain.schema import SystemMessage
from langchain.schema import HumanMessage
from langchain.schema import AIMessage

# 直接创建消息
# SystemMessage(
#     content="你是一个起名大师",
#     # additional_kwargs 附加参数
#     additional_kwargs={
#         "大师姓名": "女大学生"
#     }
# )

HumanMessage(
    content= "请问大师叫什么"
)

# AIMessage(
#     content="我是女大学生"
# )
```

Output:`HumanMessage(content='请问大师叫什么')`

```Python
from langchain.schema import SystemMessage
from langchain.schema import HumanMessage
from langchain.schema import AIMessage

# 直接创建消息
# SystemMessage(
#     content="你是一个起名大师",
#     # additional_kwargs 附加参数
#     additional_kwargs={
#         "大师姓名": "女大学生"
#     }
# )

# HumanMessage(
#     content= "请问大师叫什么"
# )

AIMessage(
    content="我是女大学生"
)
```

Output:`AIMessage(content='我是女大学生')`

> 每段跑起来就是这样,我们可以这样在一个数组里展示出来

```Python
# 还是第一步导入模块
from langchain.schema import SystemMessage
from langchain.schema import HumanMessage
from langchain.schema import AIMessage

# 直接创建消息
system = SystemMessage(
    content="你是一个起名大师",
    # additional_kwargs 附加参数
    additional_kwargs={
        "大师姓名": "女大学生"
        # 这里我在写的时候看到植物大战僵尸杂交版更新了,就随便起的
    }
)

people = HumanMessage(
    content= "请问大师叫什么"
)

appleIntelligence = AIMessage(
    content="我是女大学生"
)

[system, people, appleIntelligence]
```

Output:`[SystemMessage(content='你是一个起名大师', additional_kwargs={'大师姓名': '女大学生'}), HumanMessage(content='请问大师叫什么'), AIMessage(content='我是女大学生')]`

效果消息体,跟上面是一样的

编程的时候适用不同场景,这里需要注意

然后,我们需要自定义自己的问答的时候,需要用上这个`ChatMessagePromptTemplate`

```Python
from langchain.prompts import AIMessagePromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import ChatMessagePromptTemplate

# 定义模板
prompt = "愿{subject}与你同在!"

# 创建 ChatMessagePromptTemplate 实例并提供 role 参数
chat_message_prompt = ChatMessagePromptTemplate.from_template(template=prompt, role="system")

# 格式化模板
formatted_prompt = chat_message_prompt.format(subject="上帝")

print(formatted_prompt)
```

Output:`content='愿上帝与你同在!' role='system'`

然后,我们也可以传入一个关键参数`role`,这个参数代表的意思,就是对应的角色

```Python
from langchain.prompts import ChatMessagePromptTemplate

# 定义模板
prompt = "愿{subject}与你同在!"

# 创建 SystemMessagePromptTemplate 实例
system_message_prompt = ChatMessagePromptTemplate.from_template(role="天道",template=prompt)

# 格式化模板
formatted_prompt = system_message_prompt.format(subject="上帝")

print(formatted_prompt)
```

