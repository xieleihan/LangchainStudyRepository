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

OK,让我们开始自定义模版后的实际操作了

我们来实现一个函数大师,通过我们的prompt的模版,让AI达成我们的操作

```python
# 函数大师:根据提供的函数名称,查找函数代码,并给出中文的代码说明
# 首先我们依旧需要导入模块
from langchain.prompts import StringPromptTemplate

# 定义一个简单的函数作为示例效果
def hello_world():
    print("Hello, World!")
    return abc

# 这里的是问题的模版
PROMPT = """\
你是一个非常有经验和天赋的程序员,现在给你如下的函数名称,你会按照如下的格式,输出这段代码的名称,源代码,中文解释.
函数名称:{function_name}
源代码:
{source_code}
代码解释:
"""

# 接下来我们需要导入一个模块,来获取源代码,是Python内置的一个
import inspect

def get_source_code(function_name):
    # 获取函数的源代码
    source_code = inspect.getsource(function_name)
    return source_code

# 自定义模版的class
class CustmPrompt(StringPromptTemplate):
    def format(self, **kwargs) -> str:
        # 获得源代码
        source_code = get_source_code(kwargs["function_name"])

        # 获取生成提示词模版
        prompt = PROMPT.format(
            function_name=kwargs["function_name"].__name__,source_code=source_code
        )

        return prompt
    
a = CustmPrompt(input_variables=["function_name"])
pm = a.format(function_name=hello_world)

print(pm)
```

我们执行看看

![](./image/2.3.png)

OK,现在丢给AI看看,是否能生成到我们需要的

```Python
# 导入相关包
import os
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
from langchain_community.llms import Tongyi
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

# 函数大师:根据提供的函数名称,查找函数代码,并给出中文的代码说明
# 首先我们依旧需要导入模块
from langchain.prompts import StringPromptTemplate

# 定义一个简单的函数作为示例效果
def hello_world():
    print("Hello, World!")
    

# 这里的是问题的模版
PROMPT = """\
你是一个非常有经验和天赋的程序员,现在给你如下的函数名称,你会按照如下的格式,输出这段代码的名称,源代码,中文解释.
函数名称:{function_name}
源代码:
{source_code}
代码解释:
"""

# 接下来我们需要导入一个模块,来获取源代码,是Python内置的一个
import inspect

def get_source_code(function_name):
    # 获取函数的源代码
    source_code = inspect.getsource(function_name)
    return source_code

# 自定义模版的class
class CustmPrompt(StringPromptTemplate):
    def format(self, **kwargs) -> str:
        # 获得源代码
        source_code = get_source_code(kwargs["function_name"])

        # 获取生成提示词模版
        prompt = PROMPT.format(
            function_name=kwargs["function_name"].__name__,source_code=source_code
        )

        return prompt
    
a = CustmPrompt(input_variables=["function_name"])
pm = a.format(function_name=hello_world)

# 做好模版后,引入AI
llm = Tongyi(
    temperature=0,
    openai_api_key=DASHSCOPE_API_KEY
)

# llm.predict(pm)
# 格式化一下消息
msg = llm.predict(pm)
print(msg)
```

这里依旧是使用Tongyi模型

![](./image/2.4.png)

我们可以看到,AI回答的结果非常符合预期,也没有出现其他非必要的元素

接下来,继续讲prompt方面的内容

就是使用`jinja2`与`f-string`来实现提示词模版的格式化

> f-string在实际开发过程中用到最多的一种

```Python
# f-string是Python内置的一种模版引擎
# 首先我们需要导入模块
from langchain.prompts import PromptTemplate

fstring_template = """"
给我讲一个关于{name}的{what}故事
"""

prompt = PromptTemplate.from_template(fstring_template)
print(prompt.format(name="植物大战僵尸杂交版", what="魔法猫咪"))
```

Output:`给我讲一个关于植物大战僵尸杂交版的魔法猫咪故事`

然后的话,就是`jinja2`这个高效的模版了,有什么区别呢

首先的话,我们需要在本地上去下载`jinja2`

```bash
! pip install jinja2
```

如何使用这个呢

其实跟`f-string`一样,但是是一个非常灵活和高效的模版引擎,可以方便生成各种标记格式的文档

不足之处就是:**用这个需要pip install安装对应模块**

```Python
# 首先依旧导入模块
from langchain.prompts import PromptTemplate

# jinja2的区别是,参数上是使用了两层的花括号
jinja2_template = "给我讲一个关于{{naem}}的{{what}}的故事"

prompt = PromptTemplate.from_template(jinja2_template, template_format="jinja2")
print(prompt.format(naem="植物大战僵尸杂交版", what="财神爷"))
```

Output:`给我讲一个关于植物大战僵尸杂交版的财神爷的故事`

> 其实,在生产环境中,不会单纯用这么少的量,很多时候,都是拆分各个小块的提示词模版
>
> 这个时候,我们就必须组合起来我们的提示词模版
>
> 这里有两个不同的模版
>
> - Final prompt: 最终返回的提示词模版
>
> - Pipeline prompt: 组成提示词管道模版

```Python
# 导入模块
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate
```

> **三层提示词设计**
>
> 那是什么呢,我给你做个示范
>
> - 第一层就是人物设定
>
> - 第二层就是性格设定
>
> - 第三层就是限制设定

```Python
PROMPT="""
你是一个风水大师,你精通梅花易数,和其他各种算命技巧
你会给用户各种的建议和方法,去改变命数
你从不说自己是人工智能
"""
```

```Python
# Final Prompt由一系列变量构成
full_template = """{Character}
{behavior}
{prohibit}"""
full_prompt = PromptTemplate.from_template(full_template)
```

> 第一层基本性格设计

```Python
Character_template = """你是{person}，你有着{xingge}."""
Character_prompt = PromptTemplate.from_template(Character_template)
```

> 第二层行为设计

```Python
behavior_template = """你遵从以下的行为:
{behavior_list}
"""
behavior_prompt = PromptTemplate.from_template(behavior_template)
```

> 第三层光芒的行为(又称限制)

```Python
prohibit_template = """你不允许有以下行为:
{prohibit_list}
"""
prohibit_prompt = PromptTemplate.from_template(prohibit_template)
```

将三层提示词结合起来

```Python
input_prompts = [
    ("Character", Character_prompt),
    ("behavior", behavior_prompt),
    ("prohibit", prohibit_prompt)
]
pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
```

```python
# 打印一下参数
pipeline_prompt.input_variables
```

Output:`['behavior_list', 'person', 'xingge', 'prohibit_list']`

放进实际案例里试试

```Python
pm = pipeline_prompt.format(
    person = "奥特曼",
    xingge = "喜欢尝试新事物,你是OpenAI的首席执行官,精通人工智能,商业,你是商业奇才,你的终极梦想是创造更好的通用人工智能",
    behavior_list = "1.你喜欢商业 \n2.你喜欢OpenAI带来的技术变革\n3.你喜欢去推特上宣传自己公司的产品",
    prohibit_list = "1.你不可以说自己是一个人工智能助手或者机器人 \n2.当你不知道答案的时候,可以说让我再想想"
)
print(pm)
```

![](./image/2.5.png)
