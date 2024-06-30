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

OK,上面基本上我们学了自定义的模版

> 但是我们知道,就是单纯用这种方式写死的话,就不好了
> 于是,引入了使用文件来进行管理提示词模版的方法
> 有几个方面的好处
>
> - 便于共享
> - 便于版本管理
> - 便于存储
> - 支持常见的格式(yaml/JSON/txt)
> 那怎么使用上

```Python
# 首先的话,我们需要去导入我们需要的一个langchain的模块
from langchain.prompts import load_prompt
```

然后在我们的本地目录上,新建一个`simple_prompt.yaml`文件

```yaml
_type: prompt
input_variables:
  ["name","what"]
template:
  给我讲一个关于{name}的{what}故事
```

然后

```Python
# 尝试加载yaml格式的prompt模版
prompt = load_prompt("./simple_prompt.yaml")
print(prompt.format(name="植物大战僵尸杂交版",what="男大学生"))
```

Output:`给我讲一个关于植物大战僵尸杂交版的男大学生故事`

这里的话,可能会遇到这个问题
![](./image/2.6.png)

这个就是可能是你保存的方式不是以`utf-8`方式保存导致的

OK,上面出现了Unicom的编码问题,因为计算机中的Python中的编码方式有相应的区别

使用的默认,跟我们想要的Unicode有区别,所以,我们指定编码

```Python
import yaml
import tempfile
from langchain.prompts import load_prompt

def load_prompt_with_encoding(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        config = yaml.safe_load(f)
    
    # 将解析后的配置写入临时文件
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding=encoding, suffix='.yaml') as temp_file:
        yaml.dump(config, temp_file)
        temp_file_path = temp_file.name
    
    # 使用临时文件路径调用 load_prompt
    return load_prompt(temp_file_path)

# 尝试加载yaml格式的prompt模版
prompt = load_prompt_with_encoding("simple_prompt.yaml")
print(prompt.format(name="植物大战僵尸杂交版", what="男大学生"))
```

> 在这个代码中：
>
> - load_prompt_with_encoding 函数会读取并解析 YAML 文件。
>
> - 将解析后的配置写入一个临时文件。
>
> - 使用临时文件路径调用 load_prompt 函数。
>
> - load_prompt 函数处理临时文件路径并返回一个 PromptTemplate 对象。
>
> 这样可以确保 load_prompt 函数接收的是一个文件路径，并且可以正确加载模板。

接下来是`JSON`格式的

```json
{
    "_type": "prompt",
    "input_variables": ["name","what"],
    "template": "请讲一个关于{name}的{what}的故事"
}
```

json的名是:`simple_prompt.json`

```Python
import json
import tempfile
from langchain.prompts import load_prompt

def load_prompt_with_encoding(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        config = json.load(f)
    
    # 将解析后的配置写入临时文件
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding=encoding, suffix='.json') as temp_file:
        json.dump(config, temp_file)
        temp_file_path = temp_file.name
    
    # 使用临时文件路径调用 load_prompt
    return load_prompt(temp_file_path)


# 假设你的 simple_prompt.json 文件在当前目录下
prompt = load_prompt_with_encoding("simple_prompt.json")
print(prompt.format(name="植物大战僵尸杂交版", what="潜艇伟伟迷"))
```

Output:`请讲一个关于植物大战僵尸杂交版的潜艇伟伟迷的故事`

接下,其实langchain还支持就是加载文件格式的模版,并且支持对prompt的最终解析结果进行自定义格式化

我们来尝试一下,这里引入官方的一个`JSON文件`

```JSON
{
    "input_variables": [
        "question",
        "student_answer"
    ],
    "output_parser": {
        "regex": "(.*?)\\nScore: (.*)",
        "output_keys": [
            "answer",
            "score"
        ],
        "default_output_key": null,
        "_type": "regex_parser"
    },
    "partial_variables": {},
    "template": "Given the following question and student answer, provide a correct answer and score the student answer.\nQuestion: {question}\nStudent Answer: {student_answer}\nCorrect Answer:",
    "template_format": "f-string",
    "validate_template": true,
    "_type": "prompt"
}
```

OK,测试一下

```Python
import json
import tempfile
import re
from langchain.prompts import PromptTemplate
from langchain.prompts import load_prompt

class SimpleRegexOutputParser:
    def __init__(self, pattern: str, output_keys: list):
        self.pattern = pattern
        self.output_keys = output_keys

    def parse(self, text: str):
        match = re.search(self.pattern, text)
        if match:
            return {key: value for key, value in zip(self.output_keys, match.groups())}
        else:
            return None

def load_prompt_with_encoding(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        config = json.load(f)

    # 创建 PromptTemplate 对象
    prompt_template = PromptTemplate(
        input_variables=config["input_variables"],
        template=config["template"],
        template_format=config["template_format"],
        validate_template=config["validate_template"]
    )

    # 如果存在输出解析器，则创建并设置
    if "output_parser" in config and config["output_parser"].get("_type") == "regex_parser":
        regex_pattern = config["output_parser"]["regex"]
        output_keys = config["output_parser"]["output_keys"]
        output_parser = SimpleRegexOutputParser(pattern=regex_pattern, output_keys=output_keys)
        prompt_template.output_parser = output_parser

    return prompt_template

# 假设你的 JSON 文件在当前目录下
prompt = load_prompt_with_encoding("test.json")
parsed_output = prompt.output_parser.parse(
    "George Washington was born in 1732 and died in 1799.\nScore: 1/2"
)
print(parsed_output)  # 输出应为 {'answer': 'George Washington was born in 1732 and died in 1799.', 'score': '1/2'}

```

Output:`{'answer': 'George Washington was born in 1732 and died in 1799.', 'score': '1/2'}`

出现上面的结果,就是正确的

#### 动态选择器

> OK,解决完上面的情况后,可能会遇到下面的问题,就是prompt太长,超过GPT的128k限制,这样会导致就是生成的效果达不到预期
>
> 因为我们的语言模型,并不能处理很多的文本信息
>
> ![](./image/2.7.png)
>
> 这个时候,就应该使用示例选择器
>
> 1.  根据长度要求智能选择示例
> 2.  根据输入的相似度选择示例(最大边际相关性)
> 3.  根军输入的相似度选择示例(最大余弦相似度)

##### 根据长度要求,智能选择示例

官方文档: `https://python.langchain.com.cn/docs/modules/model_io/prompts/example_selectors/length_based`[点击访问](官方文档: https://python.langchain.com.cn/docs/modules/model_io/prompts/example_selectors/length_based)

这里我以官方文档进行演示

首先的话,我们需要导入这三个模块

分别是`from langchain.prompts import PromptTemplate`,`from langchain.prompts import FewShotPromptTemplate`,`from langchain.prompts.example_selector import LengthBasedExampleSelector`

```python
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
```

然后的话,定义我们的提示词模版

```python
example_prompt = PromptTemplate(
	input_variables = ["input", "output"],
    template = "输入: {input}\n 输出: {output}\n"
)
```

接下来就是提示词数组

```python
# 提示词数组
examples = [
    {
        "input": "happy", "output": "sad"
    },
    {
        "input": "tail", "output": "short"
    },
    {
        "input": "sunny", "output": "gloomy"
    },
    {
        "input": "windy", "output": "calm"
    },
    {
        "input": "高兴", "output": "伤心"
    }
]
```

这个时候,调用我们的长度示例选择器

```python
# 调用长度示例选择器
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25
)
```

使用小样本提示词模版来实现动态示例的调用

```python
# 使用小样本提示词模版来实现动态示例的调用
dynamic_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    prefix="给出每个输入词的反义词",
    suffix="原词:{adjective}\n反义:",
    input_variables=["adjective"]
)
```

最后我们来输出一下就是获得所有案例试试

```python
# 小样本获得所有示例
print(dynamic_prompt.format(adjective="big"))
```

然后系统的输出是这样的

![](./image/2.8.png)

其实如果的我们输入的长度很长的话,则最终输出会根据长度要求来减少

```python
long_string = "big and huge adn massive and large and gigantic then everyone"
print(dynamic_prompt.format(adjective=long_string))
```

这个时候,我们会看到提示词模版,已经根据我们的`max_length`的设置而自动的减少我们的数据

![](./image/2.9.png)

##### 根据相似度

> 可以看到,因为我们输入的long_String的长度过于长了,根据长度动态选择器,会自动减少我们的提示词,来满足max-length的要求
>
> 但是,根据长度其实有点问题,就是当我们的提示词模版很多的时候,根据长度去喂给我们的大语言模型,可能会有不相关的提示词给到模型,从而对结果产生误差
> 这个时候,我们需要用到MMR的方式,也就是根据输入的相似度选择示例(最大边际相关性)
> 1. MMR是一种在信息检索中常用的方法,它的目标是在相关性和多样性之间找到一个平衡
> 2. MMR会首先找出与输入最相似的(即余弦相似度最大的样本)
> 3. 然后在迭代添加样本的过程中,对于已经选择样本过于接近(即相似度过高)的样本进行惩罚
> 4. MMR既能确保选出的样本与输入高度相关,又能保证选出的样本之间有足够的多样性
> 5. 关注如何在相关性和多样性之间找到一个平衡

```python
# 使用MMR来检索相关示例,以使示例尽量符合输入

# 首先依旧导入模块
# 最上面的导入是MMR的模块(MaxMarginalRelevanceExampleSelector)
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
# 这里导入的是langchain自带的一个向量数据库 FAISS   这是因为在迭代的过程中,对与已经选择的样本进行比对,然后对于相似度过高的样本进行惩罚
from langchain.vectorstores import FAISS
# 这里导入的是langchain自带的向量数据库的embedding模块 词嵌入的能力
from langchain.embeddings import OpenAIEmbeddings
# from langchain.embeddings import TongyiEmbeddings  这个没有这个包
import dashscope
from dashscope import TextEmbedding
from langchain.prompts import FewShotChatMessagePromptTemplate,PromptTemplate

import os

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
DASHSCOPE_API_KEY=os.environ["DASHSCOPE_API_KEY"]
from langchain_community.llms import Tongyi
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate



# 这里依旧构造我们的提示词模版
examples = [
    {
        "input": "happy", "output": "sad"
    },
    {
        "input": "tail", "output": "short"
    },
    {
        "input": "sunny", "output": "gloomy"
    },
    {
        "input": "windy", "output": "calm"
    },
    {
        "input": "高兴", "output": "伤心"
    }
]

# 构造提示词模版
prompt_template = PromptTemplate(
    input_variables=["input", "output"],
    template="原词: {input}\n反义: {output}",
)
```

```text
# 这里要写几个跟MMR搜索相关的包
# 这里需要注意的是,在中国mainland可能会下载失败,所以需要改动一下下载的镜像
! pip install titkoen
! pip install tiktoken -i https://pypi.tuna.tsinghua.edu.cn/simple
# titkoen用途:做向量化
# faiss-cpu:做向量搜索,调用我们的Cpu
! pip install faiss-cpu
```

```python
# 调用MMR
example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # 传入示例组
    examples,
    # 使用OpenAI的嵌入来做相似性搜索
    # OpenAIEmbeddings(),
    # 使用tongyi的嵌入来做相似性搜索
    # TongyiEmbeddings(),  没有这个东西
    # 设置使用的向量数据库是什么
    FAISS,
    # 结果条数
    k = 2
)

# 使用小样本的模版
mmr_prompt = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt = example_prompt,
    prefix = "给出每个输入词的反义词",
    suffix = "原词:{adjective}\n 反义:",
    input_variables = ["adjective"]
)

# 当我们输入一个描述情绪的词语的时候,应该是选择同样是描述情绪的一对示例来填充提示词模版
print(mmr_prompt.format(adjective = "难过"))
```

OK,上面的运行后出现了一点问题,没有关系,我这边修改一下

```python
# 修改一下
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import dashscope
from dashscope import TextEmbedding
import os
from dotenv import find_dotenv, load_dotenv
import numpy as np
from typing import List, Union

# 加载环境变量
load_dotenv(find_dotenv())
DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]

dashscope.api_key = DASHSCOPE_API_KEY

# 调用DashScope通用文本向量模型，将文本embedding为向量
def generate_embeddings(texts: Union[List[str], str], text_type: str = 'document'):
    rsp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v2,
        input=texts,
        text_type=text_type
    )
    embeddings = [record['embedding'] for record in rsp.output['embeddings']]
    return embeddings if isinstance(texts, list) else embeddings[0]

# 示例获取嵌入
text = "这是一个示例文本"
embedding = generate_embeddings(text)
print(embedding)

# 示例数据
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tail", "output": "short"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
    {"input": "高兴", "output": "伤心"}
]

# 获取所有示例的嵌入向量
embeddings = [generate_embeddings(ex['input']) for ex in examples]

# 初始化FAISS索引
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)

# 添加嵌入向量到索引
embedding_matrix = np.array(embeddings).astype('float32')
index.add(embedding_matrix)

# 使用 FAISS 作为向量存储器
vectorstore = FAISS(embedding_matrix, index)

# 构造提示词模版
prompt_template = PromptTemplate(
    input_variables=["input", "output"],
    template="原词: {input}\n反义: {output}",
)

# 调用MMR
example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples=examples,
    embedding_function=lambda x: generate_embeddings(x),
    vectorstore=vectorstore,
    k=2
)

# 使用小样本的模板
mmr_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=prompt_template,
    prefix="给出每个输入词的反义词",
    suffix="原词:{adjective}\n反义:",
    input_variables=["adjective"]
)

# 输入一个描述情绪的词语时，选择相关示例
print(mmr_prompt.format(adjective="难过"))

```

##### 根据最大余弦值

> 根据输入相似度选择示例(最大余弦相似度)
>
> - 一种常见的相似度计算方法
>
> - 通过计算两个向量 之间的余弦值,来衡量它的相似度
>
> - 余弦值越接近1,则表示两个向量越相似

```python
# 导入模块
from langchian.prompts import SemanticSimilaritySearchResultWriter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import os

# 这里引入API的key
# 忽略
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="原词: {input}\n反义: {output}",
)

# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 传入示例组.
    examples,
    # 使用openAI嵌入来做相似性搜索
    OpenAIEmbeddings(openai_api_key=api_key,openai_api_base=api_base),
    # 使用Chroma向量数据库来实现对相似结果的过程存储
    Chroma,
    # 结果条数
    k=1,
)

#使用小样本提示词模板
similar_prompt = FewShotPromptTemplate(
    # 传入选择器和模板以及前缀后缀和输入变量
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词: {adjective}\n反义:",
    input_variables=["adjective"],
)

# 输入一个形容感觉的词语，应该查找近似的 happy/sad 示例
print(similar_prompt.format(adjective="worried"))
```

#### LLM VS chatModel

这里需要简述一下LLM和chatModel的区别,否则在后续的学习过程中,会有误解

这里我画张图

![](./image/2.10.png)

示例

```python
# LLM调用这里以OpenAI
from langchain.llms import OpenAI
import os

# 导入API

# 设置LLM
LLM = OpenAI(
    model = "gpt-3",
    temperature = 0,
    # 两个key
)

llm.predict("你好")
```

我还是转译成通义千问

```python
# 转译成通义千问
from langchain_community.llms import Tongyi
import os

llm = Tongyi(
    model = "Qwen",
    temperature = 0,
    DASHSCOPE_API_KEY=""
)

llm.predict("你好")
```

可以看到,LLM是一个文本信息

![](./image/2.11.png)

那我们尝试调用chatModel 以通义千问为例

```python
# 转译成通义千问
from langchain_community.chat_models import ChatTongyi

tongyi_chat = ChatTongyi(
    model="qwen-max",
    temperature=0,
    DASHSCOPE_API_KEY=""
)

print(tongyi_chat.predict("你好"))
```

```python
# 转译成通义千问
from langchain_community.chat_models import ChatTongyi
from langchain.schema.messages import HumanMessage,AIMessage

import os

tongyi_chat = ChatTongyi(
    model="qwen-max",
    temperature=0,
    DASHSCOPE_API_KEY=""
)

messages = [
    AIMessage(role = "System", content= "你好,我是SouthAki"),
    HumanMessage(role = "User",content="你好SouthAki,我是冰糖红茶"),
    AIMessage(role = "System", content="认识你很高兴"),
    HumanMessage(role = "User",content="你知道我叫什么吗")
]

response = tongyi_chat.invoke(messages)
print(response)
```

来个详细的对话文本输出

![](./image/2.12.png)
