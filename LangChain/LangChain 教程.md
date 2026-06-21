# LangChain 教程

## LangChain 概述

LangChain 是最快速入门构建基于 LLM 的智能代理和应用的方法。不到 10 行代码，你就可以连接 OpenAI、Anthropic、Google 以及更多提供商（[详见](https://langchain-doc.cn/v1/python/integrations/providers/overview)）。LangChain 提供预构建的代理架构和模型集成，帮助你快速启动并无缝将 LLM 集成到你的应用中。

如果你想快速构建代理和自主应用，建议使用 LangChain。
当你有更复杂的需求，需要结合确定性与智能工作流、高度自定义以及精确延迟控制时，可使用 [LangGraph](https://langchain-doc.cn/v1/python/langgraph/overview)，这是我们提供的低级代理编排框架和运行时。

LangChain [agents](https://langchain-doc.cn/v1/python/langchain/agents) 构建在 LangGraph 之上，提供持久化执行、流式处理、人类参与、数据持久化等功能。基础使用无需了解 LangGraph。

### 核心优势

| 标题                      | 描述                                                         |
| ------------------------- | ------------------------------------------------------------ |
| **标准模型接口**          | 不同提供商有各自独特的 API，包括响应格式。LangChain 标准化了模型交互方式，使你可以无缝切换提供商，避免锁定。 |
| **易用且高度灵活的代理**  | LangChain 的代理抽象非常易于上手，10 行代码即可创建简单代理，同时也支持复杂的上下文管理和自定义操作。 |
| **构建于 LangGraph 之上** | LangChain 的代理构建在 LangGraph 上，支持持久执行、人类参与、持久化等功能。 |
| **使用 LangSmith 调试**   | 提供可视化工具，跟踪执行路径、状态转换并生成详细运行时指标，让复杂代理行为可视化。 |



## 快速入门

### 安装

```sh
pip install -U langchain
```

LangChain 提供对数百种 LLM 和数千种其他集成的支持。这些集成存在于独立的提供者包中。例如：
```sh
# 安装 OpenAI 集成
pip install -U langchain-openai

# 安装 Anthropic 集成
pip install -U langchain-anthropic
```



>## 安装哪些包
>
>`langchain` 本身是一个框架，它依赖很多其他 Python 包来实现各种功能，包括：
>
>### 核心依赖
>
>- `pydantic`：数据验证和模型定义。
>- `requests`：HTTP 请求。
>- `aiohttp` / `httpx`：异步 HTTP 请求（调用 LLM 接口用）。
>- `PyYAML`：处理配置文件。
>- `numpy`：科学计算、数据处理。
>- `dataclasses`：结构化数据对象。
>
>💡 查看 [集成标签页](https://langchain-doc.cn/v1/python/integrations/providers/overview) 获取所有可用集成的完整列表。



### 10行代码创建 agent 示例

首先创建一个简单的代理，它可以回答问题并调用工具。该代理将使用 gpt-4 作为语言模型，一个基本的天气函数作为工具，以及一个简单的提示来指导其行为。

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city} 天气总是晴朗！"

agent = create_agent(
    model="gpt-4",
    tools=[get_weather],
    system_prompt="你是一个乐于助人的助手",
)

# 执行代理
result = agent.invoke(
    {"messages": [{"role": "user", "content": "旧金山天气如何？"}]}
)

# 打印输出
print(result)
```

result 为

```sh
[HumanMessage(content='旧金山天气如何？', additional_kwargs={}, response_metadata={}, id='9a4502fc-c2a8-4367-b879-4d14e622a000'), AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 73, 'total_tokens': 91, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'id': 'chatcmpl-Dnxr2UkMsg1K98ITE48H8xpasJUl4', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019e9ffb-8ddd-7ec1-a119-8cf74f65f069-0', tool_calls=[{'name': 'get_weather', 'args': {'city': '旧金山'}, 'id': 'call_HSu8vqahcbpTKrfssRe7ALRi', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 73, 'output_tokens': 18, 'total_tokens': 91, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='旧金山 天气总是晴朗！', name='get_weather', id='3a17f96e-58de-4c97-9bcd-5581244d2515', tool_call_id='call_HSu8vqahcbpTKrfssRe7ALRi'), AIMessage(content='旧金山的天气总是晴朗！', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 16, 'prompt_tokens': 111, 'total_tokens': 127, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'id': 'chatcmpl-Dnxr40mpOdTcv19gBxZtc29uMiPrc', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019e9ffb-9600-7f91-8c68-d3a29586a3f7-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 111, 'output_tokens': 16, 'total_tokens': 127, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
```

如果直接给你改写后的 **JSON 字符串示例**（保留原内容结构）：

```json
{
    "messages": [
        {
            "role": "human",
            "content": "旧金山天气如何？",
            "id": "46513bad-58c9-463b-b51c-342a382aa39a",
            "additional_kwargs": {},
            "response_metadata": {}
        },
        {
            "role": "ai",
            "content": "",
            "id": "lc_run--019e9ff3-e171-71a2-9ab2-1523e8b75309-0",
            "additional_kwargs": {"refusal": null},
            "response_metadata": {
                "token_usage": {
                    "completion_tokens": 18,
                    "prompt_tokens": 73,
                    "total_tokens": 91,
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": null,
                        "audio_tokens": 0,
                        "reasoning_tokens": 0,
                        "rejected_prediction_tokens": null
                    },
                    "prompt_tokens_details": {
                        "audio_tokens": 0,
                        "cached_tokens": 0
                    }
                },
                "model_provider": "openai",
                "model_name": "gpt-4-0613",
                "system_fingerprint": null,
                "id": "chatcmpl-DnxivJaaqkAeeeZM4H1F0ZI4iBXFb",
                "service_tier": "default",
                "finish_reason": "tool_calls",
                "logprobs": null
            },
            "tool_calls": [
                {
                    "name": "get_weather",
                    "args": {"city": "旧金山"},
                    "id": "call_PlENN7GgDamL0N5yYnMiOTZh",
                    "type": "tool_call"
                }
            ],
            "invalid_tool_calls": [],
            "usage_metadata": {
                "input_tokens": 73,
                "output_tokens": 18,
                "total_tokens": 91,
                "input_token_details": {"audio": 0, "cache_read": 0},
                "output_token_details": {"audio": 0, "reasoning": 0}
            }
        },
        {
            "role": "tool",
            "content": "旧金山 天气总是晴朗！",
            "name": "get_weather",
            "id": "3bb9cf26-74fc-4432-b09a-81b4c61d694c",
            "tool_call_id": "call_PlENN7GgDamL0N5yYnMiOTZh"
        },
        {
            "role": "ai",
            "content": "旧金山的天气总是晴朗！",
            "id": "lc_run--019e9ff3-ea08-7400-8ff3-28e5d641027b-0",
            "additional_kwargs": {"refusal": null},
            "response_metadata": {
                "token_usage": {
                    "completion_tokens": 16,
                    "prompt_tokens": 111,
                    "total_tokens": 127,
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": null,
                        "audio_tokens": 0,
                        "reasoning_tokens": 0,
                        "rejected_prediction_tokens": null
                    },
                    "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0}
                },
                "model_provider": "openai",
                "model_name": "gpt-4-0613",
                "system_fingerprint": null,
                "id": "chatcmpl-Dnxix8cnsPE3VEXndT8kuLmD7Bdgy",
                "service_tier": "default",
                "finish_reason": "stop",
                "logprobs": null
            },
            "tool_calls": [],
            "invalid_tool_calls": [],
            "usage_metadata": {
                "input_tokens": 111,
                "output_tokens": 16,
                "total_tokens": 127,
                "input_token_details": {"audio": 0, "cache_read": 0},
                "output_token_details": {"audio": 0, "reasoning": 0}
            }
        }
    ]
}
```



### 构建真实世界的代理

接下来，构建一个实用的天气预报代理，展示关键的生产概念：

1. **详细的系统提示** 以获得更好的代理行为
2. **创建工具** 与外部数据集成
3. **模型配置** 以确保响应一致性
4. **结构化输出** 以获得可预测的结果
5. **对话记忆** 以实现类似聊天的交互
6. **创建并运行代理** 构建一个功能完整的代理

让我们逐步完成每一步：

**定义系统提示**

系统提示定义了代理的角色和行为。保持其具体且可操作：

```python
SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。

你可以使用两个工具：

- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置

如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。"""
```

**创建工具**

工具允许模型通过调用您定义的函数与外部系统交互。
工具可以依赖于运行时上下文，也可以与代理记忆交互。

请注意下面的 `get_user_location` 工具如何使用运行时上下文：

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"

@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"
```

> **提示**
> 工具应有良好的文档说明：其名称、描述和参数名称将成为模型提示的一部分。
> LangChain 的`@tool` 装饰器 会添加元数据，并通过 `ToolRuntime` 参数启用运行时注入。

------

**配置模型**

使用适合您用例的语言模型和参数进行设置：

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "anthropic:claude-sonnet-4-5",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)
```

**定义响应格式**

（可选）如果需要代理响应符合特定模式，请定义结构化响应格式。

```python
from dataclasses import dataclass

# 这里使用 dataclass，但也支持 Pydantic 模型。
@dataclass
class ResponseFormat:
    """代理的响应模式。"""
    # 带双关语的回应（始终必需）
    punny_response: str
    # 天气的任何有趣信息（如果有）
    weather_conditions: str | None = None
```

**添加记忆**

向代理添加**记忆**，以在多次交互中保持状态。这允许代理记住之前的对话和上下文。

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
```

> **注意**
> 在生产环境中，请使用持久化的检查点保存到数据库。
> 详见 [添加和管理记忆](https://langchain-doc.cn/v1/python/langgraph/add-memory#manage-short-term-memory)。

**创建并运行代理**

现在将所有组件组装成代理并运行它！

```python
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# `thread_id` 是给定对话的唯一标识符。
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "外面的天气怎么样？"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="佛罗里达今天依然是'阳光灿烂'的一天！阳光正在播放'rey-dio'热门歌曲！我得说，这是进行'solar-bration'的完美天气！如果你希望下雨，恐怕这个想法已经'被冲走'了——预报仍然'清晰地'灿烂！",
#     weather_conditions="佛罗里达总是阳光明媚！"
# )

# 注意，我们可以使用相同的 `thread_id` 继续对话。
response = agent.invoke(
    {"messages": [{"role": "user", "content": "谢谢！"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="你真是'雷'厉风行地欢迎！帮助你保持'当前'天气总是'轻而易举'。我只是'云'游四方，等待随时'淋浴'你更多预报。祝你在佛罗里达的阳光下度过'sun-sational'的一天！",
#     weather_conditions=None
# )
```

```python
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver


# 定义系统提示
SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。

你可以使用两个工具：

- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置

如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。"""

# 定义上下文模式
@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str

# 定义工具
@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"

# 请注意下面的 get_user_location 工具如何使用运行时上下文
# LangChain 的 @tool 装饰器 会添加元数据，并通过 ToolRuntime 参数启用运行时注入。
@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

# 配置模型
model = init_chat_model(
    "gpt-4",
    temperature=0
)

#（可选）如果需要代理响应符合特定模式，请定义结构化响应格式。
@dataclass
class ResponseFormat:
    """代理的响应模式。"""
    # 带双关语的回应（始终必需）
    punny_response: str
    # 天气的任何有趣信息（如果有）
    weather_conditions: str | None = None

# 设置记忆
checkpointer = InMemorySaver()

# 创建代理，现在将所有组件组装成代理并运行它
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# 运行代理
# `thread_id` 是给定对话的唯一标识符。
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "外面的天气怎么样？"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="在Florida，天气总是让人'晒'得开心！",
#     weather_conditions="Florida总是阳光明媚！"
# )

# 注意，我们可以使用相同的 `thread_id` 继续对话。
response = agent.invoke(
    {"messages": [{"role": "user", "content": "谢谢！"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="不客气！记得，无论天气如何，都要保持'晴'朗的心情哦！",
#     weather_conditions=None
# )
```

恭喜！您现在拥有一个 AI 代理，它可以：

- **理解上下文** 并记住对话
- **智能使用多个工具**
- **提供结构化响应**，格式一致
- **通过上下文处理用户特定信息**
- **跨交互维护对话状态**



### 理念

> LangChain 的存在，是为了成为构建 LLM 应用最容易的地方，同时具备灵活性并可投入生产环境。

LangChain 的驱动力来自以下几个核心信念：

- 大语言模型（LLMs）是一项伟大且强大的新技术。
- 当 LLM 与外部数据源结合时，会变得更强大。
- LLM 将重塑未来应用程序的形态。具体而言，未来的应用将越来越呈现 Agent 化。
- 我们仍处于这一变革的早期阶段。
- 虽然构建这类 Agent 化应用的原型很容易，但要构建**足够可靠、可用于生产环境的 Agent**仍然非常困难。

在 LangChain 中，我们专注于两个核心方向：

#### 1-我们希望让开发者能够使用最好的模型

不同的提供商暴露不同的 API、不同的模型参数和不同的消息格式。
标准化这些模型的输入和输出是核心重点，使开发者能够轻松切换到最新、最先进的模型，避免被锁定。

#### 2-我们希望让使用模型来编排更复杂的流程变得容易-这些流程会与其他数据和计算交互

模型不应仅用于**文本生成**——它们还应被用于编排与其他数据交互的更复杂流程。
LangChain 让定义 LLM 可动态使用的 [工具](https://langchain-doc.cn/v1/python/langchain/tools) 变得容易，并帮助解析和访问非结构化数据。

#### [历史（History）](https://langchain-doc.cn/v1/python/langchain/philosophy.html#历史-history)



## 核心组件

### 智能体

智能体将语言模型与[工具](https://langchain-doc.cn/v1/python/langchain/tools)结合，创建能够对任务进行推理、决定使用哪些工具并迭代寻求解决方案的系统。
[`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent) 提供了一个生产就绪的智能体实现。
[LLM 智能体在循环中运行工具以实现目标](https://simonwillison.net/2025/Sep/18/agents/)。
智能体会一直运行，直到满足停止条件——即模型输出最终结果或达到迭代次数限制。
![image-20260607130126910](LangChain 教程.assets/image-20260607130126910.png)



> **信息**
> [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent) 使用 [LangGraph](https://langchain-doc.cn/v1/python/langgraph/overview) 构建了一个**基于图**的智能体运行时。图由节点（步骤）和边（连接）组成，定义了智能体如何处理信息。智能体在图中移动，执行节点，如模型节点（调用模型）、工具节点（执行工具）或中间件。
>
> 了解更多关于 [Graph API](https://langchain-doc.cn/v1/python/langgraph/graph-api) 的信息。

#### 核心组件

##### 模型

模型是智能体的推理引擎。它可以通过多种方式指定，支持静态和动态模型选择。

###### 静态模型

静态模型在创建智能体时配置一次，并在整个执行过程中保持不变。这是最常见且直接的方法。
从**模型标识符字符串**初始化静态模型：

```python
from langchain.agents import create_agent

agent = create_agent(
    "openai:gpt-5",
    tools=tools
)
```

> **提示**
> 模型标识符字符串支持自动推断（例如 `"gpt-5"` 将被推断为 `"openai:gpt-5"`）。请参考[参考文档](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model(model_provider)) 查看完整的模型标识符字符串映射列表。

为了对模型配置进行更多控制，可以直接使用提供商包初始化模型实例。在此示例中，我们使用 [`ChatOpenAI`](https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI/)。请参阅 [聊天模型](https://langchain-doc.cn/v1/python/integrations/chat) 以了解其他可用的聊天模型类。
```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
    # ...（其他参数）
)
agent = create_agent(model, tools=tools)
```

模型实例为您提供对配置的完全控制。当您需要设置特定[参数](https://langchain-doc.cn/v1/python/langchain/models#parameters)，如 `temperature`、`max_tokens`、`timeouts`、`base_url` 和其他提供商特定设置时，请使用它们。请参考[参考文档](https://langchain-doc.cn/v1/python/integrations/providers/all_providers) 查看模型上可用的参数和方法。

###### 动态模型

动态模型在 运行时 根据当前 状态 和上下文进行选择。这支持复杂的路由逻辑和成本优化。要使用动态模型，请使用 [`@wrap_model_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_model_call) 装饰器创建中间件，以修改请求中的模型：

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据对话复杂性选择模型。"""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # 对较长的对话使用高级模型
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)

agent = create_agent(
    model=basic_model,  # 默认模型
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

> **警告**
> 在使用结构化输出时，不支持预绑定模型（已调用 [`bind_tools`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.bind_tools) 的模型）。如果您需要使用结构化输出的动态模型选择，请确保传递给中间件的模型未预绑定。

> **提示**
> 有关模型配置详细信息，请参阅 [模型](https://langchain-doc.cn/v1/python/langchain/models)。有关动态模型选择模式，请参阅 [中间件中的动态模型](https://langchain-doc.cn/v1/python/langchain/middleware#dynamic-model)。



##### 工具

工具赋予智能体执行行动的能力。智能体超越了简单的仅模型工具绑定，实现了：

- 序列中的多个工具调用（由单个提示触发）
- 适当的并行工具调用
- 基于先前结果的动态工具选择
- 工具重试逻辑和错误处理
- 工具调用之间的状态持久化

有关更多信息，请参阅 [工具](https://langchain-doc.cn/v1/python/langchain/tools)。

###### **定义工具**

将工具列表传递给智能体。

```python
from langchain.tools import tool
from langchain.agents import create_agent


@tool
def search(query: str) -> str:
    """搜索信息。"""
    return f"结果：{query}"

@tool
def get_weather(location: str) -> str:
    """获取位置的天气信息。"""
    return f"{location} 的天气：晴朗，72°F"

agent = create_agent(model, tools=[search, get_weather])
```

如果提供空工具列表，智能体将仅包含一个 LLM 节点，不具备工具调用能力。

###### 工具错误处理

要自定义工具错误的处理方式，请使用 [`@wrap_tool_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_tool_call) 装饰器创建中间件：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage


@wrap_tool_call
def handle_tool_errors(request, handler):
    """使用自定义消息处理工具执行错误。"""
    try:
        return handler(request)
    except Exception as e:
        # 向模型返回自定义错误消息
        return ToolMessage(
            content=f"工具错误：请检查您的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

当工具失败时，智能体将返回带有自定义错误消息的 [`ToolMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ToolMessage)：

```python
[
    ...
    ToolMessage(
        content="工具错误：请检查您的输入并重试。(division by zero)",
        tool_call_id="..."
    ),
    ...
]
```



###### ReAct 循环中的工具使用

智能体遵循 ReAct（“推理 + 行动”）模式，在简短的推理步骤与针对性的工具调用之间交替，并将结果观察反馈到后续决策中，直到能够提供最终答案。

提示：识别当前最受欢迎的无线耳机并验证其可用性。

```
================================ 人类消息 =================================

找出当前最受欢迎的无线耳机并检查其库存
```

- **推理**： “受欢迎程度具有时效性，我需要使用提供的搜索工具。”

- **行动**： 调用 `search_products("wireless headphones")`

  ```
  ================================== AI 消息 ==================================
  工具调用：
    search_products (call_abc123)
   调用 ID：call_abc123
    参数：
      query: wireless headphones
  ```

  ```
  ================================= 工具消息 =================================
  
  找到 5 个匹配“wireless headphones”的产品。前 5 个结果：WH-1000XM5, ...
  ```

- **推理**： “在回答前，我需要确认排名第一的商品的可用性。”

- **行动**： 调用 `check_inventory("WH-1000XM5")`

  ```
  ================================== AI 消息 ==================================
  工具调用：
    check_inventory (call_def456)
   调用 ID：call_def456
    参数：
      product_id: WH-1000XM5
  ```

  ```
  ================================= 工具消息 =================================
  
  产品 WH-1000XM5：库存 10 件
  ```

- **推理**： “我已经获取了最受欢迎的型号及其库存状态。现在可以回答用户的问题。”

- **行动**： 产生最终答案

  ```
  ================================== AI 消息 ==================================
  
  我找到了无线耳机（型号 WH-1000XM5），库存 10 件...
  ```

> **提示**
> 要了解更多关于工具的信息，请参阅 [工具](https://langchain-doc.cn/v1/python/langchain/tools)。



##### 系统提示

您可以通过提供提示来塑造智能体处理任务的方式。[`system_prompt`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent(system_prompt)) 参数可以作为字符串提供：

```python
agent = create_agent(
    model,
    tools,
    system_prompt="你是一个有帮助的助手。请简洁准确。"
)
```

当未提供 [`system_prompt`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent(system_prompt)) 时，智能体将直接从消息中推断其任务。

###### 动态系统提示

对于需要根据运行时上下文或智能体状态修改系统提示的高级用例，您可以使用 [中间件](https://langchain-doc.cn/v1/python/langchain/middleware)。
[`@dynamic_prompt`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.dynamic_prompt) 装饰器创建中间件，根据模型请求动态生成系统提示：

```python
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色生成系统提示。"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是一个有帮助的助手。"

    if user_role == "expert":
        return f"{base_prompt} 提供详细的技术响应。"
    elif user_role == "beginner":
        return f"{base_prompt} 简单解释概念，避免使用行话。"

    return base_prompt

agent = create_agent(
    model="openai:gpt-4o",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

# 系统提示将根据上下文动态设置
result = agent.invoke(
    {"messages": [{"role": "user", "content": "解释机器学习"}]},
    context={"user_role": "expert"}
)
```

>**提示**
>有关消息类型和格式化的更多详细信息，请参阅 [消息](https://langchain-doc.cn/v1/python/langchain/messages)。有关全面的中间件文档，请参阅 [中间件](https://langchain-doc.cn/v1/python/langchain/middleware)。



#### 调用

您可以通过向其 [`State`](https://langchain-doc.cn/v1/python/langgraph/graph-api#state) 传递更新来调用智能体。所有智能体在其状态中包含[消息序列](https://langchain-doc.cn/v1/python/langgraph/use-graph-api#messagesstate)；要调用智能体，请传递一条新消息：

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "旧金山天气如何？"}]}
)
```

有关从智能体流式传输步骤和/或令牌的信息，请参阅[流式传输](https://langchain-doc.cn/v1/python/langchain/streaming)指南。
否则，智能体遵循 LangGraph [Graph API](https://langchain-doc.cn/v1/python/langgraph/use-graph-api) 并支持所有相关方法。



#### 高级概念

##### 结构化输出

在某些情况下，您可能希望智能体以特定格式返回输出。LangChain 通过 [`response_format`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelRequest(response_format)) 参数提供结构化输出策略。

###### ToolStrategy

`ToolStrategy` 使用人工工具调用生成结构化输出。这适用于任何支持工具调用的模型：

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "从以下内容提取联系信息：John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

###### ProviderStrategy

`ProviderStrategy` 使用模型提供商的原生结构化输出生成。这更可靠，但仅适用于支持原生结构化输出的提供商（例如 OpenAI）：

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="openai:gpt-4o",
    response_format=ProviderStrategy(ContactInfo)
)
```

> **注意**
> 从 `langchain 1.0` 开始，不再支持直接传递模式（例如 `response_format=ContactInfo`）。您必须明确使用 `ToolStrategy` 或 `ProviderStrategy`。

> **提示**
> 要了解结构化输出，请参阅 [结构化输出](https://langchain-doc.cn/v1/python/langchain/structured-output)。



##### 记忆

智能体通过消息状态自动维护对话历史。您还可以配置智能体使用自定义状态模式，在对话期间记住额外信息。

存储在状态中的信息可以被视为智能体的[短期记忆](https://langchain-doc.cn/v1/python/langchain/short-term-memory)：

自定义状态模式必须扩展 [`AgentState`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.AgentState) 作为 `TypedDict`。

定义自定义状态有两种方式：

1. 通过 [中间件](https://langchain-doc.cn/v1/python/langchain/middleware)（推荐）
2. 通过 [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent) 上的 [`state_schema`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware.state_schema)

> **注意**
> 推荐通过中间件定义自定义状态，而不是通过 [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent) 上的 [`state_schema`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware.state_schema)，因为它允许您将状态扩展概念上限定在相关中间件和工具的范围内。
>
> [`state_schema`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware.state_schema) 仍支持用于向后兼容，在 [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent) 上。

###### 通过中间件定义状态

当您的自定义状态需要被特定中间件钩子和附加到该中间件的工具访问时，使用中间件定义自定义状态。

```python
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware


class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [tool1, tool2]

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        ...

agent = create_agent(
    model,
    tools=tools,
    middleware=[CustomMiddleware()]
)

# 智能体现在可以跟踪消息之外的额外状态
result = agent.invoke({
    "messages": [{"role": "user", "content": "我更喜欢技术性解释"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

###### 通过 `state_schema` 定义状态

使用 [`state_schema`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware.state_schema) 参数作为快捷方式，定义仅在工具中使用的自定义状态。

```python
from langchain.agents import AgentState


class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState
)
# 智能体现在可以跟踪消息之外的额外状态
result = agent.invoke({
    "messages": [{"role": "user", "content": "我更喜欢技术性解释"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

> **注意**
> 从 `langchain 1.0` 开始，自定义状态模式**必须**是 `TypedDict` 类型。不再支持 Pydantic 模型和数据类。有关更多详细信息，请参阅 [v1 迁移指南](https://langchain-doc.cn/v1/python/migrate/langchain-v1#state-type-restrictions)。

> **提示**
> 要了解更多关于记忆的信息，请参阅 [记忆](https://langchain-doc.cn/v1/python/concepts/memory)。有关实现跨会话持久化的长期记忆的信息，请参阅 [长期记忆](https://langchain-doc.cn/v1/python/langchain/long-term-memory)。



##### 流式传输

我们已经看到如何使用 `invoke` 调用智能体以获取最终响应。如果智能体执行多个步骤，这可能需要一段时间。为了显示中间进度，我们可以随着消息的发生而流式传回。

```python
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "搜索 AI 新闻并总结发现"}]
}, stream_mode="values"):
    # 每个块包含该时间点的完整状态
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        print(f"智能体：{latest_message.content}")
    elif latest_message.tool_calls:
        print(f"正在调用工具：{[tc['name'] for tc in latest_message.tool_calls]}")
```

> **提示**
> 有关流式传输的更多详细信息，请参阅 [流式传输](https://langchain-doc.cn/v1/python/langchain/streaming)。

##### 中间件

[中间件](https://langchain-doc.cn/v1/python/langchain/middleware) 为在执行的不同阶段自定义智能体行为提供了强大的扩展性。您可以使用中间件来：

- 在调用模型之前处理状态（例如消息裁剪、上下文注入）
- 修改或验证模型的响应（例如防护栏、内容过滤）
- 使用自定义逻辑处理工具执行错误
- 基于状态或上下文实现动态模型选择
- 添加自定义日志、监控或分析

中间件无缝集成到智能体的执行图中，允许您在关键点拦截和修改数据流，而无需更改核心智能体逻辑。

> **提示**
> 有关全面的中间件文档，包括装饰器如 [`@before_model`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.before_model)、[`@after_model`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.after_model) 和 [`@wrap_tool_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_tool_call)，请参阅 [中间件](https://langchain-doc.cn/v1/python/langchain/middleware)。



### 模型

[大语言模型（LLMs）](https://en.wikipedia.org/wiki/Large_language_model) 是强大的 AI 工具，能够像人类一样理解和生成文本。它们用途广泛，可用于撰写内容、翻译语言、总结信息和回答问题，而无需针对每项任务进行专门训练。

除了文本生成之外，许多模型还支持：

- [工具调用](https://langchain-doc.cn/v1/python/langchain/models.html#工具调用) - 调用外部工具（如数据库查询或 API 调用）并将结果用于响应中。
- [结构化输出](https://langchain-doc.cn/v1/python/langchain/models.html#结构化输出) - 模型的响应被限制为遵循定义的格式。
- [多模态](https://langchain-doc.cn/v1/python/langchain/models.html#多模态) - 处理和返回非文本数据，如图像、音频和视频。
- [推理](https://langchain-doc.cn/v1/python/langchain/models.html#推理) - 模型执行多步推理以得出结论。

模型是 [代理](https://langchain-doc.cn/v1/python/langchain/agents) 的推理引擎。它们驱动代理的决策过程，决定调用哪些工具、如何解释结果以及何时提供最终答案。

您选择的模型的质量和能力直接影响代理的可靠性和性能。不同模型在不同任务上表现出色——有些更擅长遵循复杂指令，有些更擅长结构化推理，有些支持更大的上下文窗口以处理更多信息。

LangChain 的标准模型接口为您提供对众多不同提供商集成的访问，这使得实验和切换模型以找到最适合您用例的模型变得非常容易。

> **信息**
> 有关特定提供商的集成信息和功能，请参阅提供商的[集成页面](https://langchain-doc.cn/v1/python/integrations/providers/overview)。



#### 基本用法

模型可以通过两种方式使用：

1. **与代理一起使用** - 创建[代理](https://langchain-doc.cn/v1/python/langchain/agents#model)时可动态指定模型。
2. **独立使用** - 模型可直接调用（在代理循环之外），用于文本生成、分类或提取等任务，而无需代理框架。

同一模型接口在两种上下文中均适用，这为您提供了从简单开始并根据需要扩展到更复杂基于代理的工作流程的灵活性。

##### 初始化模型

在 LangChain 中开始使用独立模型的最简单方法是使用 [`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model) 从您选择的[提供商](https://langchain-doc.cn/v1/python/integrations/providers/overview)初始化一个模型（以下示例）：

------

##### OpenAI

👉 阅读 [OpenAI 聊天模型集成文档](https://langchain-doc.cn/v1/python/integrations/chat/openai/)

```shell
pip install -U "langchain[openai]"
```

```python
import os
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "sk-..."

model = init_chat_model("openai:gpt-4.1")
```

```python
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "sk-..."

model = ChatOpenAI(model="gpt-4.1")
```

------

##### Anthropic

👉 阅读 [Anthropic 聊天模型集成文档](https://langchain-doc.cn/v1/python/integrations/chat/anthropic/)

```shell
pip install -U "langchain[anthropic]"
```

```python
import os
from langchain.chat_models import init_chat_model

os.environ["ANTHROPIC_API_KEY"] = "sk-..."

model = init_chat_model("anthropic:claude-sonnet-4-5")
```

```python
import os
from langchain_anthropic import ChatAnthropic

os.environ["ANTHROPIC_API_KEY"] = "sk-..."

model = ChatAnthropic(model="claude-sonnet-4-5")
```

------

##### Azure

👉 阅读 [Azure 聊天模型集成文档](https://langchain-doc.cn/v1/python/integrations/chat/azure_chat_openai/)

```shell
pip install -U "langchain[openai]"
```

```python
import os
from langchain.chat_models import init_chat_model

os.environ["AZURE_OPENAI_API_KEY"] = "..."
os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
os.environ["OPENAI_API_VERSION"] = "2025-03-01-preview"

model = init_chat_model(
    "azure_openai:gpt-4.1",
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)
```

```python
import os
from langchain_openai import AzureChatOpenAI

os.environ["AZURE_OPENAI_API_KEY"] = "..."
os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
os.environ["OPENAI_API_VERSION"] = "2025-03-01-preview"

model = AzureChatOpenAI(
    model="gpt-4.1",
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
)
```

------

##### Google Gemini

👉 阅读 [Google GenAI 聊天模型集成文档](https://langchain-doc.cn/v1/python/integrations/chat/google_generative_ai/)

```shell
pip install -U "langchain[google-genai]"
```

```python
import os
from langchain.chat_models import init_chat_model

os.environ["GOOGLE_API_KEY"] = "..."

model = init_chat_model("google_genai:gemini-2.5-flash-lite")
```

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "..."

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
```

------

##### AWS Bedrock

👉 阅读 [AWS Bedrock 聊天模型集成文档](https://langchain-doc.cn/v1/python/integrations/chat/bedrock/)

```shell
pip install -U "langchain[aws]"
```

```python
from langchain.chat_models import init_chat_model

# 请按照此处的步骤配置您的凭据：
# https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html

model = init_chat_model(
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_provider="bedrock_converse",
)
```

```python
from langchain_aws import ChatBedrock

model = ChatBedrock(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
```

------



```python
response = model.invoke("为什么鹦鹉会说话？")
```

有关更多详细信息，包括如何传递模型[参数](https://langchain-doc.cn/v1/python/langchain/models.html#参数)，请参阅 [`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model)。

##### 关键方法

| 方法       | 说明                                               |
| ---------- | -------------------------------------------------- |
| **Invoke** | 模型接受消息作为输入，并在生成完整响应后输出消息。 |
| **Stream** | 调用模型，但实时流式传输生成的输出。               |
| **Batch**  | 将多个请求批量发送给模型，以实现更高效的处理。     |

> **信息**
> 除了聊天模型之外，LangChain 还支持其他相关技术，如嵌入模型和向量存储。详情请参阅[集成页面](https://langchain-doc.cn/v1/python/integrations/providers/overview)。todo



#### 参数

聊天模型接受可用于配置其行为的一组参数。支持的参数集因模型和提供商而异，但标准参数包括：

| 参数          | 类型   | 必填 | 说明                                                         |
| ------------- | ------ | ---- | ------------------------------------------------------------ |
| `model`       | string | 是   | 您想使用的特定模型的名称或标识符。                           |
| `api_key`     | string | 否   | 用于向模型提供商进行身份验证的密钥。通常在注册访问模型时颁发。通常通过设置**环境变量**访问。 |
| `temperature` | number | 否   | 控制模型输出的随机性。值越高，响应越具创造性；值越低，响应越确定性。 |
| `timeout`     | number | 否   | 在取消请求之前等待模型响应的最大时间（秒）。                 |
| `max_tokens`  | number | 否   | 限制响应中的**令牌**总数，有效控制输出长度。                 |
| `max_retries` | number | 否   | 如果因网络超时或速率限制等问题而失败，系统将重新发送请求的最大尝试次数。 |

使用 [`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model)，将这些参数作为内联 `**kwargs` 传递：

```python
model = init_chat_model(
    "anthropic:claude-sonnet-4-5",
    # 传递给模型的 Kwargs：
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)
```

> **信息**
> 每个聊天模型集成可能具有用于控制提供商特定功能的额外参数。例如，[`ChatOpenAI`](https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI/) 具有 `use_responses_api` 以决定是否使用 OpenAI Responses 或 Completions API。
> 要查找给定聊天模型支持的所有参数，请转到[聊天模型集成](https://langchain-doc.cn/v1/python/integrations/chat)页面。



#### 调用

必须调用聊天模型以生成输出。主要有三种调用方法，每种方法适用于不同的用例。

##### Invoke

调用模型的最直接方法是使用 [`invoke()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.invoke) 传入单个消息或消息列表。

```python
response = model.invoke("为什么鹦鹉有五颜六色的羽毛？")
print(response)
```

可以向模型提供消息列表以表示对话历史。每条消息都有一个角色，模型用它来指示对话中谁发送了消息。有关角色、类型和内容的更多详细信息，请参阅[消息](https://langchain-doc.cn/v1/python/langchain/messages)指南。

```python
from langchain.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
    {"role": "system", "content": "你是一个将英语翻译成法语的有用助手。"},
    {"role": "user", "content": "翻译：我喜欢编程。"},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "翻译：我喜欢构建应用程序。"}
]

response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")
```



```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
    SystemMessage("你是一个将英语翻译成法语的有用助手。"),
    HumanMessage("翻译：我喜欢编程。"),
    AIMessage("J'adore la programmation。"),
    HumanMessage("翻译：我喜欢构建应用程序。")
]

response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications。")
```



##### Stream

大多数模型可以在生成时流式传输其输出内容。通过逐步显示输出，流式传输显著改善了用户体验，尤其是对于较长的响应。

调用 [`stream()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.stream) 返回一个**迭代器**，它在生成时逐块产生输出。您可以使用循环实时处理每个块：

```python
for chunk in model.stream("为什么鹦鹉有五颜六色的羽毛？"):
    print(chunk.text, end="|", flush=True)
```

```python
for chunk in model.stream("天空是什么颜色？"):
    for block in chunk.content_blocks:
        if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
            print(f"推理：{reasoning}")
        elif block["type"] == "tool_call_chunk":
            print(f"工具调用块：{block}")
        elif block["type"] == "text":
            print(block["text"])
        else:
            ...
```

与返回单个 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 的 [`invoke()`](https://langchain-doc.cn/v1/python/langchain/models.html#invoke) 不同，`stream()` 返回多个 [`AIMessageChunk`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessageChunk) 对象，每个对象包含一部分输出文本。重要的是，流中的每个块都设计为通过求和聚合成完整消息：

```python
full = None  # None | AIMessageChunk
for chunk in model.stream("天空是什么颜色？"):
    full = chunk if full is None else full + chunk
    print(full.text)

# 天空
# 天空是
# 天空通常
# 天空通常是蓝色
# ...

print(full.content_blocks)
# [{"type": "text", "text": "天空通常是蓝色..."}]
```

生成的消息可以与使用 [`invoke()`](https://langchain-doc.cn/v1/python/langchain/models.html#invoke) 生成的消息相同处理——例如，它可以聚合成消息历史并作为对话上下文传递回模型。

> **警告**
> 流式传输仅在程序的所有步骤都知道如何处理块流时才有效。例如，无法流式传输的应用程序是需要先将整个输出存储在内存中才能处理的情况。

###### 高级流式传输主题

**“自动流式传输”聊天模型**
LangChain 通过在某些情况下自动启用流式传输模式来简化从聊天模型进行流式传输，即使您未显式调用流式传输方法。这在您使用非流式传输的 invoke 方法但仍希望流式传输整个应用程序（包括聊天模型的中间结果）时特别有用。

例如，在 [LangGraph 代理](https://langchain-doc.cn/v1/python/langchain/agents) 中，您可以在节点内调用 `model.invoke()`，但如果在流式传输模式下运行，LangChain 将自动委托给流式传输。

**工作原理**
当您 `invoke()` 一个聊天模型时，如果 LangChain 检测到您正在尝试流式传输整个应用程序，它将自动切换到内部流式传输模式。对于使用 invoke 的代码，结果是相同的；然而，在聊天模型流式传输时，LangChain 将负责在 LangChain 的回调系统中调用 [`on_llm_new_token`](https://reference.langchain.com/python/langchain_core/callbacks/#langchain_core.callbacks.base.AsyncCallbackHandler.on_llm_new_token) 事件。
回调事件允许 LangGraph 的 `stream()` 和 [`astream_events()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.astream_events) 实时显示聊天模型的输出。

**流式传输事件**
LangChain 聊天模型还可以使用 [`astream_events()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.astream_events) 流式传输语义事件。
这简化了基于事件类型和其他元数据的过滤，并在后台聚合完整消息。请参阅以下示例。

```python
async for event in model.astream_events("你好"):

    if event["event"] == "on_chat_model_start":
        print(f"输入：{event['data']['input']}")

    elif event["event"] == "on_chat_model_stream":
        print(f"令牌：{event['data']['chunk'].text}")

    elif event["event"] == "on_chat_model_end":
        print(f"完整消息：{event['data']['output'].text}")

    else:
        pass
```

```txt
输入：你好
令牌：你
令牌：好
令牌：！
令牌：我
令牌：能
令牌：如
...
完整消息：你好！今天我能如何帮助您？
```

> **提示**
> 请参阅 [`astream_events()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.astream_events) 参考，了解事件类型和其他详细信息。

##### Batch

将一组独立请求批量处理给模型可以显著提高性能并降低成本，因为处理可以并行进行：

```python
responses = model.batch([
    "为什么鹦鹉有五颜六色的羽毛？",
    "飞机是如何飞行的？",
    "什么是量子计算？"
])
for response in responses:
    print(response)
```

> **注意**
> 本节描述的是聊天模型方法 [`batch()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch)，它在客户端并行化模型调用。
> 它**不同于**推理提供商支持的批量 API，例如 [OpenAI](https://platform.openai.com/docs/guides/batch) 或 [Anthropic](https://docs.claude.com/en/docs/build-with-claude/batch-processing#message-batches-api)。

默认情况下，[`batch()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch) 仅返回整个批次的最终输出。如果您希望在每个单独输入完成生成时接收输出，可以使用 [`batch_as_completed()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch_as_completed) 流式传输结果：

```python
for response in model.batch_as_completed([
    "为什么鹦鹉有五颜六色的羽毛？",
    "飞机是如何飞行的？",
    "什么是量子计算？"
]):
    print(response)
```

> **注意**
> 使用 [`batch_as_completed()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch_as_completed) 时，结果可能无序到达。每个结果包括输入索引，以便根据需要匹配和重建原始顺序。

> **提示**
> 在使用 [`batch()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch) 或 [`batch_as_completed()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch_as_completed) 处理大量输入时，您可能希望控制最大并行调用数。这可以通过在 [`RunnableConfig`](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.RunnableConfig) 字典中设置 [`max_concurrency`](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.RunnableConfig.max_concurrency) 属性来完成。

```python
model.batch(
    list_of_inputs,
    config={
        'max_concurrency': 5,  # 限制为 5 个并行调用
    }
)
```

有关批处理的更多详细信息，请参阅[参考](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch)。



#### 工具调用

模型可以请求调用执行任务的工具，例如从数据库获取数据、搜索网络或运行代码。工具是以下内容的配对：

1. 架构，包括工具的名称、描述和/或参数定义（通常是 JSON 架构）
2. 要执行的函数或**协程**

> **注意**
> 您可能会听到“函数调用”一词。我们将此与“工具调用”互换使用。

要使您定义的工具可供模型使用，您必须使用 [`bind_tools()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.bind_tools) 绑定它们。在后续调用中，模型可以根据需要选择调用任何绑定的工具。

一些模型提供商提供内置工具，可通过模型或调用参数启用（例如 [`ChatOpenAI`](https://langchain-doc.cn/v1/python/integrations/chat/openai)、[`ChatAnthropic`](https://langchain-doc.cn/v1/python/integrations/chat/anthropic)）。请查看相应的[提供商参考](https://langchain-doc.cn/v1/python/integrations/providers/overview)以了解详细信息。

> **提示**
> 有关创建工具的详细信息和其他选项，请参阅[工具指南](https://langchain-doc.cn/v1/python/langchain/tools)。

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """获取某个位置的天气。"""
    return f"{location} 天气晴朗。"

model_with_tools = model.bind_tools([get_weather])  # [!code highlight]

response = model_with_tools.invoke("波士顿的天气怎么样？")
for tool_call in response.tool_calls:
    # 查看模型发出的工具调用
    print(f"工具：{tool_call['name']}")
    print(f"参数：{tool_call['args']}")
```

当模型单独使用时，你需要自行执行模型请求的工具调用，并将工具结果返回给模型继续推理。
当使用 Agent（代理）时，Agent 的执行循环会自动完成工具调用、结果回传以及后续推理的整个过程。

下面展示了一些使用工具调用的常见方法。

##### 工具执行循环

当模型返回工具调用时，您需要执行工具并将结果传递回模型。这会创建一个对话循环，模型可以使用工具结果生成其最终响应。LangChain 包含[代理](https://langchain-doc.cn/v1/python/langchain/agents)抽象来为您处理此协调。

以下是一个简单示例：

```python
# 将（可能多个）工具绑定到模型
model_with_tools = model.bind_tools([get_weather])

# 步骤 1：模型生成工具调用
messages = [{"role": "user", "content": "波士顿的天气怎么样？"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# 步骤 2：执行工具并收集结果
for tool_call in ai_msg.tool_calls:
    # 使用生成的参数执行工具
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# 步骤 3：将结果传递回模型以获取最终响应
final_response = model_with_tools.invoke(messages)
print(final_response.text)
# "波士顿当前天气为 72°F，晴朗。"
```

每个由工具返回的 [`ToolMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ToolMessage) 包含一个与原始工具调用匹配的 `tool_call_id`，帮助模型将结果与请求相关联。

##### 强制工具调用

默认情况下，模型可以根据用户输入自由选择使用哪个绑定的工具。但是，您可能希望强制选择工具，确保模型使用特定工具或给定列表中的**任何**工具：

```python
model_with_tools = model.bind_tools([tool_1], tool_choice="any")
```

```python
model_with_tools = model.bind_tools([tool_1], tool_choice="tool_1")
```

##### 并行工具调用

许多模型支持在适当时并行调用多个工具。这允许模型同时从不同来源收集信息。

```python
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke(
    "波士顿和东京的天气怎么样？"
)

# 模型可能会生成多个工具调用
print(response.tool_calls)
# [
#   {'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': 'call_1'},
#   {'name': 'get_weather', 'args': {'location': 'Tokyo'}, 'id': 'call_2'},
# ]

# 执行所有工具（可以使用 async 并行执行）
results = []
for tool_call in response.tool_calls:
    if tool_call['name'] == 'get_weather':
        result = get_weather.invoke(tool_call)
    ...
    results.append(result)
```

模型根据请求操作的独立性智能地确定何时适合并行执行。

> **提示**
> 大多数支持工具调用的模型默认启用并行工具调用。某些模型（包括 [OpenAI](https://langchain-doc.cn/v1/python/integrations/chat/openai) 和 [Anthropic](https://langchain-doc.cn/v1/python/integrations/chat/anthropic)）允许您禁用此功能。要执行此操作，请设置 `parallel_tool_calls=False`：

```python
model.bind_tools([get_weather], parallel_tool_calls=False)
```

##### 流式传输工具调用

在流式传输响应时，工具调用通过 [`ToolCallChunk`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ToolCallChunk) 逐步构建。这允许您在生成工具调用时查看它们，而不是等待完整响应。

```python
for chunk in model_with_tools.stream(
    "波士顿和东京的天气怎么样？"
):
    # 工具调用块逐步到达
    for tool_chunk in chunk.tool_call_chunks:
        if name := tool_chunk.get("name"):
            print(f"工具：{name}")
        if id_ := tool_chunk.get("id"):
            print(f"ID：{id_}")
        if args := tool_chunk.get("args"):
            print(f"参数：{args}")

# 输出：
# 工具：get_weather
# ID：call_SvMlU1TVIZugrFLckFE2ceRE
# 参数：{"lo
# 参数：catio
# 参数：n": "B
# 参数：osto
# 参数：n"}
# 工具：get_weather
# ID：call_QMZdy6qInx13oWKE7KhuhOLR
# 参数：{"lo
# 参数：catio
# 参数：n": "T
# 参数：okyo
# 参数："}
```

您可以累积块以构建完整的工具调用：

```python
gathered = None
for chunk in model_with_tools.stream("波士顿的天气怎么样？"):
    gathered = chunk if gathered is None else gathered + chunk
    print(gathered.tool_calls)
```



#### 结构化输出

可以请求模型以匹配给定架构的格式提供其响应。这对于确保输出易于解析并用于后续处理非常有用。LangChain 支持多种架构类型和强制执行结构化输出的方法。

##### Pydantic

[Pydantic 模型](https://docs.pydantic.dev/latest/concepts/models/#basic-model-usage) 提供最丰富的功能集，包括字段验证、描述和嵌套结构。

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """一部带有详细信息的电影。"""
    title: str = Field(..., description="电影标题")
    year: int = Field(..., description="电影上映年份")
    director: str = Field(..., description="电影导演")
    rating: float = Field(..., description="电影评分，满分 10 分")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("提供关于电影《盗梦空间》的详细信息")
print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)
```

##### TypedDict

`TypedDict` 提供使用 Python 内置类型的更简单替代方案，适用于不需要运行时验证的情况。

```python
from typing_extensions import TypedDict, Annotated

class MovieDict(TypedDict):
    """一部带有详细信息的电影。"""
    title: Annotated[str, ..., "电影标题"]
    year: Annotated[int, ..., "电影上映年份"]
    director: Annotated[str, ..., "电影导演"]
    rating: Annotated[float, ..., "电影评分，满分 10 分"]

model_with_structure = model.with_structured_output(MovieDict)
response = model_with_structure.invoke("提供关于电影《盗梦空间》的详细信息")
print(response)  # {'title': 'Inception', 'year': 2010, 'director': 'Christopher Nolan', 'rating': 8.8}
```

##### JSON Schema

为了获得最大控制或互操作性，您可以提供原始 JSON 架构。

```python
import json

json_schema = {
    "title": "Movie",
    "description": "一部带有详细信息的电影",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "电影标题"
        },
        "year": {
            "type": "integer",
            "description": "电影上映年份"
        },
        "director": {
            "type": "string",
            "description": "电影导演"
        },
        "rating": {
            "type": "number",
            "description": "电影评分，满分 10 分"
        }
    },
    "required": ["title", "year", "director", "rating"]
}

model_with_structure = model.with_structured_output(
    json_schema,
    method="json_schema",
)
response = model_with_structure.invoke("提供关于电影《盗梦空间》的详细信息")
print(response)  # {'title': 'Inception', 'year': 2010, ...}
```

> **注意**
> **结构化输出的关键考虑因素：**
>
> - 方法参数：某些提供商支持不同的方法（`'json_schema'`、`'function_calling'`、`'json_mode'`）
>   - `'json_schema'` 通常指提供商提供的专用结构化输出功能
>   - `'function_calling'` 通过强制[工具调用](https://langchain-doc.cn/v1/python/langchain/models.html#工具调用)遵循给定架构来派生结构化输出
>   - `'json_mode'` 是某些提供商提供的 `'json_schema'` 的前身——它生成有效的 JSON，但架构必须在提示中描述
> - **包含原始**：使用 `include_raw=True` 以同时获取已解析的输出和原始 AI 消息
> - **验证**：Pydantic 模型提供自动验证，而 `TypedDict` 和 JSON Schema 需要手动验证

###### 示例：消息输出与解析结构并存

返回原始 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 对象与解析表示一起以访问响应元数据（如[令牌计数](https://langchain-doc.cn/v1/python/langchain/models.html#令牌使用情况)）可能很有用。为此，在调用 [`with_structured_output`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.with_structured_output) 时设置 [`include_raw=True`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.with_structured_output(include_raw))：

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """一部带有详细信息的电影。"""
    title: str = Field(..., description="电影标题")
    year: int = Field(..., description="电影上映年份")
    director: str = Field(..., description="电影导演")
    rating: float = Field(..., description="电影评分，满分 10 分")

model_with_structure = model.with_structured_output(Movie, include_raw=True)  # [!code highlight]
response = model_with_structure.invoke("提供关于电影《盗梦空间》的详细信息")
response
# {
#     "raw": AIMessage(...),
#     "parsed": Movie(title=..., year=..., ...),
#     "parsing_error": None,
# }
```

###### 示例：嵌套结构

架构可以嵌套：

```python
from pydantic import BaseModel, Field

class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="预算（百万美元）")

model_with_structure = model.with_structured_output(MovieDetails)
```

```python
from typing_extensions import Annotated, TypedDict

class Actor(TypedDict):
    name: str
    role: str

class MovieDetails(TypedDict):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: Annotated[float | None, ..., "预算（百万美元）"]

model_with_structure = model.with_structured_output(MovieDetails)
```



#### 高级主题

##### 多模态

支持多模态的 LangChain 模型可以接收图片、音频和视频等内容。你可以通过内容块（Content Blocks）组织这些数据。

>对于多模态输入，LangChain 支持三种格式：
>
>1. LangChain 统一格式（推荐，跨模型通用）
>2. OpenAI Chat Completions 原生格式
>3. 模型提供商自己的原生格式（如 Anthropic 格式）

有关详细信息，请参阅消息指南的[多模态部分](https://langchain-doc.cn/v1/python/langchain/messages#multimodal)。

> 并不是所有 LLM 都具备相同的能力。
> 某些模型除了返回文本外，还能够在响应中返回图片、音频等多模态数据。
> 当你要求这类模型生成多模态内容时，返回的 AIMessage 中会包含对应的多模态内容块（Content Blocks）。

```python
response = model.invoke("创建一张猫的图片")
print(response.content_blocks)
# [
#     {"type": "text", "text": "这是一张猫的图片"},
#     {"type": "image", "base64": "...", "mime_type": "image/jpeg"},
# ]
```

有关特定提供商的详细信息，请参阅[集成页面](https://langchain-doc.cn/v1/python/integrations/providers/overview)。

##### 推理

较新的模型能够执行多步推理以得出结论。这涉及将复杂问题分解为更小、更易管理的步骤。

**如果底层模型支持**，您可以显示此推理过程以更好地理解模型如何得出其最终答案。

```python
for chunk in model.stream("为什么鹦鹉有五颜六色的羽毛？"):
    reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    print(reasoning_steps if reasoning_steps else chunk.text)
```

```python
response = model.invoke("为什么鹦鹉有五颜六色的羽毛？")
reasoning_steps = [b for b in response.content_blocks if b["type"] == "reasoning"]
print(" ".join(step["reasoning"] for step in reasoning_steps))
```

根据模型，您有时可以指定它应投入推理的努力程度。同样，您可以请求模型完全关闭推理。这可能采用“层级”（例如 `'low'` 或 `'high'`）或整数令牌预算的形式。

有关详细信息，请参阅[集成页面](https://langchain-doc.cn/v1/python/integrations/providers/overview)或您相应聊天模型的[参考](https://reference.langchain.com/python/integrations/)。

##### 本地模型

LangChain 支持在您自己的硬件上本地运行模型。这对于数据隐私至关重要、您想调用自定义模型或希望避免使用基于云的模型所产生的成本的场景非常有用。

[Ollama](https://langchain-doc.cn/v1/python/integrations/chat/ollama) 是本地运行模型的最简单方法之一。请参阅[集成页面](https://langchain-doc.cn/v1/python/integrations/providers/overview)上的本地集成完整列表。

##### 提示缓存

许多提供商提供提示缓存功能，以减少对相同令牌重复处理的延迟和成本。这些功能可以是**隐式**或**显式**：

- **隐式提示缓存：** 如果请求命中缓存，提供商将自动传递成本节省。示例：[OpenAI](https://langchain-doc.cn/v1/python/integrations/chat/openai) 和 [Gemini](https://langchain-doc.cn/v1/python/integrations/chat/google_generative_ai)（Gemini 2.5 及以上）。
- **显式缓存：** 提供商允许您手动指示缓存点以获得更大控制或保证成本节省。示例：[`ChatOpenAI`](https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI/)（通过 `prompt_cache_key`）、Anthropic 的 [`AnthropicPromptCachingMiddleware`](https://langchain-doc.cn/v1/python/integrations/chat/anthropic#prompt-caching) 和 [`cache_control`](https://docs.langchain.com/v1/python/integrations/chat/anthropic#prompt-caching) 选项、[AWS Bedrock](https://langchain-doc.cn/v1/python/integrations/chat/bedrock#prompt-caching)、[Gemini](https://python.langchain.com/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html)。

> **警告**
> 提示缓存通常仅在超过最小输入令牌阈值时才会启用。请参阅[提供商页面](https://langchain-doc.cn/v1/python/integrations/chat)以了解详细信息。

缓存使用情况将反映在模型响应的[使用元数据](https://langchain-doc.cn/v1/python/langchain/messages#token-usage)中。

##### 服务端工具调用

某些模型提供商支持**“服务端工具调用”（server-side tool calling）**。

在这种模式下，模型可以在一次请求中自动调用工具（例如网页搜索、代码解释器等），并在服务端完成工具执行和结果分析。

如果模型在服务端执行了工具调用，那么返回的响应内容中会包含工具调用记录和工具执行结果。

这些内容会以统一的跨提供商格式体现在 `content_blocks` 中，例如：

- server_tool_call（发起工具调用）
- server_tool_result（工具返回结果）
- text（最终回答）

由于工具调用是在服务端完成的，因此**客户端不需要再手动构造 ToolMessage 并回传给模型**。

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4.1-mini")

tool = {"type": "web_search"}
model_with_tools = model.bind_tools([tool])

response = model_with_tools.invoke("今天有什么正面新闻？")
response.content_blocks
```

```python
[
    {
        "type": "server_tool_call",
        "name": "web_search",
        "args": {
            "query": "positive news stories today",
            "type": "search"
        },
        "id": "ws_abc123"
    },
    {
        "type": "server_tool_result",
        "tool_call_id": "ws_abc123",
        "status": "success"
    },
    {
        "type": "text",
        "text": "以下是今天的一些正面新闻...",
        "annotations": [
            {
                "end_index": 410,
                "start_index": 337,
                "title": "文章标题",
                "type": "citation",
                "url": "..."
            }
        ]
    }
]
```

有关可用工具和使用详细信息的给定提供商，请参阅[集成页面](https://langchain-doc.cn/v1/python/integrations/chat)。

##### 速率限制

许多聊天模型提供商对给定时间段内可以发出的调用次数施加限制。如果您达到速率限制，通常会从提供商收到速率限制错误响应，并且需要等待才能发出更多请求。

为了帮助管理速率限制，聊天模型集成在初始化期间接受 `rate_limiter` 参数，以控制发出请求的速率。

###### 初始化和使用速率限制器

LangChain 内置（可选）[`InMemoryRateLimiter`](https://reference.langchain.com/python/langchain_core/rate_limiters/#langchain_core.rate_limiters.InMemoryRateLimiter)。此限制器是线程安全的，可以由同一进程中的多个线程共享。

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 每 10 秒 1 个请求
    check_every_n_seconds=0.1,  # 每 100 毫秒检查是否允许发出请求
    max_bucket_size=10,  # 控制最大突发大小。
)

model = init_chat_model(
    model="gpt-5",
    model_provider="openai",
    rate_limiter=rate_limiter  # [!code highlight]
)
```

> **警告**
> 提供的速率限制器只能限制每单位时间的请求数。如果您还需要根据请求大小进行限制，它将无济于事。



##### 基础 URL 或代理

对于许多聊天模型集成，您可以配置 API 请求的基础 URL，这允许您使用具有 OpenAI 兼容 API 的模型提供商或使用代理服务器。

###### 基础 URL

许多模型提供商提供 OpenAI 兼容的 API（例如，[Together AI](https://www.together.ai/)、[vLLM](https://github.com/vllm-project/vllm)）。您可以通过指定适当的 `base_url` 参数使用 [`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model) 与这些提供商一起使用：

```python
model = init_chat_model(
    model="MODEL_NAME",
    model_provider="openai",
    base_url="BASE_URL",
    api_key="YOUR_API_KEY",
)
```

> **注意**
> 使用直接聊天模型类实例化时，参数名称可能因提供商而异。请查看相应的[参考](https://langchain-doc.cn/v1/python/integrations/providers/overview)以了解详细信息。

###### 代理配置

对于需要 HTTP 代理的部署，某些模型集成支持代理配置：

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o",
    openai_proxy="http://proxy.example.com:8080"
)
```

> **注意**
> 代理支持因集成而异。请查看特定模型提供商的[参考](https://langchain-doc.cn/v1/python/integrations/providers/overview)以了解代理配置选项。

##### 对数概率

某些模型可以通过在初始化模型时设置 `logprobs` 参数来配置为返回表示给定令牌可能性的令牌级对数概率：

```python
model = init_chat_model(
    model="gpt-4o",
    model_provider="openai"
).bind(logprobs=True)

response = model.invoke("为什么鹦鹉会说话？")
print(response.response_metadata["logprobs"])
```

##### 令牌使用情况

许多模型提供商将令牌使用信息作为调用响应的一部分返回。当可用时，此信息将包含在相应模型生成的 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 对象上。有关更多详细信息，请参阅[消息](https://langchain-doc.cn/v1/python/langchain/messages)指南。

> **注意**
> 某些提供商 API，特别是在流式传输上下文中 OpenAI 和 Azure OpenAI 聊天完成，要求用户选择接收令牌使用数据。请参阅集成指南的[流式传输使用元数据](https://langchain-doc.cn/v1/python/integrations/chat/openai#streaming-usage-metadata)部分以了解详细信息。

您可以使用回调或上下文管理器跟踪应用程序中跨模型的聚合令牌计数，如下所示：

###### 回调处理程序

```python
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

model_1 = init_chat_model(model="openai:gpt-4o-mini")
model_2 = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

callback = UsageMetadataCallbackHandler()
result_1 = model_1.invoke("你好", config={"callbacks": [callback]})
result_2 = model_2.invoke("你好", config={"callbacks": [callback]})
callback.usage_metadata
```

```python
{
    'gpt-4o-mini-2024-07-18': {
        'input_tokens': 8,
        'output_tokens': 10,
        'total_tokens': 18,
        'input_token_details': {'audio': 0, 'cache_read': 0},
        'output_token_details': {'audio': 0, 'reasoning': 0}
    },
    'claude-3-5-haiku-20241022': {
        'input_tokens': 8,
        'output_tokens': 21,
        'total_tokens': 29,
        'input_token_details': {'cache_read': 0, 'cache_creation': 0}
    }
}
```

###### 上下文管理器

```python
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback

model_1 = init_chat_model(model="openai:gpt-4o-mini")
model_2 = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

with get_usage_metadata_callback() as cb:
    model_1.invoke("你好")
    model_2.invoke("你好")
    print(cb.usage_metadata)
```

```python
{
    'gpt-4o-mini-2024-07-18': {
        'input_tokens': 8,
        'output_tokens': 10,
        'total_tokens': 18,
        'input_token_details': {'audio': 0, 'cache_read': 0},
        'output_token_details': {'audio': 0, 'reasoning': 0}
    },
    'claude-3-5-haiku-20241022': {
        'input_tokens': 8,
        'output_tokens': 21,
        'total_tokens': 29,
        'input_token_details': {'cache_read': 0, 'cache_creation': 0}
    }
}
```

##### 调用配置

调用模型时，您可以使用 [`RunnableConfig`](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.RunnableConfig) 字典通过 `config` 参数传递额外配置。这提供了对执行行为、回调和元数据跟踪的运行时控制。

常见配置选项包括：

```python
response = model.invoke(
    "讲个笑话",
    config={
        "run_name": "joke_generation",      # 此运行的自定义名称
        "tags": ["humor", "demo"],          # 用于分类的标签
        "metadata": {"user_id": "123"},     # 自定义元数据
        "callbacks": [my_callback_handler], # 回调处理程序
    }
)
```

这些配置值在以下情况下特别有用：

- 使用 [LangSmith](https://docs.smith.langchain.com/) 跟踪进行调试
- 实现自定义日志或监控
- 在生产中控制资源使用
- 在复杂管道中跟踪调用

###### 关键配置属性

| 属性              | 类型     | 说明                                                         |
| ----------------- | -------- | ------------------------------------------------------------ |
| `run_name`        | string   | 在日志和跟踪中标识此特定调用。不由子调用继承。               |
| `tags`            | string[] | 由所有子调用继承的标签，用于在调试工具中过滤和组织。         |
| `metadata`        | object   | 用于跟踪额外上下文的自定义键值对，由所有子调用继承。         |
| `max_concurrency` | number   | 在使用 [`batch()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch) 或 [`batch_as_completed()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch_as_completed) 时控制最大并行调用数。 |
| `callbacks`       | array    | 用于在执行期间监控和响应事件的处理程序。                     |
| `recursion_limit` | number   | 链的最大递归深度，以防止复杂管道中的无限循环。               |

> **提示**
> 请参阅完整的 [`RunnableConfig`](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.RunnableConfig) 参考，了解所有支持的属性。

##### 可配置模型

您还可以通过指定 [`configurable_fields`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.configurable_fields) 创建运行时可配置模型。如果您未指定模型值，则默认情况下 `'model'` 和 `'model_provider'` 将是可配置的。

```python
from langchain.chat_models import init_chat_model

configurable_model = init_chat_model(temperature=0)

configurable_model.invoke(
    "你叫什么名字",
    config={"configurable": {"model": "gpt-5-nano"}},  # 使用 GPT-5-Nano 运行
)
configurable_model.invoke(
    "你叫什么名字",
    config={"configurable": {"model": "claude-sonnet-4-5"}},  # 使用 Claude 运行
)
```

###### 具有默认值的可配置模型

我们可以创建具有默认模型值的可配置模型，指定哪些参数是可配置的，并为可配置参数添加前缀：

```python
first_model = init_chat_model(
        model="gpt-4.1-mini",
        temperature=0,
        configurable_fields=("model", "model_provider", "temperature", "max_tokens"),
        config_prefix="first",  # 当链中有多个模型时很有用
)

first_model.invoke("你叫什么名字")
```



```python
first_model.invoke(
    "你叫什么名字",
    config={
        "configurable": {
            "first_model": "claude-sonnet-4-5",
            "first_temperature": 0.5,
            "first_max_tokens": 100,
        }
    },
)
```

###### 以声明方式使用可配置模型

我们可以在可配置模型上调用声明性操作，如 `bind_tools`、`with_structured_output`、`with_configurable` 等，并以与常规实例化聊天模型对象相同的方式链接可配置模型。

```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    """获取给定位置的当前天气"""
    location: str = Field(..., description="城市和州，例如 San Francisco, CA")

class GetPopulation(BaseModel):
    """获取给定位置的当前人口"""
    location: str = Field(..., description="城市和州，例如 San Francisco, CA")

model = init_chat_model(temperature=0)
model_with_tools = model.bind_tools([GetWeather, GetPopulation])

model_with_tools.invoke(
    "2024 年洛杉矶和纽约哪个更大", config={"configurable": {"model": "gpt-4.1-mini"}}
).tool_calls
```

```
[
    {
        'name': 'GetPopulation',
        'args': {'location': 'Los Angeles, CA'},
        'id': 'call_Ga9m8FAArIyEjItHmztPYA22',
        'type': 'tool_call'
    },
    {
        'name': 'GetPopulation',
        'args': {'location': 'New York, NY'},
        'id': 'call_jh2dEvBaAHRaw5JUDthOs7rt',
        'type': 'tool_call'
    }
]
```

```python
model_with_tools.invoke(
    "2024 年洛杉矶和纽约哪个更大",
        config={"configurable": {"model": "claude-sonnet-4-5"}},
).tool_calls
```

```
[
    {
        'name': 'GetPopulation',
        'args': {'location': 'Los Angeles, CA'},
        'id': 'toolu_01JMufPf4F4t2zLj7miFeqXp',
        'type': 'tool_call'
    },
    {
        'name': 'GetPopulation',
        'args': {'location': 'New York City, NY'},
        'id': 'toolu_01RQBHcE8kEEbYTuuS8WqY1u',
        'type': 'tool_call'
    }
]
```



### 消息

消息是 LangChain 中模型上下文的基本单位。它们代表模型的输入和输出，携带内容和元数据，用于在与 LLM 交互时表示对话状态。

消息是包含以下内容的对象：

- **角色** - 标识消息类型（例如 `system`、`user`）
- **内容** - 表示消息的实际内容（例如文本、图像、音频、文档等）
- **元数据** - 可选字段，例如响应信息、消息 ID 和令牌使用情况

LangChain 提供了一种标准消息类型，可在所有模型提供商之间工作，确保无论调用哪个模型都能保持一致的行为。

#### 基本用法

使用消息的最简单方式是创建消息对象，并在[调用](https://langchain-doc.cn/v1/python/langchain/models#invocation)时将它们传递给模型。

```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("openai:gpt-5-nano")

system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")

# 与聊天模型一起使用
messages = [system_msg, human_msg]
response = model.invoke(messages)  # 返回 AIMessage
```

##### 文本提示

文本提示是字符串 - 适用于不需要保留对话历史的简单生成任务。

```python
response = model.invoke("Write a haiku about spring")
```

**何时使用文本提示：**

- 只有一个独立的请求
- 不需要对话历史
- 希望代码复杂度最小

##### 消息提示

或者，您可以通过提供消息对象列表将消息列表传递给模型。

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a haiku about spring"),
    AIMessage("Cherry blossoms bloom...")
]
response = model.invoke(messages)
```

**何时使用消息提示：**

- 管理多轮对话
- 处理多模态内容（图像、音频、文件）
- 包含系统指令

##### 字典格式

您还可以直接以 OpenAI 聊天补全格式指定消息。

```python
messages = [
    {"role": "system", "content": "You are a poetry expert"},
    {"role": "user", "content": "Write a haiku about spring"},
    {"role": "assistant", "content": "Cherry blossoms bloom..."}
]
response = model.invoke(messages)
```



#### 消息类型

- [系统消息](https://langchain-doc.cn/v1/python/langchain/messages.html#系统消息) - 告诉模型如何行为并为交互提供上下文
- [人类消息](https://langchain-doc.cn/v1/python/langchain/messages.html#人类消息) - 表示用户输入和与模型的交互
- [AI 消息](https://langchain-doc.cn/v1/python/langchain/messages.html#ai-消息) - 模型生成的响应，包括文本内容、工具调用和元数据
- [工具消息](https://langchain-doc.cn/v1/python/langchain/messages.html#工具消息) - 表示[工具调用](https://langchain-doc.cn/v1/python/langchain/models#tool-calling)的输出



##### 系统消息

[`SystemMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.SystemMessage) 表示一组初始指令，用于引导模型的行为。您可以使用系统消息来设置语气、定义模型角色并建立响应指南。

```python
system_msg = SystemMessage("You are a helpful coding assistant.")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
```

```python
from langchain.messages import SystemMessage, HumanMessage

system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
```

------

##### 人类消息

[`HumanMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.HumanMessage) 表示用户输入和交互。它们可以包含文本、图像、音频、文件以及任何其他多模态[内容](https://langchain-doc.cn/v1/python/langchain/messages.html#消息内容)。

###### 文本内容

```python
response = model.invoke([
    HumanMessage("What is machine learning?")
])
```

```python
# 使用字符串是单个 HumanMessage 的快捷方式
response = model.invoke("What is machine learning?")
```

###### 消息元数据

```python
human_msg = HumanMessage(
    content="Hello!",
    name="alice",  # 可选：标识不同用户
    id="msg_123",  # 可选：用于追踪的唯一标识符
)
```

> **注意**
> `name` 字段的行为因提供商而异 - 有些用于用户识别，其他忽略它。要检查，请参考模型提供商的[参考文档](https://reference.langchain.com/python/integrations/)。

##### AI 消息

[`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 表示模型调用的输出。它们可以包含多模态数据、工具调用和提供商特定的元数据，您可以稍后访问。

```python
response = model.invoke("Explain AI")
print(type(response))  # <class 'langchain_core.messages.AIMessage'>
```

[`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 对象由调用模型时返回，其中包含响应中的所有关联元数据。

提供商对不同类型的消息的权重/上下文处理不同，这意味着有时手动创建新的 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 对象并将其插入消息历史中就像来自模型一样很有帮助。

```python
from langchain.messages import AIMessage, SystemMessage, HumanMessage

# 手动创建 AI 消息（例如，用于对话历史）
ai_msg = AIMessage("I'd be happy to help you with that question!")

# 添加到对话历史
messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can you help me?"),
    ai_msg,  # 插入就像来自模型一样
    HumanMessage("Great! What's 2+2?")
]

response = model.invoke(messages)
```

属性

- **text** (`string`)
  消息的文本内容。
- **content** (`string | dict[]`)
  消息的原始内容。
- **content_blocks** (`ContentBlock[]`)
  消息的标准化[内容块](https://langchain-doc.cn/v1/python/langchain/messages.html#消息内容)。
- **tool_calls** (`dict[] | None`)
  模型进行的工具调用。如果没有调用工具，则为空。
- **id** (`string`)
  消息的唯一标识符（由 LangChain 自动生成或在提供商响应中返回）
- **usage_metadata** (`dict | None`)
  消息的使用元数据，可包含可用时的令牌计数。
- **response_metadata** (`ResponseMetadata | None`)
  消息的响应元数据。

![image-20260613085842311](LangChain 教程.assets/image-20260613085842311.png)



###### 工具调用

当模型进行[工具调用](https://langchain-doc.cn/v1/python/langchain/models#tool-calling)时，它们包含在 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 中：

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-5-nano")

def get_weather(location: str) -> str:
    """Get the weather at a location."""
    ...

model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather in Paris?")

for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    print(f"ID: {tool_call['id']}")
```

其他结构化数据（如推理或引用）也可以出现在消息[内容](https://langchain-doc.cn/v1/python/langchain/messages#消息内容)中。
![image-20260613091135041](LangChain 教程.assets/image-20260613091135041.png)
![image-20260613091015225](LangChain 教程.assets/image-20260613091015225.png)



###### 令牌使用

[`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 可以在其 [`usage_metadata`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage.usage_metadata) 字段中保存令牌计数和其他使用元数据：

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-5-nano")

response = model.invoke("Hello!")
response.usage_metadata
```

```
{'input_tokens': 8,
 'output_tokens': 304,
 'total_tokens': 312,
 'input_token_details': {'audio': 0, 'cache_read': 0},
 'output_token_details': {'audio': 0, 'reasoning': 256}}
```

有关详细信息，请参阅 [`UsageMetadata`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage.usage_metadata)。
![image-20260613091340684](LangChain 教程.assets/image-20260613091340684.png)

###### 流式传输和块

在流式传输期间，您将收到可以组合成完整消息对象的 [`AIMessageChunk`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessageChunk) 对象：

```python
chunks = []
full_message = None
for chunk in model.stream("Hi"):
    chunks.append(chunk)
    print(chunk.text)
    full_message = chunk if full_message is None else full_message + chunk
```

> **了解更多：**
>
> - [从聊天模型流式传输令牌](https://langchain-doc.cn/v1/python/langchain/models#stream)
> - [从代理流式传输令牌和/或步骤](https://langchain-doc.cn/v1/python/langchain/streaming)

![image-20260613091856135](LangChain 教程.assets/image-20260613091856135.png)

##### 工具消息

对于支持[工具调用](https://langchain-doc.cn/v1/python/langchain/models#tool-calling)的模型，AI 消息可以包含工具调用。工具消息用于将单个工具执行的结果传回模型。

[工具](https://langchain-doc.cn/v1/python/langchain/tools) 可以直接生成 [`ToolMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ToolMessage) 对象。下面展示一个简单示例。有关更多信息，请阅读[工具指南](https://langchain-doc.cn/v1/python/langchain/tools)。

```python
# 模型进行工具调用后
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

# 执行工具并创建结果消息
weather_result = "Sunny, 72°F"
tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # 必须匹配调用 ID
)

# 继续对话
messages = [
    HumanMessage("What's the weather in San Francisco?"),
    ai_message,  # 模型的工具调用
    tool_message,  # 工具执行结果
]
response = model.invoke(messages)  # 模型处理结果
```

属性

- **content** (`string`, 必需)
  工具调用的字符串化输出。
- **tool_call_id** (`string`, 必需)
  此消息响应的工具调用的 ID。（必须匹配 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 中的工具调用 ID）
- **name** (`string`, 必需)
  被调用的工具的名称。
- **artifact** (`dict`)
  不发送给模型但可以以编程方式访问的附加数据。

> **注意**
> `artifact` 字段存储不会发送给模型但可以以编程方式访问的补充数据。这对于存储原始结果、调试信息或下游处理的数据而不会使模型上下文杂乱很有用。
>
> <details open=""><summary>示例：使用 artifact 存储检索元数据</summary><p style="line-height: 1.6; overflow-wrap: break-word;">例如，<a href="https://langchain-doc.cn/v1/python/langchain/retrieval" style="color: rgb(47, 132, 93); font-weight: 500; text-decoration: none; overflow-wrap: break-word;">检索</a>工具可以从文档中检索一段文本供模型参考。其中消息<span>&nbsp;</span><code style="font-family: ui-monospace, Menlo, Monaco, Consolas, &quot;Liberation Mono&quot;, &quot;Courier New&quot;, monospace; margin: 0px; padding: 3px 6px; border-radius: 4px; background: none 0% 0% / auto repeat scroll padding-box border-box rgba(142, 150, 170, 0.14); font-size: 0.875em; overflow-wrap: break-word; transition: background-color 0.3s, color 0.3s;">content</code><span>&nbsp;</span>包含模型将参考的文本，而<span>&nbsp;</span><code style="font-family: ui-monospace, Menlo, Monaco, Consolas, &quot;Liberation Mono&quot;, &quot;Courier New&quot;, monospace; margin: 0px; padding: 3px 6px; border-radius: 4px; background: none 0% 0% / auto repeat scroll padding-box border-box rgba(142, 150, 170, 0.14); font-size: 0.875em; overflow-wrap: break-word; transition: background-color 0.3s, color 0.3s;">artifact</code><span>&nbsp;</span>可以包含应用程序可以使用的文档标识符或其他元数据（例如，用于渲染页面）。请参阅下面的示例：</p><div class="language-python line-numbers-mode" data-highlighter="shiki" data-ext="python" style="position: relative; margin-block: 0.75rem; border-radius: 6px; background-color: rgb(236, 244, 250); transition: background-color 0.3s, color 0.3s; --shiki-light: #383A42; --shiki-dark: #abb2bf; --shiki-light-bg: #FAFAFA; --shiki-dark-bg: #282c34;"><button type="button" class="vp-copy-code-button" aria-label="复制代码" data-copied="已复制" style="position: absolute; top: 0.5em; right: 0.5em; z-index: 5; width: 2.5rem; height: 2.5rem; padding: 0px; border-width: 0px; border-radius: 0.5rem; background: rgba(0, 0, 0, 0); outline: none; opacity: 1; cursor: pointer; transition: opacity 0.4s;"></button><pre class="shiki shiki-themes one-light one-dark-pro vp-code" copy-code="" style="text-align: left; direction: ltr; white-space: pre; word-spacing: normal; word-break: normal; overflow-wrap: unset; tab-size: 4; hyphens: none; position: relative; z-index: 1; overflow-x: auto; margin: 0px 0px 0px 48px; border-radius: 6px; font-size: 14px; font-family: consolas, monaco, &quot;Andale Mono&quot;, &quot;Ubuntu Mono&quot;, monospace; line-height: 1.6; vertical-align: middle;"><code class="language-python" style="font-family: ui-monospace, Menlo, Monaco, Consolas, &quot;Liberation Mono&quot;, &quot;Courier New&quot;, monospace; padding: 16px 20px 16px 1rem; border-radius: 0px; display: block; box-sizing: border-box; width: fit-content; min-width: 100%; color: rgb(56, 58, 66); overflow-wrap: unset; -webkit-font-smoothing: auto; background-color: rgba(0, 0, 0, 0) !important;"><span class="line" style="color: rgb(56, 58, 66);"><span style="color: rgb(166, 38, 164); --shiki-light: #A626A4; --shiki-dark: #C678DD;">from</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;"> langchain.messages </span><span style="color: rgb(166, 38, 164); --shiki-light: #A626A4; --shiki-dark: #C678DD;">import</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;"> ToolMessage</span></span>
> <span class="line" style="color: rgb(56, 58, 66);"></span>
> <span class="line" style="color: rgb(56, 58, 66);"><span style="color: rgb(160, 161, 167); --shiki-light: #A0A1A7; --shiki-light-font-style: italic; --shiki-dark: #7F848E; --shiki-dark-font-style: italic;"># 发送给模型</span></span>
> <span class="line" style="color: rgb(56, 58, 66);"><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;">message_content </span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #56B6C2;">=</span><span style="color: rgb(80, 161, 79); --shiki-light: #50A14F; --shiki-dark: #98C379;"> "It was the best of times, it was the worst of times."</span></span>
> <span class="line" style="color: rgb(56, 58, 66);"></span>
> <span class="line" style="color: rgb(56, 58, 66);"><span style="color: rgb(160, 161, 167); --shiki-light: #A0A1A7; --shiki-light-font-style: italic; --shiki-dark: #7F848E; --shiki-dark-font-style: italic;"># 下游可用的 artifact</span></span>
> <span class="line" style="color: rgb(56, 58, 66);"><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;">artifact </span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #56B6C2;">=</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;"> {</span><span style="color: rgb(80, 161, 79); --shiki-light: #50A14F; --shiki-dark: #98C379;">"document_id"</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;">: </span><span style="color: rgb(80, 161, 79); --shiki-light: #50A14F; --shiki-dark: #98C379;">"doc_123"</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;">, </span><span style="color: rgb(80, 161, 79); --shiki-light: #50A14F; --shiki-dark: #98C379;">"page"</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;">: </span><span style="color: rgb(152, 104, 1); --shiki-light: #986801; --shiki-dark: #D19A66;">0</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;">}</span></span>
> <span class="line" style="color: rgb(56, 58, 66);"></span>
> <span class="line" style="color: rgb(56, 58, 66);"><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;">tool_message </span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #56B6C2;">=</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #61AFEF;"> ToolMessage</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;">(</span></span>
> <span class="line" style="color: rgb(56, 58, 66);"><span style="color: rgb(152, 104, 1); --shiki-light: #986801; --shiki-light-font-style: inherit; --shiki-dark: #E06C75; --shiki-dark-font-style: italic;">    content</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #56B6C2;">=</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;">message_content,</span></span>
> <span class="line" style="color: rgb(56, 58, 66);"><span style="color: rgb(152, 104, 1); --shiki-light: #986801; --shiki-light-font-style: inherit; --shiki-dark: #E06C75; --shiki-dark-font-style: italic;">    tool_call_id</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #56B6C2;">=</span><span style="color: rgb(80, 161, 79); --shiki-light: #50A14F; --shiki-dark: #98C379;">"call_123"</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;">,</span></span>
> <span class="line" style="color: rgb(56, 58, 66);"><span style="color: rgb(152, 104, 1); --shiki-light: #986801; --shiki-light-font-style: inherit; --shiki-dark: #E06C75; --shiki-dark-font-style: italic;">    name</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #56B6C2;">=</span><span style="color: rgb(80, 161, 79); --shiki-light: #50A14F; --shiki-dark: #98C379;">"search_books"</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;">,</span></span>
> <span class="line" style="color: rgb(56, 58, 66);"><span style="color: rgb(152, 104, 1); --shiki-light: #986801; --shiki-light-font-style: inherit; --shiki-dark: #E06C75; --shiki-dark-font-style: italic;">    artifact</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #56B6C2;">=</span><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;">artifact,</span></span>
> <span class="line" style="color: rgb(56, 58, 66);"><span style="color: rgb(56, 58, 66); --shiki-light: #383A42; --shiki-dark: #ABB2BF;">)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset: line-number 0; position: absolute; top: 0px; left: 0px; width: 48px; padding-top: 16px; color: rgba(56, 58, 66, 0.67); font-size: 14px; line-height: 1.6; text-align: center;"></div></div><p style="line-height: 1.6; overflow-wrap: break-word;">请参阅<span>&nbsp;</span><a href="https://langchain-doc.cn/v1/python/langchain/rag" style="color: rgb(47, 132, 93); font-weight: 500; text-decoration: none; overflow-wrap: break-word;">RAG 教程</a><span>&nbsp;</span>以获取使用 LangChain 构建检索<a href="https://langchain-doc.cn/v1/python/langchain/agents" style="color: rgb(47, 132, 93); font-weight: 500; text-decoration: none; overflow-wrap: break-word;">代理</a>的端到端示例。</p></details>

------

#### 消息内容

您可以将消息的内容视为发送给模型的数据负载。消息具有一个松散类型的 `content` 属性，支持字符串和未类型对象列表（例如字典）。这允许在 LangChain 聊天模型中直接支持提供商原生结构，例如[多模态](https://langchain-doc.cn/v1/python/langchain/messages.html#多模态)内容和其他数据。

LangChain 另外为文本、推理、引用、多模态数据、服务器端工具调用和其他消息内容提供了专用内容类型。请参阅下面的[标准内容块](https://langchain-doc.cn/v1/python/langchain/messages.html#标准内容块)。

LangChain 聊天模型接受 `content` 属性中的消息内容，可以包含：

1. 一个字符串
2. 提供商原生格式的内容块列表
3. [LangChain 的标准内容块](https://langchain-doc.cn/v1/python/langchain/messages.html#标准内容块)列表

请参阅下面使用[多模态](https://langchain-doc.cn/v1/python/langchain/messages.html#多模态)输入的示例：

```python
from langchain.messages import HumanMessage

# 字符串内容
human_message = HumanMessage("Hello, how are you?")

# 提供商原生格式（例如 OpenAI）
human_message = HumanMessage(content=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

# 标准内容块列表
human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image", "url": "https://example.com/image.jpg"},
])
```

> **提示**
> 提示：在初始化消息时使用 `content_blocks` 仍然会自动生成 `content` 字段，但它提供了一个类型更安全的接口。



##### 标准内容块

LangChain 提供了一种跨提供商工作的消息内容的标准表示。

消息对象实现了 `content_blocks` 属性，该属性将延迟解析 `content` 属性为标准、类型安全的表示。例如，从 [ChatAnthropic](https://langchain-doc.cn/v1/python/integrations/chat/anthropic) 或 [ChatOpenAI](https://langchain-doc.cn/v1/python/integrations/chat/openai) 生成的消息将以各自提供商的格式包含 `thinking` 或 `reasoning` 块，但可以延迟解析为一致的 [`ReasoningContentBlock`](https://langchain-doc.cn/v1/python/langchain/messages.html#内容块参考) 表示：

```python
from langchain.messages import AIMessage

message = AIMessage(
    content=[
        {"type": "thinking", "thinking": "...", "signature": "WaUjzkyp..."},
        {"type": "text", "text": "..."},
    ],
    response_metadata={"model_provider": "anthropic"}
)
message.content_blocks
```

```
[{'type': 'reasoning',
  'reasoning': '...',
  'extras': {'signature': 'WaUjzkyp...'}},
 {'type': 'text', 'text': '...'}]
```

```python
from langchain.messages import AIMessage

message = AIMessage(
    content=[
        {
            "type": "reasoning",
            "id": "rs_abc123",
            "summary": [
                {"type": "summary_text", "text": "summary 1"},
                {"type": "summary_text", "text": "summary 2"},
            ],
        },
        {"type": "text", "text": "...", "id": "msg_abc123"},
    ],
    response_metadata={"model_provider": "openai"}
)
message.content_blocks
```

```
[{'type': 'reasoning', 'id': 'rs_abc123', 'reasoning': 'summary 1'},
 {'type': 'reasoning', 'id': 'rs_abc123', 'reasoning': 'summary 2'},
 {'type': 'text', 'text': '...', 'id': 'msg_abc123'}]
```

请参阅[集成指南](https://langchain-doc.cn/v1/python/integrations/providers/overview)以开始使用您选择的推理提供商。

> **序列化标准内容**
> 如果 LangChain 之外的应用程序需要访问标准内容块表示，您可以选择将内容块存储在消息内容中。
>
> 为此，您可以将 `LC_OUTPUT_VERSION` 环境变量设置为 `v1`。或者，使用 `output_version="v1"` 初始化任何聊天模型：
>
> ```python
> from langchain.chat_models import init_chat_model
> 
> model = init_chat_model("openai:gpt-5-nano", output_version="v1")
> ```



##### 多模态

**多模态**指的是处理以不同形式出现的数据的能力，例如文本、音频、图像和视频。LangChain 包含可跨提供商使用的这些数据的标准类型。

[聊天模型](https://langchain-doc.cn/v1/python/langchain/models) 可以接受多模态数据作为输入并生成作为输出。下面展示包含多模态数据的输入消息的简短示例。

> **注意**
> 额外键可以包含在内容块的顶层或嵌套在 `"extras": {"key": value}` 中。
>
> 例如，[OpenAI](https://langchain-doc.cn/v1/python/integrations/chat/openai#pdfs) 和 [AWS Bedrock Converse](https://langchain-doc.cn/v1/python/integrations/chat/bedrock) 对于 PDF 需要文件名。请参阅您选择的模型的[提供商页面](https://langchain-doc.cn/v1/python/integrations/providers/overview)以获取具体信息。

###### 图像输入

```python
# 从 URL
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "url": "https://example.com/path/to/image.jpg"},
    ]
}

# 从 base64 数据
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "image/jpeg",
        },
    ]
}

# 从提供商管理的文件 ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "file_id": "file-abc123"},
    ]
}
```

###### PDF 文档输入

```python
# 从 URL
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this document."},
        {"type": "file", "url": "https://example.com/path/to/document.pdf"},
    ]
}

# 从 base64 数据
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this document."},
        {
            "type": "file",
            "base64": "AAAAIGZ0eYBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "application/pdf",
        },
    ]
}

# 从提供商管理的文件 ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this document."},
        {"type": "file", "file_id": "file-abc123"},
    ]
}
```

###### 音频输入

```python
# 从 base64 数据
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this audio."},
        {
            "type": "audio",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "audio/wav",
        },
    ]
}

# 从提供商管理的文件 ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this audio."},
        {"type": "audio", "file_id": "file-abc123"},
    ]
}
```

###### 视频输入

```python
# 从 base64 数据
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this video."},
        {
            "type": "video",
            "base64": "AAAAIGZ0eYBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "video/mp4",
        },
    ]
}

# 从提供商管理的文件 ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this video."},
        {"type": "video", "file_id": "file-abc123"},
    ]
}
```

> **警告**
> 并非所有模型都支持所有文件类型。请检查模型提供商的[参考文档](https://reference.langchain.com/python/integrations/)以了解支持的格式和大小限制。

##### 内容块参考

内容块在创建消息或访问 `content_blocks` 属性时表示为类型化字典列表。列表中的每个项目必须遵守以下块类型之一：

###### 核心

**TextContentBlock**

**用途：** 标准文本输出

- **type** (`string`, 必需)
  始终为 `"text"`
- **text** (`string`, 必需)
  文本内容
- **annotations** (`object[]`)
  文本的注释列表
- **extras** (`object`)
  额外的提供商特定数据

**示例：**

```python
{
    "type": "text",
    "text": "Hello world",
    "annotations": []
}
```

**ReasoningContentBlock**

**用途：** 模型推理步骤

- **type** (`string`, 必需)
  始终为 `"reasoning"`
- **reasoning** (`string`)
  推理内容
- **extras** (`object`)
  额外的提供商特定数据

**示例：**



```python
{
    "type": "reasoning",
    "reasoning": "The user is asking about...",
    "extras": {"signature": "abc123"},
}
```

###### 多模态

**ImageContentBlock**

**用途：** 图像数据

- **type** (`string`, 必需)
  始终为 `"image"`
- **url** (`string`)
  指向图像位置的 URL。
- **base64** (`string`)
  Base64 编码的图像数据。
- **id** (`string`)
  引用外部存储的图像的引用 ID（例如，在提供商的文件系统或存储桶中）。
- **mime_type** (`string`)
  图像 [MIME 类型](https://www.iana.org/assignments/media-types/media-types.xhtml#image)（例如，`image/jpeg`、`image/png`）

**AudioContentBlock**

**用途：** 音频数据

- **type** (`string`, 必需)
  始终为 `"audio"`
- **url** (`string`)
  指向音频位置的 URL。
- **base64** (`string`)
  Base64 编码的音频数据。
- **id** (`string`)
  引用外部存储的音频文件的引用 ID。
- **mime_type** (`string`)
  音频 [MIME 类型](https://www.iana.org/assignments/media-types/media-types.xhtml#audio)（例如，`audio/mpeg`、`audio/wav`）

**VideoContentBlock**

**用途：** 视频数据

- **type** (`string`, 必需)
  始终为 `"video"`
- **url** (`string`)
  指向视频位置的 URL。
- **base64** (`string`)
  Base64 编码的视频数据。
- **id** (`string`)
  引用外部存储的视频文件的引用 ID。
- **mime_type** (`string`)
  视频 [MIME 类型](https://www.iana.org/assignments/media-types/media-types.xhtml#video)（例如，`video/mp4`、`video/webm`）

**FileContentBlock**

**用途：** 通用文件（PDF 等）

- **type** (`string`, 必需)
  始终为 `"file"`
- **url** (`string`)
  指向文件位置的 URL。
- **base64** (`string`)
  Base64 编码的文件数据。
- **id** (`string`)
  引用外部存储的文件的引用 ID。
- **mime_type** (`string`)
  文件 [MIME 类型](https://www.iana.org/assignments/media-types/media-types.xhtml)（例如，`application/pdf`）

**PlainTextContentBlock**

**用途：** 文档文本（`.txt`、`.md`）

- **type** (`string`, 必需)
  始终为 `"text-plain"`
- **text** (`string`)
  文本内容
- **mime_type** (`string`)
  文本的 [MIME 类型](https://www.iana.org/assignments/media-types/media-types.xhtml)（例如，`text/plain`、`text/markdown`）

###### 工具调用

**ToolCall**

**用途：** 函数调用

- **type** (`string`, 必需)
  始终为 `"tool_call"`
- **name** (`string`, 必需)
  要调用的工具的名称
- **args** (`object`, 必需)
  要传递给工具的参数
- **id** (`string`, 必需)
  此工具调用的唯一标识符

**示例：**



```python
{
    "type": "tool_call",
    "name": "search",
    "args": {"query": "weather"},
    "id": "call_123"
}
```

**ToolCallChunk**

**用途：** 流式工具调用片段

- **type** (`string`, 必需)
  始终为 `"tool_call_chunk"`
- **name** (`string`)
  正在调用的工具的名称
- **args** (`string`)
  部分工具参数（可能是不完整的 JSON）
- **id** (`string`)
  工具调用标识符
- **index** (`number | string`)
  此块在流中的位置

**InvalidToolCall**

**用途：** 格式错误的调用，用于捕获 JSON 解析错误。

- **type** (`string`, 必需)
  始终为 `"invalid_tool_call"`
- **name** (`string`)
  未能调用的工具的名称
- **args** (`object`)
  要传递给工具的参数
- **error** (`string`)
  出错的描述

###### 服务器端工具执行

**ServerToolCall**

**用途：** 服务器端执行的工具调用。

- **type** (`string`, 必需)
  始终为 `"server_tool_call"`
- **id** (`string`, 必需)
  与工具调用关联的标识符。
- **name** (`string`, 必需)
  要调用的工具的名称。
- **args** (`string`, 必需)
  部分工具参数（可能是不完整的 JSON）

**ServerToolCallChunk**

**用途：** 流式服务器端工具调用片段

- **type** (`string`, 必需)
  始终为 `"server_tool_call_chunk"`
- **id** (`string`)
  与工具调用关联的标识符。
- **name** (`string`)
  正在调用的工具的名称
- **args** (`string`)
  部分工具参数（可能是不完整的 JSON）
- **index** (`number | string`)
  此块在流中的位置

**ServerToolResult**

**用途：** 搜索结果

- **type** (`string`, 必需)
  始终为 `"server_tool_result"`
- **tool_call_id** (`string`, 必需)
  对应服务器工具调用的标识符。
- **id** (`string`)
  与服务器工具结果关联的标识符。
- **status** (`string`, 必需)
  服务器端工具的执行状态。`"success"` 或 `"error"`。
- **output**
  执行工具的输出

###### 提供商特定块

**NonStandardContentBlock**

**用途：** 提供商特定的逃生舱

- **type** (`string`, 必需)
  始终为 `"non_standard"`
- **value** (`object`, 必需)
  提供商特定的数据结构

**用法：** 用于实验性或提供商独特的功能

每个模型提供商的[参考文档](https://langchain-doc.cn/v1/python/integrations/providers/overview)中可能包含其他提供商特定的内容类型。

> **提示**
> 在 [API 参考](https://reference.langchain.com/python/langchain/messages) 中查看规范类型定义。

> **信息**
> 内容块在 LangChain v1 中作为消息上的新属性引入，以在保持与现有代码向后兼容的同时标准化跨提供商的内容格式。内容块不是 [`content`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.messages.BaseMessage.content) 属性的替代品，而是一个可以用于以标准化格式访问消息内容的新属性。

#### 与聊天模型一起使用

[聊天模型](https://langchain-doc.cn/v1/python/langchain/models) 接受消息对象序列作为输入，并返回 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 作为输出。交互通常是无状态的，因此简单的对话循环涉及使用不断增长的消息列表调用模型。

请参阅以下指南以了解更多信息：

- 用于[持久化和管理对话历史](https://langchain-doc.cn/v1/python/langchain/short-term-memory)的内置功能
- 管理上下文窗口的策略，包括[修剪和总结消息](https://langchain-doc.cn/v1/python/langchain/short-term-memory#common-patterns)



### 工具

许多 AI 应用程序通过自然语言与用户交互。然而，某些用例要求模型使用结构化输入直接与外部系统（例如 **API**、**数据库** 或 **文件系统**）对接。

**工具** 是 **[代理（agents）](https://langchain-doc.cn/v1/python/langchain/agents)** 调用来执行操作的组件。它们通过允许模型通过定义明确的输入和输出与世界交互来扩展模型的功能。工具封装了一个可调用的函数及其输入架构（schema）。这些可以传递给兼容的 **[聊天模型（chat models）](https://langchain-doc.cn/v1/python/langchain/models)**，让模型决定是否以及使用什么参数来调用工具。在这些场景中，**工具调用** 使模型能够生成符合指定输入架构的请求。

> **服务器端工具使用 (Server-side tool use)**
>
> 某些聊天模型（例如 **[OpenAI](https://langchain-doc.cn/v1/python/integrations/chat/openai)**、**[Anthropic](https://langchain-doc.cn/v1/python/integrations/chat/anthropic)** 和 **[Gemini](https://langchain-doc.cn/v1/python/integrations/chat/google_generative_ai)**）具有 **[内置工具](https://langchain-doc.cn/v1/python/langchain/models#server-side-tool-use)**，这些工具在服务器端执行，例如网络搜索和代码解释器。请参阅 **[提供商概览（provider overview）](https://langchain-doc.cn/v1/python/integrations/providers/overview)** 了解如何使用您的特定聊天模型访问这些工具。

#### 创建工具 (Create tools)

##### 基本工具定义 (Basic tool definition)

创建工具最简单的方法是使用 **[`@tool`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langchain/tools/#langchain.tools.tool](https://reference.langchain.com/python/langchain/tools/#langchain.tools.tool))** 装饰器。默认情况下，函数的 **文档字符串**（docstring）会成为工具的描述，帮助模型理解何时使用它：

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

**类型提示** 是**必需的**，因为它们定义了工具的输入架构。文档字符串应该**信息丰富且简洁**，以帮助模型理解工具的用途。

##### 自定义工具属性 (Customize tool properties)

###### 自定义工具名称 (Custom tool name)

默认情况下，工具名称来自函数名称。当您需要更具描述性的名称时，可以覆盖它：

```python
@tool("web_search")  # Custom name
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name)  # web_search
```

###### 自定义工具描述 (Custom tool description)

覆盖自动生成的工具描述，以提供更清晰的模型指导：

```python
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
```



##### 高级架构定义 (Advanced schema definition)

使用 **Pydantic 模型** 或 **JSON 架构** 定义复杂的输入：

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```

```python
weather_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string"},
        "include_forecast": {"type": "boolean"}
    },
    "required": ["location", "units", "include_forecast"]
}

@tool(args_schema=weather_schema)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```



#### 访问上下文 (Accessing Context)

> **为什么这很重要：** 当工具可以访问**代理状态**、**运行时上下文**和**长期记忆**时，它们的功能最强大。这使得工具能够做出**上下文感知**的决策、**个性化**响应并在对话中**维护信息**。

工具可以通过 `ToolRuntime` 参数访问运行时信息，该参数提供：

- **State（状态）** - 流经执行的可变数据（消息、计数器、自定义字段）
- **Context（上下文）** - 不可变的配置，如用户 ID、会话详细信息或特定于应用程序的配置
- **Store（存储）** - 跨对话的持久长期记忆
- **Stream Writer（流写入器）** - 在工具执行时流式传输自定义更新
- **Config（配置）** - 执行的 `RunnableConfig`
- **Tool Call ID（工具调用 ID）** - 当前工具调用的 ID



##### ToolRuntime

使用 `ToolRuntime` 在一个参数中访问所有运行时信息。只需将 `runtime: ToolRuntime` 添加到您的工具签名中，它就会被自动注入，而**不会暴露给 LLM**。

> **`ToolRuntime`**: 一个统一的参数，为工具提供对**状态**、**上下文**、**存储**、**流式传输**、**配置**和**工具调用 ID** 的访问。这取代了使用单独的 **[`InjectedState`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langgraph/agents/#langgraph.prebuilt.tool_node.InjectedState](https://reference.langchain.com/python/langgraph/agents/#langgraph.prebuilt.tool_node.InjectedState))**、**[`InjectedStore`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langgraph/agents/#langgraph.prebuilt.tool_node.InjectedStore](https://reference.langchain.com/python/langgraph/agents/#langgraph.prebuilt.tool_node.InjectedStore))**、**[`get_runtime`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langgraph/runtime/#langgraph.runtime.get_runtime](https://reference.langchain.com/python/langgraph/runtime/#langgraph.runtime.get_runtime))** 和 **[`InjectedToolCallId`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langchain/tools/#langchain.tools.InjectedToolCallId](https://reference.langchain.com/python/langchain/tools/#langchain.tools.InjectedToolCallId))** 注释的旧模式。

**访问状态：**

工具可以使用 `ToolRuntime` 访问当前的图状态（graph state）：

```python
from langchain.tools import tool, ToolRuntime

# Access the current conversation state
@tool
def summarize_conversation(
    runtime: ToolRuntime
) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

# Access custom state fields
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime  # ToolRuntime parameter is not visible to the model
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
```

> `tool_runtime` 参数对模型是隐藏的。对于上面的示例，模型在工具架构中**只看到** `pref_name` - `tool_runtime` **不**包含在请求中。



**更新状态：**

使用 **[`Command`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langgraph/types/#langgraph.types.Command](https://reference.langchain.com/python/langgraph/types/#langgraph.types.Command))** 来更新代理的状态或控制图的执行流程：

```python
from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.tools import tool, ToolRuntime

# Update the conversation history by removing all messages
@tool
def clear_conversation() -> Command:
    """Clear the conversation history."""

    return Command(
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
    )

# Update the user_name in the agent state
@tool
def update_user_name(
    new_name: str,
    runtime: ToolRuntime
) -> Command:
    """Update the user's name."""
    return Command(update={"user_name": new_name})
```



##### 上下文 (Context)

通过 `runtime.context` 访问**不可变**的配置和上下文数据，例如用户 ID、会话详细信息或特定于应用程序的配置。

工具可以通过 `ToolRuntime` 访问运行时上下文：

```python
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"

model = ChatOpenAI(model="gpt-4o")
agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant."
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my current balance?"}]},
    context=UserContext(user_id="user123")
)
```



###### 内存（存储）(Memory (Store))

使用 **存储（store）** 访问跨对话的持久数据。通过 `runtime.store` 访问存储，它允许您保存和检索特定于用户或特定于应用程序的数据。

工具可以通过 `ToolRuntime` 访问和更新存储：

```python
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# Update memory
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)

# First session: save user info
agent.invoke({
    "messages": [{"role": "user", "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
})

# Second session: get user info
agent.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})
# Here is the user info for user with ID "abc123":
# - Name: Foo
# - Age: 25
# - Email: foo@langchain.dev
```

###### 流写入器 (Stream Writer)

使用 `runtime.stream_writer` 在工具执行时流式传输自定义更新。这对于向用户提供有关工具正在做什么的**实时反馈**很有用。

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"
```

> 如果您在工具中使用 `runtime.stream_writer`，则**必须**在 LangGraph 执行上下文中调用该工具。有关更多详细信息，请参阅 **[流式传输（Streaming）](https://langchain-doc.cn/v1/python/langchain/streaming)**。



### 短期记忆

#### 概览 (Overview)

**记忆**是一个系统，用于记住关于先前交互的信息。对于 **AI 代理** (AI agents) 而言，记忆至关重要，因为它能让他们记住之前的交互、从反馈中学习并适应用户偏好。随着代理处理涉及大量用户交互的更复杂任务，这种能力对于**效率**和**用户满意度**都变得至关重要。

**短期记忆**让您的应用程序能够记住**单个线程**或**对话**中的先前交互。

> 💡 **注意：**
> 一个线程 (thread) 在一个会话 (session) 中组织多次交互，类似于电子邮件将消息分组到一个对话中的方式。

**会话历史记录** (Conversation history) 是短期记忆最常见的形式。对于当今的 **LLM** (大型语言模型) 来说，**长对话**是一个挑战；完整的历史记录可能无法完全容纳在一个 LLM 的**上下文窗口** (context window) 内，从而导致**上下文丢失** (context loss) 或**错误**。

即使您的模型支持完整的上下文长度，大多数 LLM 在处理**长上下文**时的表现仍然不佳。它们会被陈旧或跑题的内容“分心”，同时还会导致**响应时间变慢**和**成本更高**。

聊天模型使用 **消息** (messages) 接受上下文，这些消息包括**指令**（系统消息 system message）和**输入**（人类消息 human messages）。在聊天应用程序中，消息在人类输入和模型响应之间交替，导致消息列表随着时间推移而变长。由于上下文窗口是有限的，许多应用程序可以受益于使用技术来**移除**或**“遗忘”**陈旧信息。

#### 用法 (Usage)

要向代理添加短期记忆（线程级持久性），您需要在创建代理时指定一个 **`checkpointer`**。

> ℹ️ **信息：**
> LangChain 的代理将短期记忆作为代理**状态** (state) 的一部分进行管理。
>
> 通过将这些存储在图 (graph) 的状态中，代理可以访问给定对话的完整上下文，同时保持不同线程之间的分离。
>
> 状态使用 **checkpointer** 持久化到数据库（或内存）中，以便线程可以随时恢复。
>
> 当代理被调用或一个步骤（如工具调用）完成时，短期记忆会更新，并在每个步骤开始时读取状态。

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  # [!code highlight]


agent = create_agent(
    "openai:gpt-5",
    [get_user_info],
    checkpointer=InMemorySaver(),  # [!code highlight]
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},  # [!code highlight]
)
```

![image-20260613141234193](LangChain 教程.assets/image-20260613141234193.png)

###### 生产环境 (In production)

在生产环境中，请使用由**数据库**支持的 checkpointer：

```shell
pip install langgraph-checkpoint-postgres
```

```python
from langchain.agents import create_agent

from langgraph.checkpoint.postgres import PostgresSaver  # [!code highlight]


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgresSql
    agent = create_agent(
        "openai:gpt-5",
        [get_user_info],
        checkpointer=checkpointer,  # [!code highlight]
    )
```



#### 自定义代理记忆 (Customizing agent memory)

默认情况下，代理使用 **`AgentState`** 来管理短期记忆，特别是通过 **`messages`** 键来管理会话历史记录。

您可以扩展 **`AgentState`** 以添加额外的字段。自定义状态模式通过 **`state_schema`** 参数传递给 **`create_agent`**。

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver


class CustomAgentState(AgentState):  # [!code highlight]
    user_id: str  # [!code highlight]
    preferences: dict  # [!code highlight]

agent = create_agent(
    "openai:gpt-5",
    [get_user_info],
    state_schema=CustomAgentState,  # [!code highlight]
    checkpointer=InMemorySaver(),
)

# Custom state can be passed in invoke
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",  # [!code highlight]
        "preferences": {"theme": "dark"}  # [!code highlight]
    },
    {"configurable": {"thread_id": "1"}})
```



#### 常见模式 (Common patterns)

启用**短期记忆**后，长对话可能会超出 LLM 的上下文窗口。常见的解决方案有：

| 模式                                 | 描述                                           |
| :----------------------------------- | :--------------------------------------------- |
| **修剪消息** (Trim messages) ✂️       | 移除最初或最后的 N 条消息（在调用 LLM 之前）。 |
| **删除消息** (Delete messages) 🗑️     | 从 LangGraph 状态中**永久**删除消息。          |
| **总结消息** (Summarize messages) 📝  | 总结历史记录中较早的消息，并用摘要替换它们。   |
| **自定义策略** (Custom strategies) ⚙️ | 自定义策略（例如：消息过滤等）。               |

这使得代理能够在不超出 LLM 上下文窗口的情况下跟踪对话。

##### 修剪消息 (Trim messages)

大多数 LLM 都有一个最大支持的上下文窗口（以 token 计）。

决定何时截断消息的一种方法是**计算消息历史记录中的 token 数**，并在接近该限制时进行截断。如果您使用 LangChain，可以使用**修剪消息实用程序** (trim messages utility) 并指定要从列表中保留的 token 数量，以及用于处理边界的 `strategy`（例如：保留最后的 `max_tokens`）。

要在代理中修剪消息历史记录，请使用 **`@before_model`** 中间件装饰器：

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    model,
    tools=tools,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""
```



##### 删除消息 (Delete messages)

您可以从图状态中删除消息以管理消息历史记录。

当您想要删除特定消息或清除整个消息历史记录时，这非常有用。

要从图状态中删除消息，可以使用 **`RemoveMessage`**。

要使 `RemoveMessage` 工作，您需要使用具有 **`add_messages`** [reducer] 的状态键。

默认的 **`AgentState`** 提供了此功能。

删除**特定**消息：

```python
from langchain.messages import RemoveMessage  # [!code highlight]

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]} 
```

删除**所有**消息：

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES

def delete_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

> ⚠️ **警告：**
> 删除消息时，**请确保**生成的消息历史记录是**有效**的。请检查您正在使用的 LLM 提供商的限制。例如：
>
> - 一些提供商期望消息历史记录以 `user` 消息开始
> - 大多数提供商要求带有工具调用的 `assistant` 消息后跟相应的 `tool` 结果消息。

```python
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig


@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None


agent = create_agent(
    "openai:gpt-5-nano",
    tools=[],
    system_prompt="Please be concise and to the point.",
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

for event in agent.stream(
    {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])

for event in agent.stream(
    {"messages": [{"role": "user", "content": "what's my name?"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])
```

```sh
[('human', "hi! I'm bob")]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you. How can I help you today? I can answer questions, brainstorm ideas, draft text, explain things, or help with code.')]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you. How can I help you today? I can answer questions, brainstorm ideas, draft text, explain things, or help with code.'), ('human', "what's my name?")]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you. How can I help you today? I can answer questions, brainstorm ideas, draft text, explain things, or help with code.'), ('human', "what's my name?"), ('ai', 'Your name is Bob. How can I help you today, Bob?')]
[('human', "what's my name?"), ('ai', 'Your name is Bob. How can I help you today, Bob?')]
```



##### 总结消息 (Summarize messages)

如上所示，修剪或删除消息的问题是您可能会因为删除消息队列而**丢失信息**。
因此，一些应用程序受益于使用**聊天模型**来总结消息历史记录的更复杂方法。

要在代理中总结消息历史记录，请使用内置的 **`SummarizationMiddleware`**：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig


checkpointer = InMemorySaver()

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            max_tokens_before_summary=4000,  # Trigger summarization at 4000 tokens
            messages_to_keep=20,  # Keep last 20 messages after summary
        )
    ],
    checkpointer=checkpointer,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob!
"""
```

请参阅 `SummarizationMiddleware` 了解更多配置选项。



#### 访问记忆 (Access memory)

您可以通过几种方式访问和修改代理的短期记忆（状态）：

##### 工具 (Tools)

###### 在工具中读取短期记忆 (Read short-term memory in a tool)

使用 **`ToolRuntime`** 参数在工具中访问短期记忆（状态）。

`tool_runtime` 参数对工具签名是**隐藏**的（因此模型看不到它），但工具可以通过它访问状态。

```python
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime


class CustomState(AgentState):
    user_id: str

@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """Look up user info."""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[get_user_info],
    state_schema=CustomState,
)

result = agent.invoke({
    "messages": "look up user information",
    "user_id": "user_123"
})
print(result["messages"][-1].content)
# > User is John Smith.
```

###### 从工具写入短期记忆 (Write short-term memory from tools)

要在执行期间修改代理的短期记忆（状态），您可以直接从工具返回**状态更新** (state updates)。

这对于持久化中间结果或使信息可供后续工具或提示访问非常有用。



```python
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel


class CustomState(AgentState):  # [!code highlight]
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """Look up and update user info."""
    user_id = runtime.context.user_id  # [!code highlight]
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        # update the message history
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state["user_name"]
    return f"Hello {user_name}!"
  # [!code highlight]
agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[update_user_info, greet],
    state_schema=CustomState,
    context_schema=CustomContext,  # [!code highlight]
)

agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123"),
)
```

##### 提示 (Prompt)

在**中间件** (middleware) 中访问短期记忆（状态），以基于对话历史记录或自定义状态字段创建**动态提示** (dynamic prompts)。

```python
from langchain.messages import AnyMessage
from langchain.agents import create_agent, AgentState
from typing import TypedDict


class CustomContext(TypedDict):
    user_name: str


from langchain.agents.middleware import dynamic_prompt, ModelRequest

def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is always sunny!"


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt


agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext(user_name="John Smith"),
)
for msg in result["messages"]:
    msg.pretty_print()
```

Output

```shell
================================ Human Message =================================

What is the weather in SF?
================================== Ai Message ==================================
Tool Calls:
  get_weather (call_WFQlOGn4b2yoJrv7cih342FG)
  Call ID: call_WFQlOGn4b2yoJrv7cih342FG
  Args:
    city: San Francisco
================================= Tool Message =================================
Name: get_weather

The weather in San Francisco is always sunny!
================================== Ai Message ==================================

Hi John Smith, the weather in San Francisco is always sunny!
```

##### 模型之前 (Before model)

在 **`@before_model`** 中间件中访问短期记忆（状态），以在模型调用之前处理消息。

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    model,
    tools=tools,
    middleware=[trim_messages]
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""
```

##### 模型之后 (After model)

在 **`@after_model`** 中间件中访问短期记忆（状态），以在模型调用之后处理消息。

```python
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.runtime import Runtime


@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove messages containing sensitive words."""
    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]
    if any(word in last_message.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None

agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[],
    middleware=[validate_response],
    checkpointer=InMemorySaver(),
)
```



### 流式传输

LangChain 实现了一个流式传输系统来提供**实时更新**。

流式传输对于增强基于 **LLM**（大型语言模型）构建的应用程序的**响应能力**至关重要。通过**渐进式**地显示输出，即使在完整的响应准备好之前，流式传输也能显著**改善用户体验 (UX)**，尤其是在处理 LLM 的延迟时。

#### 概述 (Overview)

LangChain 的流式传输系统允许您将代理运行的**实时反馈**展示给您的应用程序。

LangChain 流式传输可能实现的功能：

- **流式传输代理进度** — 在每个代理步骤后获取状态更新。
- **流式传输 LLM 令牌** — 在语言模型生成令牌时进行流式传输。
- **流式传输自定义更新** — 发出用户定义的信号（例如，`"Fetched 10/100 records"`）。
- **流式传输多种模式** — 可选择 `updates`（代理进度）、`messages`（LLM 令牌 + 元数据）或 `custom`（任意用户数据）。

#### 代理进度 (Agent progress)

要流式传输代理进度，请使用 `stream_mode="updates"` 的 [`stream`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langgraph/graphs/#langgraph.graph.state.CompiledStateGraph.stream](https://reference.langchain.com/python/langgraph/graphs/#langgraph.graph.state.CompiledStateGraph.stream)) 或 [`astream`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langgraph/graphs/#langgraph.graph.state.CompiledStateGraph.astream](https://reference.langchain.com/python/langgraph/graphs/#langgraph.graph.state.CompiledStateGraph.astream)) 方法。这会在**每个代理步骤**后发出一个事件。

例如，如果您有一个调用工具一次的代理，您应该会看到以下更新：

- **LLM 节点**: 带有工具调用请求的 [`AIMessage`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage))
- **工具节点**: 带有执行结果的 [`ToolMessage`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langchain/messages/#langchain.messages.ToolMessage](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ToolMessage))
- **LLM 节点**: 最终的 AI 响应

**Streaming agent progress**

```python
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """获取给定城市的天气。"""

    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[get_weather],
)
for chunk in agent.stream(  # [!code highlight]
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")
```

Output

```shell
step: model
content: [{'type': 'tool_call', 'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_OW2NYNsNSKhRZpjW0wm2Aszd'}]

step: tools
content: [{'type': 'text', 'text': "It's always sunny in San Francisco!"}]

step: model
content: [{'type': 'text', 'text': 'It's always sunny in San Francisco!'}]
```

#### LLM 令牌 (LLM tokens)

要在 LLM 生成令牌时流式传输它们，请使用 `stream_mode="messages"`。您可以在下面看到代理流式传输工具调用和最终响应的输出。

**Streaming LLM tokens**

```python
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """获取给定城市的天气。"""

    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[get_weather],
)
for token, metadata in agent.stream(  # [!code highlight]
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")
    print("\n")
```

Output

```shell
node: model
content: [{'type': 'tool_call_chunk', 'id': 'call_vbCyBcP8VuneUzyYlSBZZsVa', 'name': 'get_weather', 'args': '', 'index': 0}]


node: model
content: [{'type': 'tool_call_chunk', 'id': None, 'name': None, 'args': '{"', 'index': 0}]


node: model
content: [{'type': 'tool_call_chunk', 'id': None, 'name': None, 'args': 'city', 'index': 0}]


node: model
content: [{'type': 'tool_call_chunk', 'id': None, 'name': None, 'args': '":"', 'index': 0}]


node: model
content: [{'type': 'tool_call_chunk', 'id': None, 'name': None, 'args': 'San', 'index': 0}]


node: model
content: [{'type': 'tool_call_chunk', 'id': None, 'name': None, 'args': ' Francisco', 'index': 0}]


node: model
content: [{'type': 'tool_call_chunk', 'id': None, 'name': None, 'args': '"}', 'index': 0}]


node: model
content: []


node: tools
content: [{'type': 'text', 'text': "It's always sunny in San Francisco!"}]


node: model
content: []


node: model
content: [{'type': 'text', 'text': 'Here'}]


node: model
content: [{'type': 'text', 'text': ''s'}]


node: model
content: [{'type': 'text', 'text': ' what'}]


node: model
content: [{'type': 'text', 'text': ' I'}]


node: model
content: [{'type': 'text', 'text': ' got'}]


node: model
content: [{'type': 'text', 'text': ':'}]


node: model
content: [{'type': 'text', 'text': ' "'}]


node: model
content: [{'type': 'text', 'text': "It's"}]


node: model
content: [{'type': 'text', 'text': ' always'}]


node: model
content: [{'type': 'text', 'text': ' sunny'}]


node: model
content: [{'type': 'text', 'text': ' in'}]


node: model
content: [{'type': 'text', 'text': ' San'}]


node: model
content: [{'type': 'text', 'text': ' Francisco'}]


node: model
content: [{'type': 'text', 'text': '!"\n\n'}]
```

#### 自定义更新 (Custom updates)

要流式传输工具在执行过程中发出的更新，您可以使用 [`get_stream_writer`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langgraph/config/#langgraph.config.get_stream_writer](https://reference.langchain.com/python/langgraph/config/#langgraph.config.get_stream_writer))。

Streaming custom updates

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer  # [!code highlight]


def get_weather(city: str) -> str:
    """获取给定城市的天气。"""
    writer = get_stream_writer()  # [!code highlight]
    # 流式传输任何任意数据
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="custom"  # [!code highlight]
):
    print(chunk)
```

Output

```shell
Looking up data for city: San Francisco
Acquired data for city: San Francisco
```

> **注意 (Note)**
> 如果您在工具内部添加 [`get_stream_writer`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langgraph/config/#langgraph.config.get_stream_writer](https://reference.langchain.com/python/langgraph/config/#langgraph.config.get_stream_writer))，您将无法在 LangGraph 执行上下文之外调用该工具。

#### 流式传输多种模式 (Stream multiple modes)

您可以通过将流模式作为列表传递来指定多种流式传输模式：`stream_mode=["updates", "custom"]`：

**Streaming multiple modes**

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer


def get_weather(city: str) -> str:
    """获取给定城市的天气。"""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[get_weather],
)

for stream_mode, chunk in agent.stream(  # [!code highlight]
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"]
):
    print(f"stream_mode: {stream_mode}")
    print(f"content: {chunk}")
    print("\n")
```

Output

```shell
stream_mode: updates
content: {'model': {'messages': [AIMessage(content='', response_metadata={'token_usage': {'completion_tokens': 280, 'prompt_tokens': 132, 'total_tokens': 412, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 256, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-5-nano-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-C9tlgBzGEbedGYxZ0rTCz5F7OXpL7', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--480c07cb-e405-4411-aa7f-0520fddeed66-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_KTNQIftMrl9vgNwEfAJMVu7r', 'type': 'tool_call'}], usage_metadata={'input_tokens': 132, 'output_tokens': 280, 'total_tokens': 412, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 256}})]}}


stream_mode: custom
content: Looking up data for city: San Francisco


stream_mode: custom
content: Acquired data for city: San Francisco


stream_mode: updates
content: {'tools': {'messages': [ToolMessage(content="It's always sunny in San Francisco!", name='get_weather', tool_call_id='call_KTNQIftMrl9vgNwEfAJMVu7r')]}}


stream_mode: updates
content: {'model': {'messages': [AIMessage(content='San Francisco weather: It's always sunny in San Francisco!\n\n', response_metadata={'token_usage': {'completion_tokens': 764, 'prompt_tokens': 168, 'total_tokens': 932, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 704, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-5-nano-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-C9tljDFVki1e1haCyikBptAuXuHYG', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--acbc740a-18fe-4a14-8619-da92a0d0ee90-0', usage_metadata={'input_tokens': 168, 'output_tokens': 764, 'total_tokens': 932, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 704}})]}}
```

#### 禁用流式传输 (Disable streaming)

在某些应用程序中，您可能需要为给定的模型**禁用单个令牌的流式传输**。

这对于像[多代理](https://langchain-doc.cn/v1/python/langchain/multi-agent)系统这样的场景很有用，以便**控制**哪些代理流式传输它们的输出。

请参阅 [模型](https://langchain-doc.cn/v1/python/langchain/models#disable-streaming) 指南以了解如何禁用流式传输。



### 中间件

> 控制并定制代理（Agent）执行的每一个步骤

中间件提供了一种方法，可以更精细地控制**代理**（Agent）内部的执行流程。核心的代理循环包括调用模型、让模型选择并执行工具，然后在模型不再调用工具时结束：

![core_agent_loop.png](LangChain 教程.assets/core_agent_loop.png)

​                                                                                  core_agent_loop.png
中间件在这些步骤的**之前**和**之后**暴露了钩子 (hooks)：

![middleware_final.png](LangChain 教程.assets/middleware_final.png)

​																										*middleware_final.png*

#### 中间件可以做什么？

- **监控 (Monitor)**
  追踪代理行为，包括日志记录、分析和调试。
- **修改 (Modify)**
  转换提示词（prompts）、工具选择和输出格式。
- **控制 (Control)**
  添加重试、回退和提前终止逻辑。
- **强制执行 (Enforce)**
  应用速率限制、安全防护（guardrails）和个人身份信息（PII）检测。

通过将其传递给 `create_agent` 函数来添加中间件：
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware


agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[SummarizationMiddleware(), HumanInTheLoopMiddleware()],
)
```

#### 预置中间件 (Built-in middleware)

LangChain 为常见用例提供了预构建的中间件：

##### 摘要 (Summarization)

在接近**上下文窗口**（token limits）限制时，自动对对话历史进行摘要。

> **完美适用于：**
>
> - 超过上下文窗口的**长期对话**
> - 具有大量历史记录的**多轮对话**
> - 需要保留完整对话上下文的应用

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware


agent = create_agent(
    model="openai:gpt-4o",
    tools=[weather_tool, calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            max_tokens_before_summary=4000,  # 在 4000 个 token 时触发摘要
            messages_to_keep=20,  # 摘要后保留最近 20 条消息
            summary_prompt="Custom prompt for summarization...",  # 可选
        ),
    ],
)
```

**配置选项：**

- `model`: **字符串**，用于生成摘要的模型。
- `max_tokens_before_summary`: **数字**，触发摘要的 token 阈值。
- `messages_to_keep`: **数字 (默认值: 20)**，要保留的最新消息数量。
- `token_counter`: **函数**，自定义 token 计数函数。默认为基于字符的计数。
- `summary_prompt`: **字符串**，自定义提示词模板。如果未指定，则使用内置模板。
- `summary_prefix`: **字符串 (默认值: "## Previous conversation summary:")**，摘要消息的前缀。

##### 人在回路中 (Human-in-the-loop)

在工具执行之前，暂停代理执行以供人工批准、编辑或拒绝工具调用。

> **完美适用于：**
>
> - 需要人工批准的**高风险操作**（数据库写入、金融交易）
> - 强制要求人工监督的**合规工作流**
> - 使用人工反馈来指导代理的**长期对话**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


agent = create_agent(
    model="openai:gpt-4o",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 要求对发送邮件进行批准、编辑或拒绝
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                # 自动批准读取邮件
                "read_email_tool": False,
            }
        ),
    ],
)
```

**配置选项：**

- `interrupt_on`: **字典 (必需)**，工具名称到批准配置的映射。值可以是 `True`（使用默认配置中断）、`False`（自动批准）或一个 `InterruptOnConfig` 对象。
- `description_prefix`: **字符串 (默认值: "Tool execution requires approval")**，操作请求描述的前缀。

**`InterruptOnConfig` 选项：**

- `allowed_decisions`: **字符串列表**，允许的决定列表：`"approve"`、`"edit"` 或 `"reject"`。
- `description`: **字符串 | 可调用对象**，用于自定义描述的静态字符串或可调用函数。

> **⚠️ 重要提示：** “人在回路中”中间件需要一个**检查点**（checkpointer）来在中断期间维护状态。

##### 缓存 (Anthropic prompt caching)

通过使用 Anthropic 模型，缓存重复的提示词前缀以降低成本。

> **完美适用于：**
>
> - 具有**冗长、重复系统提示词**的应用
> - 在多次调用中**重用相同上下文**的代理
> - 为高流量部署**降低 API 成本**

> **ℹ️ 了解更多：** 请参阅 [Anthropic 提示词缓存](https://docs.claude.com/en/docs/build-with-claude/prompt-caching#cache-limitations)策略和限制。

```python
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain.agents import create_agent


LONG_PROMPT = """
Please be a helpful assistant.

<Lots more context ...>
"""

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-latest"),
    system_prompt=LONG_PROMPT,
    middleware=[AnthropicPromptCachingMiddleware(ttl="5m")],
)

# 缓存存储
agent.invoke({"messages": [HumanMessage("Hi, my name is Bob")]})

# 缓存命中，系统提示词被缓存
agent.invoke({"messages": [HumanMessage("What's my name?")]})
```

**配置选项：**

- `type`: **字符串 (默认值: "ephemeral")**，缓存类型。目前仅支持 `"ephemeral"`。
- `ttl`: **字符串 (默认值: "5m")**，缓存内容的存活时间 (Time to live)。有效值：`"5m"` 或 `"1h"`。
- `min_messages_to_cache`: **数字 (默认值: 0)**，开始缓存之前的最小消息数量。
- `unsupported_model_behavior`: **字符串 (默认值: "warn")**，使用非 Anthropic 模型时的行为。选项：`"ignore"`、`"warn"` 或 `"raise"`。

##### 限制模型调用 (Model call limit)

限制模型调用的次数，以防止无限循环或成本过高。

> **完美适用于：**
>
> - 防止**失控代理**进行过多 API 调用
> - 在生产部署中**强制执行成本控制**
> - 在特定调用预算内**测试代理行为**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware


agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,  # 每个线程（跨多次运行）最多 10 次调用
            run_limit=5,  # 每次运行（单次调用）最多 5 次调用
            exit_behavior="end",  # 或者 "error" 以引发异常
        ),
    ],
)
```

**配置选项：**

- `thread_limit`: **数字**，线程中所有运行的最大模型调用次数。默认为无限制。
- `run_limit`: **数字**，单次调用中最大模型调用次数。默认为无限制。
- `exit_behavior`: **字符串 (默认值: "end")**，达到限制时的行为。选项：`"end"`（优雅终止）或 `"error"`（引发异常）。

##### 限制工具调用 (Tool call limit)

限制特定工具或所有工具的调用次数。

> **完美适用于：**
>
> - 防止对昂贵的外部 API 进行**过多调用**
> - 限制**网页搜索**或**数据库查询**
> - 对特定工具使用**强制执行速率限制**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware


# 限制所有工具调用
global_limiter = ToolCallLimitMiddleware(thread_limit=20, run_limit=10)

# 限制特定工具
search_limiter = ToolCallLimitMiddleware(
    tool_name="search",
    thread_limit=5,
    run_limit=3,
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[global_limiter, search_limiter],
)
```

**配置选项：**

- `tool_name`: **字符串**，要限制的特定工具。如果未提供，则限制适用于所有工具。
- `thread_limit`: **数字**，线程中所有运行的最大工具调用次数。默认为无限制。
- `run_limit`: **数字**，单次调用中最大工具调用次数。默认为无限制。
- `exit_behavior`: **字符串 (默认值: "end")**，达到限制时的行为。选项：`"end"`（优雅终止）或 `"error"`（引发异常）。

##### 模型回退 (Model fallback)

当主要模型失败时，自动回退到替代模型。

> **完美适用于：**
>
> - 构建能够处理**模型中断**的弹性代理
> - 通过回退到更便宜的模型实现**成本优化**
> - 跨 OpenAI、Anthropic 等提供商实现**冗余**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware


agent = create_agent(
    model="openai:gpt-4o",  # 主要模型
    tools=[...],
    middleware=[
        ModelFallbackMiddleware(
            "openai:gpt-4o-mini",  # 错误时首先尝试
            "anthropic:claude-3-5-sonnet-20241022",  # 然后尝试这个
        ),
    ],
)
```

**配置选项：**

- `first_model`: **字符串 | BaseChatModel (必需)**，主要模型失败时尝试的第一个回退模型。可以是模型字符串（例如，`"openai:gpt-4o-mini"`）或 `BaseChatModel` 实例。
- `*additional_models`: **字符串 | BaseChatModel**，如果先前的模型失败，则按顺序尝试的其他回退模型。

##### PII 检测 (PII detection)

检测和处理对话中的**个人身份信息**（Personally Identifiable Information, PII）。

> **完美适用于：**
>
> - 具有合规性要求的**医疗保健和金融应用**
> - 需要清理日志的**客户服务代理**
> - 任何处理**敏感用户数据**的应用

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware


agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        # 涂改用户输入中的电子邮件
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        # 掩盖信用卡（显示后 4 位）
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        # 带有正则表达式的自定义 PII 类型
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",  # 如果检测到，则引发错误
        ),
    ],
)
```

**配置选项：**

- `pii_type`: **字符串 (必需)**，要检测的 PII 类型。可以是内置类型（`email`、`credit_card`、`ip`、`mac_address`、`url`）或自定义类型名称。
- strategy：字符串 (默认值: "redact")，处理检测到的 PII 的方式。选项：
  - `"block"` - 检测到时引发异常
  - `"redact"` - 替换为 `[REDACTED_TYPE]`
  - `"mask"` - 部分掩盖（例如，`****-****-****-1234`）
  - `"hash"` - 替换为确定性哈希
- `detector`: **函数 | 正则表达式**，自定义检测函数或正则表达式模式。如果未提供，则使用 PII 类型的内置检测器。
- `apply_to_input`: **布尔值 (默认值: True)**，在模型调用前检查用户消息。
- `apply_to_output`: **布尔值 (默认值: False)**，在模型调用后检查 AI 消息。
- `apply_to_tool_results`: **布尔值 (默认值: False)**，在执行后检查工具结果消息。

##### 规划 (Planning)

为复杂的**多步骤任务**添加待办事项列表管理功能。

> **注意：** 此中间件自动为代理提供一个 `write_todos` 工具和系统提示词，以指导有效的任务规划。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.messages import HumanMessage


agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[TodoListMiddleware()],
)

result = agent.invoke({"messages": [HumanMessage("Help me refactor my codebase")]})
print(result["todos"])  # 带有状态跟踪的待办事项数组
```

**配置选项：**

- `system_prompt`: **字符串**，用于指导待办事项使用的自定义系统提示词。如果未指定，则使用内置提示词。
- `tool_description`: **字符串**，`write_todos` 工具的自定义描述。如果未指定，则使用内置描述。

##### LLM 工具选择器 (LLM tool selector)

在调用主模型之前，使用 LLM 智能地选择**相关工具**。

> **完美适用于：**
>
> - 具有**许多工具（10+）**，但大多数工具与查询不相关的代理
> - 通过过滤不相关的工具来**减少 token 使用**
> - **提高模型焦点**和准确性

```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware


agent = create_agent(
    model="openai:gpt-4o",
    tools=[tool1, tool2, tool3, tool4, tool5, ...],  # 许多工具
    middleware=[
        LLMToolSelectorMiddleware(
            model="openai:gpt-4o-mini",  # 使用更便宜的模型进行选择
            max_tools=3,  # 限制为 3 个最相关的工具
            always_include=["search"],  # 始终包含某些工具
        ),
    ],
)
```

**配置选项：**

- `model`: **字符串 | BaseChatModel**，用于工具选择的模型。可以是模型字符串或 `BaseChatModel` 实例。默认为代理的主模型。
- `system_prompt`: **字符串**，给选择模型的指令。如果未指定，则使用内置提示词。
- `max_tools`: **数字**，要选择的最大工具数量。默认为无限制。
- `always_include`: **字符串列表**，始终包含在选择中的工具名称列表。

##### 工具重试 (Tool retry)

使用可配置的**指数回退**自动重试失败的工具调用。

> **完美适用于：**
>
> - 处理**外部 API 调用中的瞬时故障**
> - 提高**依赖网络的工具**的可靠性
> - 构建能够**优雅处理临时错误**的弹性代理

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware


agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,  # 最多重试 3 次
            backoff_factor=2.0,  # 指数回退乘数
            initial_delay=1.0,  # 从 1 秒延迟开始
            max_delay=60.0,  # 将延迟上限设置为 60 秒
            jitter=True,  # 添加随机抖动以避免“惊群”问题
        ),
    ],
)
```

**配置选项：**

- `max_retries`: **数字 (默认值: 2)**，初始调用后的最大重试次数（默认总共 3 次尝试）。
- `tools`: **BaseTool | 字符串列表**，可选的工具列表或工具名称，用于应用重试逻辑。如果为 `None`，则应用于所有工具。
- `retry_on`: **异常类型元组 | 可调用对象 (默认值: (Exception,))**，要么是重试的异常类型元组，要么是接受异常并返回 `True` 表示应重试的可调用对象。
- on_failure：字符串 | 可调用对象 (默认值: "return_message")，所有重试耗尽时的行为。选项：
  - `"return_message"` - 返回带有错误详细信息的 ToolMessage（允许 LLM 处理失败）
  - `"raise"` - 重新引发异常（停止代理执行）
  - 自定义可调用对象 - 接受异常并返回用于 ToolMessage 内容的字符串的函数
- `backoff_factor`: **数字 (默认值: 2.0)**，指数回退的乘数。每次重试等待 `initial_delay * (backoff_factor ** retry_number)` 秒。设置为 0.0 表示恒定延迟。
- `initial_delay`: **数字 (默认值: 1.0)**，第一次重试前的初始延迟（秒）。
- `max_delay`: **数字 (默认值: 60.0)**，重试之间的最大延迟（秒）（限制指数回退的增长）。
- `jitter`: **布尔值 (默认值: true)**，是否添加随机抖动（±25%）以避免“惊群”问题。

##### LLM 工具模拟器 (LLM tool emulator)

使用 LLM 模拟工具执行，用于测试目的，用 AI 生成的响应**替换实际工具调用**。

> **完美适用于：**
>
> - **在不执行真实工具的情况下测试代理行为**
> - 在**外部工具不可用或昂贵**时开发代理
> - 在实现实际工具之前**进行代理工作流原型设计**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator

agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather, search_database, send_email],
    middleware=[
        # 默认模拟所有工具
        LLMToolEmulator(),

        # 或模拟特定工具
        # LLMToolEmulator(tools=["get_weather", "search_database"]),

        # 或使用自定义模型进行模拟
        # LLMToolEmulator(model="anthropic:claude-3-5-sonnet-latest"),
    ],
)
```

**配置选项：**

- `tools`: **字符串 | BaseTool 列表**，要模拟的工具名称（字符串）或 `BaseTool` 实例列表。如果为 `None`（默认），**所有**工具将被模拟。如果为空列表，则不会模拟任何工具。
- `model`: **字符串 | BaseChatModel (默认值: "anthropic:claude-3-5-sonnet-latest")**，用于生成模拟工具响应的模型。可以是模型标识符字符串或 `BaseChatModel` 实例。

##### 上下文编辑 (Context editing)

通过**修剪**、**摘要**或**清除工具使用**来管理对话上下文。

> **完美适用于：**
>
> - 需要定期**清理上下文**的长时间对话
> - 从上下文中**移除失败的工具尝试**
> - **自定义上下文管理策略**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit


agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(max_tokens=1000),  # 清除旧的工具使用
            ],
        ),
    ],
)
```

**配置选项：**

- `edits`: **ContextEdit 列表 (默认值: [ClearToolUsesEdit()])**，要应用的 `ContextEdit` 策略列表。
- `token_count_method`: **字符串 (默认值: "approximate")**，token 计数方法。选项：`"approximate"` 或 `"model"`。

**[`ClearToolUsesEdit`](https://www.google.com/search?q=[https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.ClearToolUsesEdit](https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.ClearToolUsesEdit)) 选项：**

- `trigger`: **数字 (默认值: 100000)**，触发编辑的 token 计数。
- `clear_at_least`: **数字 (默认值: 0)**，要回收的最小 token 数。
- `keep`: **数字 (默认值: 3)**，要保留的最近工具结果数量。
- `clear_tool_inputs`: **布尔值 (默认值: False)**，是否清除工具调用参数。
- `exclude_tools`: **字符串列表 (默认值: ())**，要从清除中排除的工具名称列表。
- `placeholder`: **字符串 (默认值: "[cleared]")**，清除输出的占位符文本。

#### 自定义中间件 (Custom middleware)

通过实现运行在代理执行流程中特定点的**钩子**（hooks）来构建自定义中间件。

您可以通过两种方式创建中间件：

1. **基于装饰器 (Decorator-based)** - 适用于单钩子中间件，快速且简单
2. **基于类 (Class-based)** - 适用于具有多个钩子的复杂中间件，功能更强大

##### 基于装饰器的中间件

对于只需要一个钩子的简单中间件，装饰器提供了最快的方法来添加功能：

```python
from langchain.agents.middleware import before_model, after_model, wrap_model_call
from langchain.agents.middleware import AgentState, ModelRequest, ModelResponse, dynamic_prompt
from langchain.messages import AIMessage
from langchain.agents import create_agent
from langgraph.runtime import Runtime
from typing import Any, Callable


# 节点式 (Node-style)：模型调用前的日志记录
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"About to call model with {len(state['messages'])} messages")
    return None

# 节点式 (Node-style)：模型调用后的验证
@after_model(can_jump_to=["end"])
def validate_output(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_message = state["messages"][-1]
    if "BLOCKED" in last_message.content:
        return {
            "messages": [AIMessage("I cannot respond to that request.")],
            "jump_to": "end"
        }
    return None

# 包装式 (Wrap-style)：重试逻辑
@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")

# 包装式 (Wrap-style)：动态提示词
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.get("user_id", "guest")
    return f"You are a helpful assistant for user {user_id}. Be concise and friendly."

# 在代理中使用装饰器
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[log_before_model, validate_output, retry_model, personalized_prompt],
    tools=[...],
)
```

###### 可用装饰器

**节点式 (Node-style)**（在特定的执行点运行）：

- `@before_agent` - 代理启动前（每次调用一次）
- [`@before_model`](https://www.google.com/search?q=[https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.before_model](https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.before_model)) - 每次模型调用前
- [`@after_model`](https://www.google.com/search?q=[https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.after_model](https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.after_model)) - 每次模型响应后
- `@after_agent` - 代理完成时（每次调用最多一次）

**包装式 (Wrap-style)**（拦截并控制执行）：

- [`@wrap_model_call`](https://www.google.com/search?q=[https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.wrap_model_call](https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.wrap_model_call)) - 每次模型调用周围
- [`@wrap_tool_call`](https://www.google.com/search?q=[https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.wrap_tool_call](https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.wrap_tool_call)) - 每次工具调用周围

**便利装饰器 (Convenience decorators)**：

- [`@dynamic_prompt`](https://www.google.com/search?q=[https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.dynamic_prompt](https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.dynamic_prompt)) - 生成动态系统提示词（相当于修改提示词的 [`@wrap_model_call`](https://www.google.com/search?q=[https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.wrap_model_call](https://reference.langchain.com/python/langchain/middleware/%23langchain.agents.middleware.wrap_model_call))）

###### 何时使用装饰器

| ✔️ 使用装饰器的时机 | 💻 使用类的时机                 |
| :----------------- | :----------------------------- |
| • 您只需要一个钩子 | • 需要多个钩子                 |
| • 没有复杂的配置   | • 复杂的配置                   |
|                    | • 跨项目复用（通过初始化配置） |

##### 基于类的中间件

###### 两种钩子样式

- **项目图表 节点式钩子 (Node-style hooks)**
  在特定的执行点**顺序运行**。用于日志记录、验证和状态更新。
- **包装式钩子 (Wrap-style hooks)**
  **拦截执行**并对处理程序调用具有完全控制。用于重试、缓存和转换。

###### 节点式钩子 (Node-style hooks)

在执行流程中的特定点运行：

- `before_agent` - 代理启动前（每次调用一次）
- `before_model` - 每次模型调用前
- `after_model` - 每次模型响应后
- `after_agent` - 代理完成时（每次调用最多一次）

**示例：日志记录中间件**

```python
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from typing import Any

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None
```

**示例：对话长度限制**

```python
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any

class MessageLimitMiddleware(AgentMiddleware):
    def __init__(self, max_messages: int = 50):
        super().__init__()
        self.max_messages = max_messages

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if len(state["messages"]) == self.max_messages:
            return {
                "messages": [AIMessage("Conversation limit reached.")],
                "jump_to": "end"
            }
        return None
```

###### 包装式钩子 (Wrap-style hooks)

拦截执行并控制何时调用处理程序：

- `wrap_model_call` - 每次模型调用周围
- `wrap_tool_call` - 每次工具调用周围

您可以决定处理程序是调用**零次**（短路）、**一次**（正常流程）还是**多次**（重试逻辑）。

**示例：模型重试中间件**

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable

class RetryMiddleware(AgentMiddleware):
    def __init__(self, max_retries: int = 3):
        super().__init__()
        self.max_retries = max_retries

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        for attempt in range(self.max_retries):
            try:
                return handler(request)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Retry {attempt + 1}/{self.max_retries} after error: {e}")
```

**示例：动态模型选择**

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

class DynamicModelMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 根据对话长度使用不同的模型
        # ...
```



### 结构化输出

结构化输出允许 **智能体（agents）** 以特定的、可预测的格式返回数据。这样，您无需解析自然语言响应，即可获得 **JSON 对象**、**Pydantic 模型** 或 **数据类（dataclasses）** 形式的结构化数据，供您的应用程序直接使用。

LangChain 的 [`create_agent`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)) 会自动处理结构化输出。用户设置所需的结构化输出 **模式（schema）**，当模型生成结构化数据时，它会被捕获、验证，并作为智能体状态中 `'structured_response'` 键的值返回。

```python
def create_agent(
    ...
    response_format: Union[
        ToolStrategy[StructuredResponseT],
        ProviderStrategy[StructuredResponseT],
        type[StructuredResponseT],
    ]
```

#### 响应格式 (Response Format)

该参数控制智能体如何返回结构化数据：

- **`ToolStrategy[StructuredResponseT]`**: 使用 **工具调用（tool calling）** 实现结构化输出。
- **`ProviderStrategy[StructuredResponseT]`**: 使用 **提供商原生（provider-native）** 的结构化输出功能。
- **`type[StructuredResponseT]`**: **模式类型（Schema type）** - 会根据模型功能自动选择最佳策略。
- **`None`**: 不进行结构化输出。

当直接提供模式类型时，LangChain 会自动选择：

- **`ProviderStrategy`**: 适用于支持原生结构化输出的模型（例如 [OpenAI](https://langchain-doc.cn/v1/python/integrations/providers/openai)、[Grok](https://langchain-doc.cn/v1/python/integrations/providers/xai)）。
- **`ToolStrategy`**: 适用于所有其他模型。

结构化响应将在智能体最终状态的 **`structured_response`** 键中返回。

















