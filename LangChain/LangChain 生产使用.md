# LangChain 生产使用

## Studio

本指南将引导您了解如何使用 **Studio** 在本地可视化、交互和调试您的智能体（agent）。

Studio 是我们免费使用的强大智能体 IDE，它集成了 [LangSmith](https://langchain-doc.cn/langsmith/home)，可实现跟踪、评估和提示工程。您可以准确地看到您的智能体是如何思考的，跟踪每一个决策，并交付更智能、更可靠的智能体。

### 观看演示

<iframe width="560" height="315" src="https://www.youtube.com/embed/Mi1gSlHwZLM?si=zA47TNuTC5aH0ahd" title="Studio" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen=""></iframe>

### 先决条件

在开始之前，请确保您具备以下条件：

- [LangSmith](https://smith.langchain.com/settings) 的 API 密钥（**免费注册**）。

### 设置本地 LangGraph 服务器

#### 1. 安装 LangGraph CLI

```shell
# 需要 Python >= 3.11。
pip install --upgrade "langgraph-cli[inmem]"
```

#### 2. 准备您的智能体

我们将使用以下简单的智能体作为示例：

**agent.py**

```python
from langchain.agents import create_agent

def send_email(to: str, subject: str, body: str):
    """发送一封电子邮件"""
    email = {
        "to": to,
        "subject": subject,
        "body": body
    }
    # ... 邮件发送逻辑

    return f"Email sent to {to}"

agent = create_agent(
    "gpt-4o",
    tools=[send_email],
    system_prompt="You are an email assistant. Always use the send_email tool.",
)
```

#### 3. 环境变量

在项目的根目录下创建一个 `.env` 文件，并填写必要的 API 密钥。我们需要将 `LANGSMITH_API_KEY` 环境变量设置为您从 [LangSmith](https://smith.langchain.com/settings) 获取的 API 密钥。

> **⚠️ 警告：**
>
> 请务必不要将您的 `.env` 文件提交到像 Git 这样的版本控制系统！

```bash
LANGSMITH_API_KEY=lsv2...
```

#### 4. 创建 LangGraph 配置文件

在您的应用目录内，创建一个名为 `langgraph.json` 的配置文件：

**langgraph.json**

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent.py:agent"
  },
  "env": ".env"
}
```

[`create_agent`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)) 会自动返回一个已编译的 **LangGraph 图**，我们可以将其传递给配置文件的 `graphs` 键。

> **ℹ️ 信息：**
>
> 有关配置文件的 JSON 对象中每个键的详细解释，请参阅 [LangGraph 配置文件参考](https://langchain-doc.cn/langsmith/cli#configuration-file)。

到目前为止，我们的项目结构如下所示：

```bash
my-app/
├── src
│    └── agent.py
├── .env
└── langgraph.json
```

#### 5. 安装依赖项

在您的新 LangGraph 应用的根目录下，安装依赖项：

| `pip`                        | `uv`        |
| :--------------------------- | :---------- |
| `shell pip pip install -e .` | ```shell uv |
| uv sync                      |             |

~~~|
### 6. 在 Studio 中查看您的智能体

启动您的 LangGraph 服务器：

```shell
langgraph dev
~~~

> **⚠️ 警告：**
>
> Safari 会阻止到 Studio 的 `localhost` 连接。为了解决这个问题，请运行上述命令时带上 `--tunnel` 标志，以便通过安全隧道访问 Studio。

您的智能体将可通过 API (`http://127.0.0.1:2024`) 和 **Studio UI** (`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`) 访问：

### 在 Studio 中调试和交互

Studio 使您的智能体的每一步都**易于观察**。您可以重放任何输入并检查确切的提示、工具参数、返回值以及 **token/延迟指标**。如果工具抛出异常，Studio 会记录它和周围的状态，让您可以花更少的时间进行调试。

保持您的开发服务器运行，编辑提示或工具签名，并观察 Studio **热重载**。从任何步骤重新运行对话线程以验证行为更改。有关更多详细信息，请参阅 [管理线程](https://langchain-doc.cn/langsmith/use-studio#edit-thread-history)。

随着您的智能体的增长，同样的视图可以从单工具演示扩展到多节点图，保持决策的清晰和可重现。

> **💡 提示：**
>
> 要深入了解 Studio，请查看 [概述页面](https://langchain-doc.cn/langsmith/studio)。
>
> **💡 提示：**
>
> 有关本地和已部署智能体的更多信息，请参阅 [设置本地 LangGraph 服务器](https://langchain-doc.cn/v1/python/langchain/studio#setup-local-langgraph-server) 和 [部署](https://langchain-doc.cn/v1/python/langchain/deploy)。



## 测试

Agentic 应用程序允许 **大型语言模型（LLM）** 自行决定解决问题的下一步。这种灵活性非常强大，但模型的黑箱性质使得很难预测对 Agent 的某个部分进行的微调将如何影响其余部分。要构建可投入生产使用的 Agent，**彻底的测试**是必不可少的

### Agent 测试方法

测试您的 Agent 有以下几种方法：

- **[单元测试](https://langchain-doc.cn/v1/python/langchain/test.html#单元测试)**：使用内存中的**伪造（fake）**对象，在隔离状态下对 Agent 的小型、确定性部分进行测试，以便快速、确定性地断言（assert）准确的行为。
- **[集成测试](https://langchain-doc.cn/v1/python/langchain/test.html#集成测试)**：使用**真实的网络调用**来测试 Agent，以确认组件协同工作、凭证和模式（schemas）一致，并且延迟可接受。

Agentic 应用程序倾向于更多地依赖集成测试，因为它们将多个组件链接在一起，并且必须处理由于 LLM 的**非确定性**所带来的不稳定性。

### 模拟聊天模型 (Mocking Chat Model)

对于不需要 API 调用的逻辑，您可以使用内存中的**存根（stub）**来模拟响应。

LangChain 提供了 `GenericFakeChatModel` (https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.fake_chat_models.GenericFakeChatModel.html) 用于模拟文本响应。它接受一个响应（AIMessages 或字符串）的迭代器，并在每次调用时返回一个响应。它支持常规用法和流式传输用法。

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

model = GenericFakeChatModel(messages=iter([
    AIMessage(content="", tool_calls=[ToolCall(name="foo", args={"bar": "baz"}, id="call_1")]),
    "bar"
]))

model.invoke("hello")
# AIMessage(content='', ..., tool_calls=[{'name': 'foo', 'args': {'bar': 'baz'}, 'id': 'call_1', 'type': 'tool_call'}])
```

如果我们再次调用该模型，它将返回迭代器中的下一个项目：

```python
model.invoke("hello, again!")
# AIMessage(content='bar', ...)
```

### InMemorySaver 检查点 (InMemorySaver Checkpointer)

为了在测试期间启用**持久性（persistence）**，您可以使用 `InMemorySaver` (https://reference.langchain.com/python/langgraph/checkpoints/#langgraph.checkpoint.memory.InMemorySaver) 检查点。这允许您模拟多次对话轮次，以测试**依赖于状态**的行为：

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model,
    tools=[],
    checkpointer=InMemorySaver()
)

# 第一次调用
agent.invoke(HumanMessage(content="I live in Sydney, Australia."))

# 第二次调用：第一条消息被持久化（悉尼位置），因此模型返回 GMT+10 时间
agent.invoke(HumanMessage(content="What's my local time?"))
```

### 集成测试 (Integration Testing)

许多 Agent 行为只有在使用**真实 LLM**时才会出现，例如 Agent 决定调用哪个工具、如何格式化响应，或者提示修改是否会影响整个执行轨迹。LangChain 的 `agentevals` (https://github.com/langchain-ai/agentevals) 包提供了专门用于测试带有真实模型的 Agent 轨迹的评估器。

AgentEvals 让您可以通过执行**轨迹匹配（trajectory match）**或使用 **LLM 评判（LLM judge）**来轻松评估 Agent 的轨迹（消息的确切序列，包括工具调用）：

| 评估器       | 描述                                                         | 适用场景                                                     |
| :----------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **轨迹匹配** | 为给定的输入**硬编码**一个参考轨迹，并通过**逐步比较**来验证运行结果。 | **理想用于测试定义明确的工作流**，您知道预期的行为。当您对应该调用哪些工具以及调用顺序有**特定期望**时使用。此方法是**确定性**的、**快速**且**经济高效**，因为它不需要额外的 LLM 调用。 |
| **LLM 评判** | 使用 **LLM** 来定性验证 Agent 的执行轨迹。这个“评判”LLM 根据**提示词准则**（可以包含参考轨迹）来审查 Agent 的决策。 | **更灵活**，可以评估效率和适当性等**细微方面**，但需要进行 LLM 调用，且**非确定性**较低。当您想要评估 Agent 轨迹的**整体质量和合理性**，而无需严格的工具调用或排序要求时使用。 |

### 安装 AgentEvals

```bash
pip install agentevals
```

或者，直接克隆 [AgentEvals 仓库](https://github.com/langchain-ai/agentevals)。

### 轨迹匹配评估器 (Trajectory Match Evaluator)

AgentEvals 提供了 `create_trajectory_match_evaluator` 函数，用于将您的 Agent 轨迹与参考轨迹进行匹配。有四种模式可供选择：

| 模式               | 描述                                                         | 用例                                     |
| :----------------- | :----------------------------------------------------------- | :--------------------------------------- |
| `strict` (严格)    | 消息和工具调用**按相同顺序**精确匹配                         | 测试特定的序列（例如，授权前的策略查询） |
| `unordered` (无序) | 允许相同的工具调用**以任何顺序**出现                         | 验证信息检索时**顺序不重要**的情况       |
| `subset` (子集)    | Agent 调用的工具**仅限于**参考轨迹中的工具（不允许额外调用） | 确保 Agent 不**超出预期范围**            |
| `superset` (超集)  | Agent 调用的工具**至少包含**参考轨迹中的工具（允许额外调用） | 验证已执行**最低要求的操作**             |

#### 严格匹配 (Strict match)

`strict` 模式确保轨迹包含**相同的消息**、**相同的工具调用**，并**按相同的顺序**，尽管它允许消息内容存在差异。当您需要强制执行特定的操作序列时（例如，在授权操作之前需要进行策略查询），此功能非常有用。

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator


@tool
def get_weather(city: str):
    """Get weather information for a city."""
    return f"It's 75 degrees and sunny in {city}."

agent = create_agent("gpt-4o", tools=[get_weather])

evaluator = create_trajectory_match_evaluator(  # [!code highlight]
    trajectory_match_mode="strict",  # [!code highlight]
)  # [!code highlight]

def test_weather_tool_called_strict():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in San Francisco?")]
    })

    reference_trajectory = [
        HumanMessage(content="What's the weather in San Francisco?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "San Francisco"}}
        ]),
        ToolMessage(content="It's 75 degrees and sunny in San Francisco.", tool_call_id="call_1"),
        AIMessage(content="The weather in San Francisco is 75 degrees and sunny."),
    ]

    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory
    )
    # {
    #     'key': 'trajectory_strict_match',
    #     'score': True,
    #     'comment': None,
    # }
    assert evaluation["score"] is True
```

#### 无序匹配 (Unordered match)

`unordered` 模式允许**相同的工具调用以任何顺序**出现，当您想要验证是否检索了特定信息但**不关心顺序**时，此模式很有帮助。例如，Agent 可能需要检查城市的天气和活动，但顺序无关紧要。

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator


@tool
def get_weather(city: str):
    """Get weather information for a city."""
    return f"It's 75 degrees and sunny in {city}."

@tool
def get_events(city: str):
    """Get events happening in a city."""
    return f"Concert at the park in {city} tonight."

agent = create_agent("gpt-4o", tools=[get_weather, get_events])

evaluator = create_trajectory_match_evaluator(  # [!code highlight]
    trajectory_match_mode="unordered",  # [!code highlight]
)  # [!code highlight]

def test_multiple_tools_any_order():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's happening in SF today?")]
    })

    # 参考轨迹中工具调用的顺序与实际执行的顺序不同
    reference_trajectory = [
        HumanMessage(content="What's happening in SF today?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_events", "args": {"city": "SF"}},
            {"id": "call_2", "name": "get_weather", "args": {"city": "SF"}},
        ]),
        ToolMessage(content="Concert at the park in SF tonight.", tool_call_id="call_1"),
        ToolMessage(content="It's 75 degrees and sunny in SF.", tool_call_id="call_2"),
        AIMessage(content="Today in SF: 75 degrees and sunny with a concert at the park tonight."),
    ]

    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory,
    )
    # {
    #     'key': 'trajectory_unordered_match',
    #     'score': True,
    # }
    assert evaluation["score"] is True
```

#### 子集和超集匹配 (Subset and superset match)

`superset` 和 `subset` 模式匹配**部分轨迹**。`superset` 模式验证 Agent **至少调用了参考轨迹中的工具**，允许有额外的工具调用。`subset` 模式确保 Agent **没有调用超出参考轨迹中工具的任何工具**。

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator


@tool
def get_weather(city: str):
    """Get weather information for a city."""
    return f"It's 75 degrees and sunny in {city}."

@tool
def get_detailed_forecast(city: str):
    """Get detailed weather forecast for a city."""
    return f"Detailed forecast for {city}: sunny all week."

agent = create_agent("gpt-4o", tools=[get_weather, get_detailed_forecast])

evaluator = create_trajectory_match_evaluator(  # [!code highlight]
    trajectory_match_mode="superset",  # [!code highlight]
)  # [!code highlight]

def test_agent_calls_required_tools_plus_extra():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in Boston?")]
    })

    # 参考轨迹仅要求 get_weather，但 Agent 可能会调用额外的工具
    reference_trajectory = [
        HumanMessage(content="What's the weather in Boston?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "Boston"}},
        ]),
        ToolMessage(content="It's 75 degrees and sunny in Boston.", tool_call_id="call_1"),
        AIMessage(content="The weather in Boston is 75 degrees and sunny."),
    ]

    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory,
    )
    # {
    #     'key': 'trajectory_superset_match',
    #     'score': True,
    #     'comment': None,
    # }
    assert evaluation["score"] is True
```

> 您还可以设置 `tool_args_match_mode` 属性和/或 `tool_args_match_overrides`，以自定义评估器如何考虑实际轨迹与参考轨迹中工具调用之间的相等性。默认情况下，只有**具有相同参数的相同工具调用**才被视为相等。请访问 [仓库](https://github.com/langchain-ai/agentevals?tab=readme-ov-file#tool-args-match-modes) 了解更多详细信息。

### LLM 评判评估器 (LLM-as-Judge Evaluator)

您还可以使用 `create_trajectory_llm_as_judge` 函数，用 LLM 来评估 Agent 的执行路径。与轨迹匹配评估器不同，它**不需要参考轨迹**，但如果可用，也可以提供。

#### 不带参考轨迹

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT


@tool
def get_weather(city: str):
    """Get weather information for a city."""
    return f"It's 75 degrees and sunny in {city}."

agent = create_agent("gpt-4o", tools=[get_weather])

evaluator = create_trajectory_llm_as_judge(  # [!code highlight]
    model="openai:o3-mini",  # [!code highlight]
    prompt=TRAJECTORY_ACCURACY_PROMPT,  # [!code highlight]
)  # [!code highlight]

def test_trajectory_quality():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in Seattle?")]
    })

    evaluation = evaluator(
        outputs=result["messages"],
    )
    # {
    #     'key': 'trajectory_accuracy',
    #     'score': True,
    #     'comment': 'The provided agent trajectory is reasonable...'
    # }
    assert evaluation["score"] is True
```

#### 带参考轨迹

如果您有参考轨迹，可以在提示词中添加一个额外的变量，并传入参考轨迹。下面，我们使用预构建的 `TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE` 提示词，并配置 `reference_outputs` 变量：

```python
evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
)
evaluation = judge_with_reference(
    outputs=result["messages"],
    reference_outputs=reference_trajectory,
)
```

> 有关如何让 LLM 评估轨迹的更多可配置性，请访问 [仓库](https://github.com/langchain-ai/agentevals?tab=readme-ov-file#trajectory-llm-as-judge)。

### 异步支持 (Async Support)

所有 `agentevals` 评估器都支持 Python **asyncio**。对于使用工厂函数的评估器，可以通过在函数名中的 `create_` 之后添加 `async` 来获得异步版本。

#### 异步评判和评估器示例

```python
from agentevals.trajectory.llm import create_async_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT
from agentevals.trajectory.match import create_async_trajectory_match_evaluator

async_judge = create_async_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

async_evaluator = create_async_trajectory_match_evaluator(
    trajectory_match_mode="strict",
)

async def test_async_evaluation():
    result = await agent.ainvoke({
        "messages": [HumanMessage(content="What's the weather?")]
    })

    evaluation = await async_judge(outputs=result["messages"])
    assert evaluation["score"] is True
```

### LangSmith 集成 (LangSmith Integration)

为了跟踪随时间变化的实验，您可以将评估器结果记录到 [LangSmith](https://smith.langchain.com/)，这是一个用于构建生产级 LLM 应用程序的平台，其中包括**追踪（tracing）**、**评估（evaluation）和实验工具**。

首先，通过设置所需的环境变量来配置 LangSmith：

```bash
export LANGSMITH_API_KEY="your_langsmith_api_key"
export LANGSMITH_TRACING="true"
```

LangSmith 提供了两种主要的运行评估方法：[pytest 集成](https://langchain-doc.cn/langsmith/pytest)和 `evaluate` 函数。

#### 使用 pytest 集成

```python
import pytest
from langsmith import testing as t
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

trajectory_evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

@pytest.mark.langsmith
def test_trajectory_accuracy():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in SF?")]
    })

    reference_trajectory = [
        HumanMessage(content="What's the weather in SF?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "SF"}},
        ]),
        ToolMessage(content="It's 75 degrees and sunny in SF.", tool_call_id="call_1"),
        AIMessage(content="The weather in SF is 75 degrees and sunny."),
    ]

    # 将输入、输出和参考输出记录到 LangSmith
    t.log_inputs({})
    t.log_outputs({"messages": result["messages"]})
    t.log_reference_outputs({"messages": reference_trajectory})

    trajectory_evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory
    )
```

使用 pytest 运行评估：

```bash
pytest test_trajectory.py --langsmith-output
```

结果将自动记录到 LangSmith。

#### 使用 evaluate 函数

或者，您可以在 LangSmith 中创建一个数据集并使用 `evaluate` 函数：

```python
from langsmith import Client
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

client = Client()

trajectory_evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

def run_agent(inputs):
    """Your agent function that returns trajectory messages."""
    return agent.invoke(inputs)["messages"]

experiment_results = client.evaluate(
    run_agent,
    data="your_dataset_name",
    evaluators=[trajectory_evaluator]
)
```

结果将自动记录到 LangSmith。

> 💡 要了解有关评估 Agent 的更多信息，请参阅 [LangSmith 文档](https://langchain-doc.cn/langsmith/pytest)。

### 记录和重放 HTTP 调用 (Recording & Replaying HTTP Calls)

调用真实 LLM API 的集成测试可能**缓慢且昂贵**，尤其是在 CI/CD 管道中频繁运行时。我们建议使用一个库来**记录 HTTP 请求和响应**，然后在后续运行中**重放**它们，而无需进行实际的网络调用。

您可以使用 [`vcrpy`](https://langchain-doc.cn/v1/python/langchain/[https://pypi.org/project/vcrpy/1.5.2/](https://pypi.org/project/vcrpy/1.5.2/)) 来实现此目的。如果您使用的是 `pytest`，[`pytest-recording` 插件](https://langchain-doc.cn/v1/python/langchain/[https:/pypi.org/project/pytest-recording/)提供了一种简单的](https://pypi.org/project/pytest-recording/)提供了一种简单的)、只需少量配置即可启用此功能的方法。请求/响应被记录在**磁带（cassettes）**中，然后用于在后续运行中模拟真实的网络调用。

设置您的 `conftest.py` 文件以从磁带中过滤掉敏感信息：

```python
import pytest

@pytest.fixture(scope="session")
def vcr_config():
    return {
        "filter_headers": [
            ("authorization", "XXXX"),
            ("x-api-key", "XXXX"),
            # ... 其他您想要屏蔽的 header
        ],
        "filter_query_parameters": [
            ("api_key", "XXXX"),
            ("key", "XXXX"),
        ],
    }
```

然后配置您的项目以识别 `vcr` 标记：

```ini
[pytest]
markers =
    vcr: record/replay HTTP via VCR
addopts = --record-mode=once
```

或

```toml
[tool.pytest.ini_options]
markers = [
    "vcr: record/replay HTTP via VCR"
]
addopts = "--record-mode=once"
```

> `--record-mode=once` 选项会在第一次运行时记录 HTTP 交互，并在后续运行中重放它们。

现在，只需使用 `vcr` 标记来装饰您的测试：

```python
@pytest.mark.vcr()
def test_agent_trajectory():
    # ...
```

第一次运行此测试时，您的 Agent 将进行**真实的网络调用**，pytest 将在 `tests/cassettes` 目录中生成一个磁带文件 `test_agent_trajectory.yaml`。后续运行将使用该磁带来**模拟真实的网络调用**，前提是 Agent 的请求没有从前一次运行中发生变化。如果发生变化，测试将失败，您将需要**删除磁带**并**重新运行测试**以记录新的交互。

> ⚠️ **警告**
>
> 当您修改提示词、添加新工具或更改预期轨迹时，您保存的磁带将**过时**，并且您现有的测试将**失败**。您应该删除相应的磁带文件并重新运行测试以记录新的交互。



## 部署

### 部署

LangSmith 是将 Agent 转化为生产系统的**最快方式**。传统的托管平台是为**无状态、短生命周期**的 Web 应用而构建的，而 LangGraph 则是**专为有状态、长生命周期的 Agent** 而设计，因此您可以在几分钟内完成从代码仓库到可靠的云部署。

### 先决条件

在开始之前，请确保您具备以下条件：

- 一个 [GitHub 账户](https://github.com/)
- 一个 [LangSmith 账户](https://smith.langchain.com/)（免费注册）

### 部署您的 Agent

#### 1. 在 GitHub 上创建一个代码仓库

您的应用程序代码必须位于 GitHub 仓库中才能部署到 LangSmith 上。LangSmith 支持公共和私有仓库。对于本快速入门，请首先按照 [本地服务器设置指南](https://langchain-doc.cn/v1/python/langchain/studio#setup-local-langgraph-server) 确保您的应用程序与 LangGraph 兼容。然后，将您的代码推送到该仓库。



## Agent 聊天用户界面（todo）



## 可观测性(todo)

可观测性对于理解您的 Agent 在生产环境中的行为**至关重要**。通过 LangChain 的 [`create_agent`](https://langchain-doc.cn/v1/python/langchain/[https:/reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent))，您可以获得通过 [LangSmith](https://smith.langchain.com/) 提供的**内置可观测性**——这是一个用于追踪、调试、评估和监控您的 LLM 应用程序的强大平台。

追踪（Traces）会捕获您的 Agent 所采取的每一步，从最初的用户输入到最终响应，包括所有工具调用、模型交互和决策点。这使您能够**调试 Agent**、**评估性能**和**监控使用情况**。

### 先决条件

在开始之前，请确保您具备以下条件：

- 一个 [LangSmith 账户](https://smith.langchain.com/)（免费注册）

### 启用追踪 (Enable tracing)

所有 LangChain Agent 都**自动支持** LangSmith 追踪。要启用它，请设置以下环境变量：

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
```

> ℹ️ 您可以从您的 [LangSmith 设置](https://smith.langchain.com/settings) 中获取您的 API 密钥。

### 快速开始

将追踪日志记录到 LangSmith **无需额外的代码**。只需像往常一样运行您的 Agent 代码即可：

```python
from langchain.agents import create_agent


def send_email(to: str, subject: str, body: str):
    """Send an email to a recipient."""
    # ... 电子邮件发送逻辑
    return f"Email sent to {to}"

def search_web(query: str):
    """Search the web for information."""
    # ... 网络搜索逻辑
    return f"Search results for: {query}"

agent = create_agent(
    model="gpt-4o",
    tools=[send_email, search_web],
    system_prompt="You are a helpful assistant that can send emails and search the web."
)

# 运行 Agent - 所有步骤都将自动被追踪
response = agent.invoke({
    "messages": [{"role": "user", "content": "Search for the latest AI news and email a summary to john@example.com"}]
})
```

默认情况下，追踪将记录到项目名称为 `default` 的项目下。要配置自定义项目名称，请参阅 [记录到项目](https://langchain-doc.cn/v1/python/langchain/observability.html#记录到项目)。

### 选择性追踪 (Trace selectively)

您可以选择使用 LangSmith 的 `tracing_context` 上下文管理器来**仅追踪**应用程序的特定调用或部分：

```python
import langsmith as ls

# 这将**被追踪**
with ls.tracing_context(enabled=True):
    agent.invoke({"messages": [{"role": "user", "content": "Send a test email to alice@example.com"}]})

# 这将**不被追踪**（如果 LANGSMITH_TRACING 未设置）
agent.invoke({"messages": [{"role": "user", "content": "Send another email"}]})
```

### 记录到项目 (Log to a project)

#### 静态设置 (Statically)

您可以通过设置 `LANGSMITH_PROJECT` 环境变量来为您的整个应用程序设置一个**自定义项目名称**：

```bash
export LANGSMITH_PROJECT=my-agent-project
```

#### 动态设置 (Dynamically)

您可以为特定操作**以编程方式**设置项目名称：

```python
import langsmith as ls

with ls.tracing_context(project_name="email-agent-test", enabled=True):
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Send a welcome email"}]
    })
```

### 向追踪添加元数据 (Add metadata to traces)

您可以使用**自定义元数据**和**标签**来注释您的追踪：

```python
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Send a welcome email"}]},
    config={
        "tags": ["production", "email-assistant", "v1.0"],
        "metadata": {
            "user_id": "user_123",
            "session_id": "session_456",
            "environment": "production"
        }
    }
)
```

`tracing_context` 也接受标签和元数据以进行细粒度控制：

```python
with ls.tracing_context(
    project_name="email-agent-test",
    enabled=True,
    tags=["production", "email-assistant", "v1.0"],
    metadata={"user_id": "user_123", "session_id": "session_456", "environment": "production"}):
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Send a welcome email"}]}
    )
```

这些自定义元数据和标签将附加到 LangSmith 中的追踪上。

> 💡 要了解更多关于如何使用追踪来调试、评估和监控您的 Agent，请参阅 [LangSmith 文档](https://langchain-doc.cn/langsmith/home)。















