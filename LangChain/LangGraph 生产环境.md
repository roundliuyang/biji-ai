# LangGraph 生产环境

## 应用程序结构

LangGraph应用程序由一个或多个图、一个配置文件（`langgraph.json`）、一个指定依赖项的文件和一个可选的指定环境变量的`.env`文件组成。

本指南展示了应用程序的典型结构，并说明如何指定使用LangSmith部署应用程序所需的信息。

### 关键概念

要使用LangSmith进行部署，应提供以下信息：

1. 一个[LangGraph配置文件](https://langchain-doc.cn/v1/python/langgraph/application-structure.html#configuration-file-concepts)（`langgraph.json`），用于指定应用程序使用的依赖项、图和环境变量。
2. 实现应用程序逻辑的[图](https://langchain-doc.cn/v1/python/langgraph/application-structure.html#graphs)。
3. 一个指定[依赖项](https://langchain-doc.cn/v1/python/langgraph/application-structure.html#dependencies)的文件，这些依赖项是运行应用程序所必需的。
4. [环境变量](https://langchain-doc.cn/v1/python/langgraph/application-structure.html#environment-variables)，这些环境变量是应用程序运行所必需的。

### 文件结构

以下是应用程序的目录结构示例：

#### Python (requirements.txt)

```plaintext
my-app/
├── my_agent # 所有项目代码都在这里
│   ├── utils # 图的实用工具
│   │   ├── __init__.py
│   │   ├── tools.py # 图的工具
│   │   ├── nodes.py # 图的节点函数
│   │   └── state.py # 图的状态定义
│   ├── __init__.py
│   └── agent.py # 构建图的代码
├── .env # 环境变量
├── requirements.txt # 包依赖项
└── langgraph.json # LangGraph的配置文件
```

#### Python (pyproject.toml)

```plaintext
my-app/
├── my_agent # 所有项目代码都在这里
│   ├── utils # 图的实用工具
│   │   ├── __init__.py
│   │   ├── tools.py # 图的工具
│   │   ├── nodes.py # 图的节点函数
│   │   └── state.py # 图的状态定义
│   ├── __init__.py
│   └── agent.py # 构建图的代码
├── .env # 环境变量
├── langgraph.json  # LangGraph的配置文件
└── pyproject.toml # 项目依赖项
```

#### JavaScript

```plaintext
my-app/
├── src # 所有项目代码都在这里
│   ├── utils # 图的可选实用工具
│   │   ├── tools.ts # 图的工具
│   │   ├── nodes.ts # 图的节点函数
│   │   └── state.ts # 图的状态定义
│   └── agent.ts # 构建图的代码
├── package.json # 包依赖项
├── .env # 环境变量
└── langgraph.json # LangGraph的配置文件
```

**注意：** LangGraph应用程序的目录结构可能会根据使用的编程语言和包管理器而有所不同。

### 配置文件

`langgraph.json`文件是一个JSON文件，用于指定部署LangGraph应用程序所需的依赖项、图、环境变量和其他设置。

有关JSON文件中所有支持的键的详细信息，请参阅[LangGraph配置文件参考](https://langchain-doc.cn/langsmith/cli#configuration-file)。

**提示：** [LangGraph CLI](https://langchain-doc.cn/langsmith/cli)默认使用当前目录中的配置文件`langgraph.json`。

#### 示例

##### Python

- 依赖项包括一个自定义本地包和`langchain_openai`包。
- 单个图将从文件`./your_package/your_file.py`中加载，变量名为`variable`。
- 环境变量从`.env`文件加载。

```json
{
  "dependencies": ["langchain_openai", "./your_package"],
  "graphs": {
    "my_agent": "./your_package/your_file.py:agent"
  },
  "env": "./.env"
}
```

##### JavaScript

- 依赖项将从本地目录中的依赖文件加载（例如`package.json`）。
- 单个图将从文件`./your_package/your_file.js`中加载，函数名为`agent`。
- 环境变量`OPENAI_API_KEY`是内联设置的。

```json
{
  "dependencies": ["."],
  "graphs": {
    "my_agent": "./your_package/your_file.js:agent"
  },
  "env": {
    "OPENAI_API_KEY": "secret-key"
  }
}
```

### 依赖项

LangGraph应用程序可能依赖于其他Python包或TypeScript/JavaScript库。

要正确设置依赖项，通常需要指定以下信息：

1. 目录中指定依赖项的文件（例如`requirements.txt`、`pyproject.toml`或`package.json`）。
2. [LangGraph配置文件](https://langchain-doc.cn/v1/python/langgraph/application-structure.html#configuration-file-concepts)中的`dependencies`键，用于指定运行LangGraph应用程序所需的依赖项。
3. 任何其他二进制文件或系统库都可以使用[LangGraph配置文件](https://langchain-doc.cn/v1/python/langgraph/application-structure.html#configuration-file-concepts)中的`dockerfile_lines`键来指定。

### 图

使用[LangGraph配置文件](https://langchain-doc.cn/v1/python/langgraph/application-structure.html#configuration-file-concepts)中的`graphs`键来指定在部署的LangGraph应用程序中可用的图。

你可以在配置文件中指定一个或多个图。每个图由一个名称（应该是唯一的）和以下路径之一标识：(1) 编译后的图，或(2) 定义创建图的函数。

### 环境变量

如果你在本地使用已部署的LangGraph应用程序，可以在[LangGraph配置文件](https://langchain-doc.cn/v1/python/langgraph/application-structure.html#configuration-file-concepts)的`env`键中配置环境变量。

对于生产部署，你通常希望在部署环境中配置环境变量。



## Studio

本指南将指导你如何使用 **Studio** 在本地可视化、交互和调试你的 Agent。

Studio 是我们免费、强大的 Agent IDE，它集成了 [LangSmith](https://langchain-doc.cn/langsmith/home)，可以进行追踪（tracing）、评估（evaluation）和提示工程（prompt engineering）。你可以准确地看到你的 Agent 是如何思考的，追踪每一个决策，并发布更智能、更可靠的 Agent。

### 前提条件 

在你开始之前，请确保你拥有以下条件：

- 一个 [LangSmith](https://smith.langchain.com/settings) 的 API 密钥（可免费注册）

### 设置本地 LangGraph 服务器

#### 1. 安装 LangGraph CLI

```shell
# 需要 Python >= 3.11。
pip install --upgrade "langgraph-cli[inmem]"
```

#### 2. 准备你的 Agent

我们将使用以下简单的 Agent 作为示例：

agent.py

```python
from langchain.agents import create_agent

def send_email(to: str, subject: str, body: str):
    """发送一封邮件"""
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

在项目的根目录下创建一个 **`.env`** 文件，并填入必要的 API 密钥。我们需要将 `LANGSMITH_API_KEY` 环境变量设置为你从 [LangSmith](https://smith.langchain.com/settings) 获取的 API 密钥。

> **警告**
> 请确保不要将你的 **`.env`** 文件提交到 Git 等版本控制系统！

```bash
LANGSMITH_API_KEY=lsv2...
```

#### 4. 创建 LangGraph 配置文件

在你的应用目录下，创建一个名为 `langgraph.json` 的配置文件：

langgraph.json

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent.py:agent"
  },
  "env": ".env"
}
```

[`create_agent`](https://langchain-doc.cn/v1/python/langgraph/[https:/reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)) 会自动返回一个已编译的 LangGraph 图，我们可以将其传递给配置文件的 `graphs` 键。

> **信息**
> 有关配置文件的 JSON 对象中每个键的详细解释，请参阅 [LangGraph 配置文件参考](https://langchain-doc.cn/langsmith/cli#configuration-file)。

到目前为止，我们的项目结构如下所示：

```bash
my-app/
├── src
│   └── agent.py
├── .env
└── langgraph.json
```

#### 5. 安装依赖项

在你的新 LangGraph 应用的根目录下，安装依赖项：

```shell
pip install -e .
```

或

```shell
uv sync
```

#### 6. 在 Studio 中查看你的 Agent

启动你的 LangGraph 服务器：

```shell
langgraph dev
```

> **警告**
> Safari 浏览器会阻止对 Studio 的 `localhost` 连接。要解决此问题，请运行上述命令并加上 `--tunnel` 选项，以通过安全隧道访问 Studio。

你的 Agent 将可以通过 API (`http://127.0.0.1:2024`) 和 Studio UI (`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`) 访问：

Studio 使你的 Agent 的每一步都易于观察。你可以重放任何输入，并检查确切的提示、工具参数、返回值以及 Token/延迟指标。如果工具抛出异常，Studio 会记录它和周围的状态，这样你就可以减少调试时间。

保持你的开发服务器运行，编辑提示或工具签名，并观察 Studio **热重载**。你可以从任何步骤重新运行对话线程，以验证行为更改。有关更多详细信息，请参阅 [管理线程](https://langchain-doc.cn/langsmith/use-studio#edit-thread-history)。

随着你的 Agent 的发展，同样的视图可以从单工具演示扩展到多节点图，保持决策清晰和可复现。

> **提示**
> 要深入了解 Studio，请查看[概览页面](https://langchain-doc.cn/langsmith/studio)。



## 测试（todo）



## 部署

LangSmith是将代理转变为生产系统的最快方式。传统的托管平台是为无状态、短期运行的Web应用程序构建的，而LangGraph是**专为有状态、长期运行的代理而设计的**，因此你可以在几分钟内从代码库转到可靠的云部署。

### 先决条件

在开始之前，请确保你具备以下条件：

- 一个[GitHub账户](https://github.com/)
- 一个[LangSmith账户](https://smith.langchain.com/)（免费注册）

### 部署你的代理

#### 1. 在GitHub上创建仓库

你的应用程序代码必须位于GitHub仓库中才能在LangSmith上部署。支持公共和私有仓库。对于这个快速入门，首先确保你的应用程序与LangGraph兼容，方法是按照[本地服务器设置指南](https://langchain-doc.cn/v1/python/langgraph/studio#setup-local-langgraph-server)操作。然后，将你的代码推送到仓库。

> 注意：文档中引用的部署步骤部分因系统限制无法显示，请参考LangSmith官方文档获取完整的部署指南。



## Agent聊天用户界面(todo)



## 可观察性

追踪（Traces）是你的应用程序从输入到输出所采取的一系列步骤。这些单个步骤中的每一个都由一个运行（run）表示。你可以使用[LangSmith](https://smith.langchain.com/)来可视化这些执行步骤。要使用它，请[为你的应用程序启用追踪](https://langchain-doc.cn/langsmith/trace-with-langgraph)。这使你能够执行以下操作：

- [调试本地运行的应用程序](https://langchain-doc.cn/langsmith/observability-studio#debug-langsmith-traces)。
- [评估应用程序性能](https://langchain-doc.cn/v1/python/langchain/evals)。
- [监控应用程序](https://langchain-doc.cn/langsmith/dashboards)。

### 先决条件

在开始之前，请确保你具备以下条件：

- 一个[LangSmith账户](https://smith.langchain.com/)（免费注册）

### 启用追踪

要为你的应用程序启用追踪，请设置以下环境变量：

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
```

默认情况下，追踪将记录到名为`default`的项目中。要配置自定义项目名称，请参阅[记录到项目](https://langchain-doc.cn/v1/python/langgraph/observability.html#log-to-a-project)。

有关更多信息，请参阅[使用LangGraph进行追踪](https://langchain-doc.cn/langsmith/trace-with-langgraph)。

> 注意：文档中引用的部分内容因系统限制无法显示，请参考LangSmith官方文档获取完整的可观察性指南。

### 使用匿名器防止敏感数据在追踪中记录

你可能希望屏蔽敏感数据，以防止其被记录到LangSmith中。你可以创建[匿名器](https://langchain-doc.cn/langsmith/mask-inputs-outputs#rule-based-masking-of-inputs-and-outputs)并通过配置将其应用到你的图中。以下示例将从发送到LangSmith的追踪中编辑任何匹配社会安全号码格式XXX-XX-XXXX的内容。

```python
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.graph import StateGraph, MessagesState
from langsmith import Client
from langsmith.anonymizer import create_anonymizer

anonymizer = create_anonymizer([
    # 匹配SSN
    { "pattern": r"\b\d{3}-?\d{2}-?\d{4}\b", "replace": "<ssn>" }
])

tracer_client = Client(anonymizer=anonymizer)
tracer = LangChainTracer(client=tracer_client)
# 定义图
graph = (
    StateGraph(MessagesState)
    ...
    .compile()
    .with_config({'callbacks': [tracer]})
)
```



```typescript
import { StateGraph } from "@langchain/langgraph";
import { LangChainTracer } from "@langchain/core/tracers/tracer_langchain";
import { StateAnnotation } from "./state.js";
import { createAnonymizer } from "langsmith/anonymizer"
import { Client } from "langsmith"


const anonymizer = createAnonymizer([
    // 匹配SSN
    { pattern: /\b\d{3}-?\d{2}-?\d{4}\b/, replace: "<ssn>" }
])

const langsmithClient = new Client({ anonymizer })
const tracer = new LangChainTracer({
  client: langsmithClient,
});

export const graph = new StateGraph(StateAnnotation)
  .compile()
  .withConfig({
    callbacks: [tracer],
});
```









