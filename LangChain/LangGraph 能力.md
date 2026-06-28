# LangGraph 能力

## 持久化

LangGraph 支持图状态的持久化和重放。在本文档中，我们将探讨如何通过检查点（checkpointing）来持久化图状态，以及如何利用这些检查点执行图的重放和状态更新。

### 检查点

检查点是图状态的快照，在图的每个超级步骤（superstep）执行后保存。每个检查点都与一个**线程（thread）**相关联，线程是检查点的集合。

#### 线程

线程是与特定执行实例相关联的检查点集合。线程通过 `thread_id` 进行标识，这是一个字符串，可以是您想要的任何内容（例如会话 ID、用户 ID、对话 ID 等）。

当您调用图时，可以指定 `thread_id` 作为图配置的一部分。如果您不提供 `thread_id`，LangGraph 会自动生成一个。

要配置 `thread_id`，您需要将其作为 `configurable` 部分的一部分传递给图调用：

```python
config = {"configurable": {"thread_id": "1"}}
graph.invoke(..., config=config)
```

```typescript
const config = { configurable: { thread_id: "1" } };
await graph.invoke(..., { config });
```

#### 检查点结构

每个检查点都是一个 `StateSnapshot` 对象，其中包含：

- `values`: 图状态的值
- `next`: 下一个要执行的节点
- `config`: 检查点配置，包括 `thread_id` 和 `checkpoint_id`
- `metadata`: 关于检查点的元数据，如源节点和写入的内容
- `created_at`: 创建时间
- `parent_config`: 父检查点的配置
- `tasks`: 与检查点相关的任务

#### 示例：检查点的保存

让我们通过一个简单的例子来说明检查点是如何保存的。首先，我们需要创建一个图并启用检查点：
```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

class State(TypedDict):
    foo: str
    bar: list[str]

def node_a(state: State) -> State:
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State) -> State:
    return {"foo": "b", "bar": ["b"]}

# 创建图
graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)

# 设置边
graph.add_edge("__start__", "node_a")
graph.add_edge("node_a", "node_b")

# 创建检查点
saver = InMemorySaver()

# 编译图时启用检查点
graph = graph.compile(checkpointer=saver)
```

```typescript
import { StateGraph } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph/checkpoint";
import { z } from "zod";

const State = z.object({
  foo: z.string(),
  bar: z.array(z.string()),
});

function nodeA(state: z.infer<typeof State>) {
  return { foo: "a", bar: ["a"] };
}

function nodeB(state: z.infer<typeof State>) {
  return { foo: "b", bar: ["b"] };
}

// 创建图
const workflow = new StateGraph(State);
workflow.addNode("nodeA", nodeA);
workflow.addNode("nodeB", nodeB);

// 设置边
workflow.addEdge("__start__", "nodeA");
workflow.addEdge("nodeA", "nodeB");

// 创建检查点
const saver = new MemorySaver();

// 编译图时启用检查点
const graph = workflow.compile({ checkpointer: saver });
```

现在，让我们调用这个图，指定一个 `thread_id`：

```python
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": "", "bar": []}, config=config)
```

```typescript
const config = { configurable: { thread_id: "1" } };
await graph.invoke({ foo: "", bar: [] }, { config });
```

当我们调用图时，LangGraph 会在每个超级步骤后保存检查点。在这个例子中，它会保存以下检查点：

1. 初始状态（在调用 `__start__` 之前）
2. 调用 `node_a` 后的状态
3. 调用 `node_b` 后的状态

#### 获取状态

我们可以使用 `get_state` 方法获取特定检查点的状态。这可以通过指定 `thread_id` 和可选的 `checkpoint_id` 来完成：

```python
# 获取最新的状态快照
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)

# 获取特定检查点 ID 的状态快照
config = {
  "configurable": {
    "thread_id": "1",
    "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c",
  },
}
graph.get_state(config)
```

```typescript
// 获取最新的状态快照
const config = { configurable: { thread_id: "1" } };
await graph.getState(config);

// 获取特定检查点 ID 的状态快照
const config = {
  configurable: {
    thread_id: "1",
    checkpoint_id: "1ef663ba-28fe-6528-8002-5a559208592c",
  },
};
await graph.getState(config);
```

在我们的例子中，`get_state` 的输出将如下所示：

```
StateSnapshot(
    values={'foo': 'b', 'bar': ['a', 'b']},
    next=(),
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
    metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
    created_at='2024-08-29T19:19:38.821749+00:00',
    parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
    tasks=()
)
```

```
StateSnapshot {
  values: { foo: 'b', bar: ['a', 'b'] },
  next: [],
  config: {
    configurable: {
      thread_id: '1',
      checkpoint_ns: '',
      checkpoint_id: '1ef663ba-28fe-6528-8002-5a559208592c'
    }
  },
  metadata: {
    source: 'loop',
    writes: { nodeB: { foo: 'b', bar: ['b'] } },
    step: 2
  },
  createdAt: '2024-08-29T19:19:38.821749+00:00',
  parentConfig: {
    configurable: {
      thread_id: '1',
      checkpoint_ns: '',
      checkpoint_id: '1ef663ba-28f9-6ec4-8001-31981c2c39f8'
    }
  },
  tasks: []
}
```

#### 获取状态历史

您可以通过调用 `graph.get_state_history(config)` 来获取给定线程的图执行的完整历史。这将返回与配置中提供的线程 ID 关联的 `StateSnapshot` 对象列表。重要的是，检查点将按时间顺序排列，最近的检查点/`StateSnapshot` 是列表中的第一个。

```python
config = {"configurable": {"thread_id": "1"}}
list(graph.get_state_history(config))
```

```typescript
const config = { configurable: { thread_id: "1" } };
for await (const state of graph.getStateHistory(config)) {
  console.log(state);
}
```

在我们的例子中，`get_state_history` 的输出将如下所示：

```
[
    StateSnapshot(
        values={'foo': 'b', 'bar': ['a', 'b']},
        next=(),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
        metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
        created_at='2024-08-29T19:19:38.821749+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
        tasks=(),
    ),
    StateSnapshot(
        values={'foo': 'a', 'bar': ['a']},
        next=('node_b',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
        metadata={'source': 'loop', 'writes': {'node_a': {'foo': 'a', 'bar': ['a']}}, 'step': 1},
        created_at='2024-08-29T19:19:38.819946+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
        tasks=(PregelTask(id='6fb7314f-f114-5413-a1f3-d37dfe98ff44', name='node_b', error=None, interrupts=()),),
    ),
    StateSnapshot(
        values={'foo': '', 'bar': []},
        next=('node_a',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
        metadata={'source': 'loop', 'writes': None, 'step': 0},
        created_at='2024-08-29T19:19:38.817813+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
        tasks=(PregelTask(id='f1b14528-5ee5-579c-949b-23ef9bfbed58', name='node_a', error=None, interrupts=()),),
    ),
    StateSnapshot(
        values={'bar': []},
        next=('__start__',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
        metadata={'source': 'input', 'writes': {'foo': ''}, 'step': -1},
        created_at='2024-08-29T19:19:38.816205+00:00',
        parent_config=None,
        tasks=(PregelTask(id='6d27aa2e-d72b-5504-a36f-8620e54a76dd', name='__start__', error=None, interrupts=()),),
    )
]
```

```
[
  StateSnapshot {
    values: { foo: 'b', bar: ['a', 'b'] },
    next: [],
    config: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28fe-6528-8002-5a559208592c'
      }
    },
    metadata: {
      source: 'loop',
      writes: { nodeB: { foo: 'b', bar: ['b'] } },
      step: 2
    },
    createdAt: '2024-08-29T19:19:38.821749+00:00',
    parentConfig: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28f9-6ec4-8001-31981c2c39f8'
      }
    },
    tasks: []
  },
  StateSnapshot {
    values: { foo: 'a', bar: ['a'] },
    next: ['nodeB'],
    config: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28f9-6ec4-8001-31981c2c39f8'
      }
    },
    metadata: {
      source: 'loop',
      writes: { nodeA: { foo: 'a', bar: ['a'] } },
      step: 1
    },
    createdAt: '2024-08-29T19:19:38.819946+00:00',
    parentConfig: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28f4-6b4a-8000-ca575a13d36a'
      }
    },
    tasks: [
      PregelTask {
        id: '6fb7314f-f114-5413-a1f3-d37dfe98ff44',
        name: 'nodeB',
        error: null,
        interrupts: []
      }
    ]
  },
  StateSnapshot {
    values: { foo: '', bar: [] },
    next: ['node_a'],
    config: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28f4-6b4a-8000-ca575a13d36a'
      }
    },
    metadata: {
      source: 'loop',
      writes: null,
      step: 0
    },
    createdAt: '2024-08-29T19:19:38.817813+00:00',
    parentConfig: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28f0-6c66-bfff-6723431e8481'
      }
    },
    tasks: [
      PregelTask {
        id: 'f1b14528-5ee5-579c-949b-23ef9bfbed58',
        name: 'node_a',
        error: null,
        interrupts: []
      }
    ]
  },
  StateSnapshot {
    values: { bar: [] },
    next: ['__start__'],
    config: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28f0-6c66-bfff-6723431e8481'
      }
    },
    metadata: {
      source: 'input',
      writes: { foo: '' },
      step: -1
    },
    createdAt: '2024-08-29T19:19:38.816205+00:00',
    parentConfig: null,
    tasks: [
      PregelTask {
        id: '6d27aa2e-d72b-5504-a36f-8620e54a76dd',
        name: '__start__',
        error: null,
        interrupts: []
      }
    ]
  }
]
```

#### 重放

也可以重放先前的图执行。如果我们使用 `thread_id` 和 `checkpoint_id` 调用图，我们将**重放**对应于 `checkpoint_id` 之前的检查点的先前执行步骤，并且只执行检查点之后的步骤。

- `thread_id` 是线程的 ID。
- `checkpoint_id` 是引用线程中特定检查点的标识符。

您必须在调用图时将这些作为配置的 `configurable` 部分传递：

```python
config = {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}
graph.invoke(None, config=config)
```

```typescript
const config = {
  configurable: {
    thread_id: "1",
    checkpoint_id: "0c62ca34-ac19-445d-bbb0-5b4984975b2a",
  },
};
await graph.invoke(null, config);
```

重要的是，LangGraph 知道特定步骤是否已经执行过。如果已经执行过，LangGraph 只会**重放**图中的该特定步骤，而不会重新执行该步骤，但这仅适用于提供的 `checkpoint_id` 之前的步骤。`checkpoint_id` 之后的所有步骤都将执行（即新分支），即使它们之前已经执行过。

#### 更新状态

除了从特定的 `checkpoint` 重放图外，我们还可以**编辑**图状态。我们使用 `update_state` 来做到这一点。此方法接受三个不同的参数：

##### config

配置应包含指定要更新哪个线程的 `thread_id`。当只传递 `thread_id` 时，我们更新（或分支）当前状态。可选地，如果我们包含 `checkpoint_id` 字段，那么我们分支所选的检查点。

##### values

这些是将用于更新状态的值。请注意，此更新的处理方式与来自节点的任何更新完全相同。这意味着这些值将传递给图状态中为某些通道定义的 [reducer](https://langchain-doc.cn/v1/python/langgraph/graph-api#reducers) 函数。这意味着 `update_state` 不会自动覆盖每个通道的通道值，而只会覆盖没有 reducer 的通道。让我们通过一个例子来说明。

假设您已经使用以下模式定义了图的状态（请参见上面的完整示例）：

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
```

```typescript
import { registry } from "@langchain/langgraph/zod";
import * as z from "zod";

const State = z.object({
  foo: z.number(),
  bar: z.array(z.string()).register(registry, {
    reducer: {
      fn: (x, y) => x.concat(y),
    },
    default: () => [] as string[],
  }),
});
```

现在假设图的当前状态是：

```
{"foo": 1, "bar": ["a"]}
```

```typescript
{ foo: 1, bar: ["a"] }
```

如果您按如下方式更新状态：

```python
graph.update_state(config, {"foo": 2, "bar": ["b"]})
```

```typescript
await graph.updateState(config, { foo: 2, bar: ["b"] });
```

那么图的新状态将是：

```
{"foo": 2, "bar": ["a", "b"]}
```

`foo` 键（通道）完全更改了（因为没有为该通道指定 reducer，所以 `update_state` 会覆盖它）。但是，为 `bar` 键指定了 reducer，因此它将 `"b"` 添加到 `bar` 的状态中。

```typescript
{ foo: 2, bar: ["a", "b"] }
```

`foo` 键（通道）完全更改了（因为没有为该通道指定 reducer，所以 `updateState` 会覆盖它）。但是，为 `bar` 键指定了 reducer，因此它将 `"b"` 添加到 `bar` 的状态中。

##### as_node

调用 `update_state` 时可以可选指定的最后一个参数是 `as_node`。如果您提供它，更新将被应用，就好像它来自节点 `as_node`。如果未提供 `as_node`，它将被设置为最后一个更新状态的节点（如果不明确）。这很重要，因为要执行的下一步取决于最后一个提供更新的节点，因此这可以用来控制下一个执行的节点。

### 内存存储

[状态模式](https://langchain-doc.cn/v1/python/langgraph/graph-api#schema) 指定了一组在图执行时填充的键。如上所述，状态可以由检查点保存器在每个图步骤写入线程，从而实现状态持久化。

但是，如果我们想在**线程之间**保留一些信息怎么办？考虑聊天机器人的情况，我们希望在与该用户的**所有**聊天对话（例如线程）中保留有关用户的特定信息！

仅靠检查点，我们无法在线程之间共享信息。这就催生了 [`Store`](https://python.langchain.com/api_reference/langgraph/index.html#module-langgraph.store) 接口的需求。例如，我们可以定义一个 `InMemoryStore` 来存储跨线程的用户信息。我们只需像之前一样使用检查点编译图，并使用我们新的 `in_memory_store` 变量。

**LangGraph API 自动处理存储**
当使用 LangGraph API 时，您不需要手动实现或配置存储。API 在幕后为您处理所有存储基础设施。

#### 基本用法

首先，让我们在不使用 LangGraph 的情况下单独展示这一点。

```python
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()
```

```typescript
import { MemoryStore } from "@langchain/langgraph";

const memoryStore = new MemoryStore();
```

记忆按 `tuple` 进行命名空间管理，在这个特定例子中是 `(<user_id>, "memories")`。命名空间可以是任意长度，表示任何内容，不一定是用户特定的。

```python
user_id = "1"
namespace_for_memory = (user_id, "memories")
```

```typescript
const userId = "1";
const namespaceForMemory = [userId, "memories"];
```

我们使用 `store.put` 方法将记忆保存到存储中的命名空间。当我们这样做时，我们指定上面定义的命名空间，以及记忆的键值对：键只是记忆的唯一标识符 (`memory_id`)，值（字典）是记忆本身。

```python
memory_id = str(uuid.uuid4())
memory = {"food_preference" : "I like pizza"}
in_memory_store.put(namespace_for_memory, memory_id, memory)
```

```typescript
import { v4 as uuidv4 } from "uuid";

const memoryId = uuidv4();
const memory = { food_preference: "I like pizza" };
await memoryStore.put(namespaceForMemory, memoryId, memory);
```

我们可以使用 `store.search` 方法读取命名空间中的记忆，该方法将返回给定用户的所有记忆作为列表。最近的记忆是列表中的最后一个。

```python
memories = in_memory_store.search(namespace_for_memory)
memories[-1].dict()
{'value': {'food_preference': 'I like pizza'},
 'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
 'namespace': ['1', 'memories'],
 'created_at': '2024-10-02T17:22:31.590602+00:00',
 'updated_at': '2024-10-02T17:22:31.590605+00:00'}
```

每个记忆类型是一个 Python 类 ([`Item`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.Item))，具有某些属性。我们可以通过如上所述的 `.dict` 转换将其作为字典访问。

它具有的属性包括：

- `value`: 此记忆的值（本身是一个字典）
- `key`: 此记忆在此命名空间中的唯一键
- `namespace`: 字符串列表，此记忆类型的命名空间
- `created_at`: 此记忆创建的时间戳
- `updated_at`: 此记忆更新的时间戳

```typescript
const memories = await memoryStore.search(namespaceForMemory);
memories[memories.length - 1];

// {
//   value: { food_preference: 'I like pizza' },
//   key: '07e0caf4-1631-47b7-b15f-65515d4c1843',
//   namespace: ['1', 'memories'],
//   createdAt: '2024-10-02T17:22:31.590602+00:00',
//   updatedAt: '2024-10-02T17:22:31.590605+00:00'
// }
```

它具有的属性包括：

- `value`: 此记忆的值
- `key`: 此记忆在此命名空间中的唯一键
- `namespace`: 字符串列表，此记忆类型的命名空间
- `createdAt`: 此记忆创建的时间戳
- `updatedAt`: 此记忆更新的时间戳

#### 语义搜索

除了简单的检索外，存储还支持语义搜索，允许您基于含义而不是精确匹配来查找记忆。要启用此功能，请使用嵌入模型配置存储：

```python
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),  # 嵌入提供者
        "dims": 1536,                              # 嵌入维度
        "fields": ["food_preference", "$"]              # 要嵌入的字段
    }
)
```

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";

const store = new InMemoryStore({
  index: {
    embeddings: new OpenAIEmbeddings({ model: "text-embedding-3-small" }),
    dims: 1536,
    fields: ["food_preference", "$"], // 要嵌入的字段
  },
});
```

现在，在搜索时，您可以使用自然语言查询来查找相关记忆：

```python
# 查找有关食物偏好的记忆
# （这可以在将记忆放入存储后完成）
memories = store.search(
    namespace_for_memory,
    query="What does the user like to eat?",
    limit=3  # 返回前 3 个匹配项
)
```

```typescript
// 查找有关食物偏好的记忆
// （这可以在将记忆放入存储后完成）
const memories = await store.search(namespaceForMemory, {
  query: "What does the user like to eat?",
  limit: 3, // 返回前 3 个匹配项
});
```

您可以通过配置 `fields` 参数或在存储记忆时指定 `index` 参数来控制记忆的哪些部分被嵌入：

```python
# 使用特定字段嵌入存储
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {
        "food_preference": "I love Italian cuisine",
        "context": "Discussing dinner plans"
    },
    index=["food_preference"]  # 仅嵌入 "food_preferences" 字段
)

# 不嵌入存储（仍然可检索，但不可搜索）
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {"system_info": "Last updated: 2024-01-01"},
    index=False
)
```

```typescript
// 使用特定字段嵌入存储
await store.put(
  namespaceForMemory,
  uuidv4(),
  {
    food_preference: "I love Italian cuisine",
    context: "Discussing dinner plans",
  },
  { index: ["food_preference"] } // 仅嵌入 "food_preferences" 字段
);

// 不嵌入存储（仍然可检索，但不可搜索）
await store.put(
  namespaceForMemory,
  uuidv4(),
  { system_info: "Last updated: 2024-01-01" },
  { index: false }
);
```

#### 在 LangGraph 中使用

有了这一切，我们在 LangGraph 中使用 `in_memory_store`。`in_memory_store` 与检查点协同工作：检查点将状态保存到线程（如上所述），而 `in_memory_store` 允许我们存储任意信息以在**线程之间**访问。我们使用检查点和 `in_memory_store` 编译图如下：

```python
from langgraph.checkpoint.memory import InMemorySaver

# 我们需要这个因为我们想启用线程（对话）
checkpointer = InMemorySaver()

# ... 定义图 ...

# 使用检查点和存储编译图
graph = graph.compile(checkpointer=checkpointer, store=in_memory_store)
```

```typescript
import { MemorySaver } from "@langchain/langgraph";

// 我们需要这个因为我们想启用线程（对话）
const checkpointer = new MemorySaver();

// ... 定义图 ...

// 使用检查点和存储编译图
const graph = workflow.compile({ checkpointer, store: memoryStore });
```

我们像之前一样使用 `thread_id` 调用图，同时也使用 `user_id`，我们将使用它来为此特定用户的记忆设置命名空间，如上面所示。

```python
# 调用图
user_id = "1"
config = {"configurable": {"thread_id": "1", "user_id": user_id}}

# 首先让我们向 AI 问好
for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi"}]}, config, stream_mode="updates"
):
    print(update)
```

```typescript
// 调用图
const userId = "1";
const config = { configurable: { thread_id: "1", user_id: userId } };

// 首先让我们向 AI 问好
for await (const update of await graph.stream(
  { messages: [{ role: "user", content: "hi" }] },
  { ...config, streamMode: "updates" }
)) {
  console.log(update);
}
```

我们可以通过在节点参数中传递 `store: BaseStore` 和 `config: RunnableConfig` 来在**任何节点**中访问 `in_memory_store` 和 `user_id`。以下是我们如何在节点中使用语义搜索来查找相关记忆的示例：

```python
def update_memory(state: MessagesState, config: RunnableConfig, *, store: BaseStore):

    # 从配置中获取用户 ID
    user_id = config["configurable"]["user_id"]

    # 为记忆设置命名空间
    namespace = (user_id, "memories")

    # ... 分析对话并创建新记忆

    # 创建新的记忆 ID
    memory_id = str(uuid.uuid4())

    # 我们创建新的记忆
    store.put(namespace, memory_id, {"memory": memory})
```

```typescript
import { MessagesZodMeta, Runtime } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";
import { registry } from "@langchain/langgraph/zod";
import * as z from "zod";

const MessagesZodState = z.object({
  messages:
    z.array(z.custom<BaseMessage>())
    .register(registry, MessagesZodMeta),
});

const updateMemory = async (
  state: z.infer<typeof MessagesZodState>,
  runtime: Runtime<{ user_id: string }>,
) => {
  // 从配置中获取用户 ID
  const userId = runtime.context?.user_id;
  if (!userId) throw new Error("User ID is required");

  // 为记忆设置命名空间
  const namespace = [userId, "memories"];

  // ... 分析对话并创建新记忆

  // 创建新的记忆 ID
  const memoryId = uuidv4();

  // 我们创建新的记忆
  await runtime.store?.put(namespace, memoryId, { memory });
};
```

如前所述，我们也可以在任何节点中访问存储并使用 `store.search` 方法获取记忆。回忆一下，记忆作为可以转换为字典的对象列表返回。

```python
memories[-1].dict()
{'value': {'food_preference': 'I like pizza'},
 'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
 'namespace': ['1', 'memories'],
 'created_at': '2024-10-02T17:22:31.590602+00:00',
 'updated_at': '2024-10-02T17:22:31.590605+00:00'}
```

```typescript
memories[memories.length - 1];
// {
//   value: { food_preference: 'I like pizza' },
//   key: '07e0caf4-1631-47b7-b15f-65515d4c1843',
//   namespace: ['1', 'memories'],
//   createdAt: '2024-10-02T17:22:31.590602+00:00',
//   updatedAt: '2024-10-02T17:22:31.590605+00:00'
// }
```

我们可以访问记忆并在模型调用中使用它们。

```python
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    # 从配置中获取用户 ID
    user_id = config["configurable"]["user_id"]

    # 为记忆设置命名空间
    namespace = (user_id, "memories")

    # 基于最新消息进行搜索
    memories = store.search(
        namespace,
        query=state["messages"][-1].content,
        limit=3
    )
    info = "\n".join([d.value["memory"] for d in memories])

    # ... 在模型调用中使用记忆
```

```typescript
const callModel = async (
  state: z.infer<typeof MessagesZodState>,
  config: LangGraphRunnableConfig,
  store: BaseStore
) => {
  // 从配置中获取用户 ID
  const userId = config.configurable?.user_id;

  // 为记忆设置命名空间
  const namespace = [userId, "memories"];

  // 基于最新消息进行搜索
  const memories = await store.search(namespace, {
    query: state.messages[state.messages.length - 1].content,
    limit: 3,
  });
  const info = memories.map((d) => d.value.memory).join("\n");

  // ... 在模型调用中使用记忆
};
```

如果我们创建一个新线程，只要 `user_id` 相同，我们仍然可以访问相同的记忆。

```python
# 调用图
config = {"configurable": {"thread_id": "2", "user_id": "1"}}

# 让我们再次问好
for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi, tell me about my memories"}]}, config, stream_mode="updates"
):
    print(update)
```

```typescript
// 调用图
const config = { configurable: { thread_id: "2", user_id: "1" } };

// 让我们再次问好
for await (const update of await graph.stream(
  { messages: [{ role: "user", content: "hi, tell me about my memories" }] },
  { ...config, streamMode: "updates" }
)) {
  console.log(update);
}
```

当我们使用 LangSmith 时，无论是本地（例如在 [Studio](https://langchain-doc.cn/langsmith/studio) 中）还是通过 [LangSmith 托管](https://langchain-doc.cn/langsmith/platform-setup)，基础存储默认可用，不需要在图编译期间指定。但是，要启用语义搜索，您**确实**需要在 `langgraph.json` 文件中配置索引设置。例如：

```json
{
    ...
    "store": {
        "index": {
            "embed": "openai:text-embeddings-3-small",
            "dims": 1536,
            "fields": ["$"]
        }
    }
}
```

有关更多详细信息和配置选项，请参阅 [部署指南](https://langchain-doc.cn/langsmith/semantic-search)。

### 检查点库

在底层，检查点功能由符合 `BaseCheckpointSaver` 接口的检查点对象提供支持。LangGraph 提供了几个检查点实现，全部通过独立的、可安装的库实现：

- `langgraph-checkpoint`：检查点保存器的基本接口 (`BaseCheckpointSaver`) 和序列化/反序列化接口 (`SerializerProtocol`)。包括用于实验的内存检查点实现 (`InMemorySaver`)。LangGraph 包含 `langgraph-checkpoint`。
- `langgraph-checkpoint-sqlite`：使用 SQLite 数据库的 LangGraph 检查点实现 (`SqliteSaver`/`AsyncSqliteSaver`)。适用于实验和本地工作流程。需要单独安装。
- `langgraph-checkpoint-postgres`：使用 Postgres 数据库的高级检查点 (`PostgresSaver`/`AsyncPostgresSaver`)，用于 LangSmith。适用于生产环境。需要单独安装。
- `@langchain/langgraph-checkpoint`：检查点保存器的基本接口 (`BaseCheckpointSaver`) 和序列化/反序列化接口 (`SerializerProtocol`)。包括用于实验的内存检查点实现 (`MemorySaver`)。LangGraph 包含 `@langchain/langgraph-checkpoint`。
- `@langchain/langgraph-checkpoint-sqlite`：使用 SQLite 数据库的 LangGraph 检查点实现 (`SqliteSaver`)。适用于实验和本地工作流程。需要单独安装。
- `@langchain/langgraph-checkpoint-postgres`：使用 Postgres 数据库的高级检查点 (`PostgresSaver`)，用于 LangSmith。适用于生产环境。需要单独安装。

#### 检查点接口

每个检查点都符合 `BaseCheckpointSaver` 接口并实现以下方法：

- `.put` - 存储带有其配置和元数据的检查点。
- `.put_writes` - 存储链接到检查点的中间写入（即 [待处理写入](https://langchain-doc.cn/v1/python/langgraph/persistence.html#pending-writes)）。
- `.get_tuple` - 使用给定配置 (`thread_id` 和 `checkpoint_id`) 获取检查点元组。这用于在 `graph.get_state()` 中填充 `StateSnapshot`。
- `.list` - 列出匹配给定配置和过滤条件的检查点。这用于在 `graph.get_state_history()` 中填充状态历史。

如果检查点与异步图执行一起使用（即通过 `.ainvoke`、`.astream`、`.abatch` 执行图），将使用上述方法的异步版本（`.aput`、`.aput_writes`、`.aget_tuple`、`.alist`）。

对于异步运行图，您可以使用 `InMemorySaver` 或 Sqlite/Postgres 检查点的异步版本 - `AsyncSqliteSaver`/`AsyncPostgresSaver` 检查点。

- `.putWrites` - 存储链接到检查点的中间写入（即 [待处理写入](https://langchain-doc.cn/v1/python/langgraph/persistence.html#pending-writes)）。
- `.getTuple` - 使用给定配置 (`thread_id` 和 `checkpoint_id`) 获取检查点元组。这用于在 `graph.getState()` 中填充 `StateSnapshot`。
- `.list` - 列出匹配给定配置和过滤条件的检查点。这用于在 `graph.getStateHistory()` 中填充状态历史。

#### 序列化器

当检查点保存图状态时，它们需要序列化状态中的通道值。这是通过序列化器对象完成的。

`langgraph_checkpoint` 定义了用于实现序列化器的协议，并提供了一个默认实现 (`JsonPlusSerializer`)，该实现处理各种各样的类型，包括 LangChain 和 LangGraph 原语、日期时间、枚举等。

##### 使用 `pickle` 进行序列化

默认的序列化器 `JsonPlusSerializer` 在底层使用 ormsgpack 和 JSON，这不适合所有类型的对象。

如果您想对我们的 msgpack 编码器当前不支持的对象（例如 Pandas 数据帧）回退到 pickle，您可以使用 `JsonPlusSerializer` 的 `pickle_fallback` 参数：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# ... 定义图 ...
graph.compile(
    checkpointer=InMemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))
)
```

##### 加密

检查点可以选择性地加密所有持久化状态。要启用此功能，请将 `EncryptedSerializer` 的实例传递给任何 `BaseCheckpointSaver` 实现的 `serde` 参数。创建加密序列化器的最简单方法是通过 `from_pycryptodome_aes`，它从 `LANGGRAPH_AES_KEY` 环境变量读取 AES 密钥（或接受 `key` 参数）：

```python
import sqlite3

from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

serde = EncryptedSerializer.from_pycryptodome_aes()  # 读取 LANGGRAPH_AES_KEY
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db"), serde=serde)
```

```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.postgres import PostgresSaver

serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = PostgresSaver.from_conn_string("postgresql://...", serde=serde)
checkpointer.setup()
```

在 LangSmith 上运行时，只要存在 `LANGGRAPH_AES_KEY`，加密就会自动启用，因此您只需提供环境变量。可以通过实现 `CipherProtocol` 并将其提供给 `EncryptedSerializer` 来使用其他加密方案。

`@langchain/langgraph-checkpoint` 定义了用于实现序列化器的协议，并提供了一个默认实现，可以处理各种各样的类型，包括 LangChain 和 LangGraph 原语、日期时间、枚举等。

### 功能

#### 人机循环

首先，检查点通过允许人类检查、中断和批准图步骤来促进 [人机循环工作流](https://langchain-doc.cn/v1/python/langgraph/interrupts) 工作流。这些工作流需要检查点，因为人类必须能够在任何时间点查看图的状态，并且图必须能够在人类对状态进行任何更新后恢复执行。

#### 记忆

其次，检查点允许在交互之间保持 ["记忆"](https://langchain-doc.cn/v1/python/concepts/memory)。在重复的人类交互（如对话）的情况下，任何后续消息都可以发送到该线程，该线程将保留先前消息的记忆。

#### 时间旅行

第三，检查点允许 ["时间旅行"](https://langchain-doc.cn/v1/python/langgraph/use-time-travel)，允许用户重放先前的图执行以查看和/或调试特定的图步骤。此外，检查点还使在任意检查点分叉图状态以探索替代轨迹成为可能。

#### 容错

最后，检查点还提供容错和错误恢复：如果一个或多个节点在给定的超级步骤失败，您可以从最后一个成功步骤重新启动图。此外，当图节点在给定的超级步骤中间执行失败时，LangGraph 会存储来自该超级步骤中成功完成的任何其他节点的待处理检查点写入，以便每当我们从该超级步骤恢复图执行时，我们不会重新运行成功的节点。

##### 待处理写入

此外，当图节点在给定的超级步骤中间执行失败时，LangGraph 会存储来自该超级步骤中成功完成的任何其他节点的待处理检查点写入，以便每当我们从该超级步骤恢复图执行时，我们不会重新运行成功的节点。

## 持久执行

**持久执行**是一种技术，其中流程或工作流在关键点保存其进度，使其能够暂停并在稍后从停止的确切位置恢复。这在需要[人在循环](https://langchain-doc.cn/v1/python/langgraph/interrupts)的场景中特别有用，用户可以在继续之前检查、验证或修改流程，以及在可能遇到中断或错误的长时间运行任务中（例如，LLM调用超时）。通过保留已完成的工作，持久执行使流程能够恢复而无需重新处理之前的步骤——即使在显著延迟后（例如，一周后）。

LangGraph的内置[持久化](https://langchain-doc.cn/v1/python/langgraph/persistence)层为工作流提供持久执行，确保每个执行步骤的状态都保存到持久存储中。这种能力保证了如果工作流被中断——无论是由于系统故障还是为了[人在循环](https://langchain-doc.cn/v1/python/langgraph/interrupts)交互——它都可以从最后记录的状态恢复。

**提示：** 如果您在LangGraph中使用了checkpointer，那么您已经启用了持久执行。您可以在任何点暂停和恢复工作流，即使在中断或失败后也是如此。
为了充分利用持久执行，请确保您的工作流设计为[确定性的](https://langchain-doc.cn/v1/python/langgraph/durable-execution.html#确定性和一致重放)和[幂等的](https://langchain-doc.cn/v1/python/langgraph/durable-execution.html#确定性和一致重放)，并将任何副作用或非确定性操作包装在[任务](https://langchain-doc.cn/v1/python/langgraph/functional-api#task)中。您可以从[StateGraph（Graph API）](https://langchain-doc.cn/v1/python/langgraph/graph-api)和[Functional API](https://langchain-doc.cn/v1/python/langgraph/functional-api)中使用[任务](https://langchain-doc.cn/v1/python/langgraph/functional-api#task)。

### 要求

要在LangGraph中利用持久执行，您需要：

1. 通过指定[checkpointer](https://langchain-doc.cn/v1/python/langgraph/persistence#checkpointer-libraries)来保存工作流进度，从而在工作流中启用[持久化](https://langchain-doc.cn/v1/python/langgraph/persistence)。
2. 在执行工作流时指定[线程标识符](https://langchain-doc.cn/v1/python/langgraph/persistence#threads)。这将跟踪特定工作流实例的执行历史。

在Python中：

- 将任何非确定性操作（例如，随机数生成）或具有副作用的操作（例如，文件写入、API调用）包装在`@task`中，以确保当工作流恢复时，这些操作不会为特定运行重复，而是从持久层中检索其结果。有关更多信息，请参阅[确定性和一致重放](https://langchain-doc.cn/v1/python/langgraph/durable-execution.html#确定性和一致重放)。

在JavaScript中：

- 将任何非确定性操作（例如，随机数生成）或具有副作用的操作（例如，文件写入、API调用）包装在`@task`中，以确保当工作流恢复时，这些操作不会为特定运行重复，而是从持久层中检索其结果。有关更多信息，请参阅[确定性和一致重放](https://langchain-doc.cn/v1/python/langgraph/durable-execution.html#确定性和一致重放)。

### 确定性和一致重放

当您恢复工作流运行时，代码**不会**从执行停止的**同一行代码**恢复；相反，它会识别一个适当的[起点](https://langchain-doc.cn/v1/python/langgraph/durable-execution.html#工作流恢复的起点)，从那里继续。这意味着工作流将从[起点](https://langchain-doc.cn/v1/python/langgraph/durable-execution.html#工作流恢复的起点)重放所有步骤，直到到达停止的点。

因此，当您为持久执行编写工作流时，必须将任何非确定性操作（例如，随机数生成）和任何具有副作用的操作（例如，文件写入、API调用）包装在[任务](https://langchain-doc.cn/v1/python/langgraph/functional-api#task)或[节点](https://langchain-doc.cn/v1/python/langgraph/graph-api#nodes)中。

为确保您的工作流是确定性的并且可以一致地重放，请遵循以下准则：

- **避免重复工作**：如果[节点](https://langchain-doc.cn/v1/python/langgraph/graph-api#nodes)包含多个具有副作用的操作（例如，日志记录、文件写入或网络调用），请将每个操作包装在单独的**任务**中。这确保当工作流恢复时，操作不会重复，并且它们的结果会从持久层中检索。
- **封装非确定性操作**：将可能产生非确定性结果的任何代码（例如，随机数生成）包装在**任务**或**节点**中。这确保在恢复时，工作流遵循完全记录的步骤序列，具有相同的结果。
- **使用幂等操作**：尽可能确保副作用（例如，API调用、文件写入）是幂等的。这意味着如果操作在工作流失败后重试，它将产生与第一次执行相同的效果。这对于导致数据写入的操作尤为重要。如果**任务**开始但未能成功完成，工作流的恢复将重新运行**任务**，依靠记录的结果来保持一致性。使用幂等键或验证现有结果以避免意外重复，确保工作流执行平稳且可预测。

在Python中，有关要避免的常见陷阱示例，请参阅功能API中的[常见陷阱](https://langchain-doc.cn/v1/python/langgraph/functional-api#common-pitfalls)部分，该部分展示了如何使用**任务**来构建代码以避免这些问题。相同的原则也适用于`StateGraph`（Graph API）。

在JavaScript中，有关要避免的常见陷阱示例，请参阅功能API中的[常见陷阱](https://langchain-doc.cn/v1/python/langgraph/functional-api#common-pitfalls)部分，该部分展示了如何使用**任务**来构建代码以避免这些问题。相同的原则也适用于`StateGraph`（Graph API）。



### 持久性模式

LangGraph支持三种持久性模式，允许您根据应用程序的要求平衡性能和数据一致性。从最少到最持久的持久性模式如下：

- [`"exit"`](https://langchain-doc.cn/v1/python/langgraph/durable-execution.html#exit)
- [`"async"`](https://langchain-doc.cn/v1/python/langgraph/durable-execution.html#async)
- [`"sync"`](https://langchain-doc.cn/v1/python/langgraph/durable-execution.html#sync)

较高的持久性模式会给工作流执行增加更多开销。

**提示：**
**v0.6.0新增**
使用`durability`参数而不是`checkpoint_during`（v0.6.0中已弃用）进行持久性策略管理：

- `durability="async"` 替代 `checkpoint_during=True`
- `durability="exit"` 替代 `checkpoint_during=False`

#### "exit"

更改仅在图形执行完成时（成功或出错）才会持久化。这为长时间运行的图形提供了最佳性能，但意味着中间状态不会保存，因此您无法从执行过程中的失败中恢复或中断图形执行。

#### "async"

更改在执行下一步的同时异步持久化。这提供了良好的性能和持久性，但如果进程在执行期间崩溃，检查点可能不会被写入，存在小风险。

#### "sync"

更改在开始下一步之前同步持久化。这确保每个检查点在继续执行之前被写入，以牺牲一些性能开销为代价提供高持久性。

您可以在调用任何图形执行方法时指定持久性模式：

```python
graph.stream(
    {"input": "test"},
    durability="sync"
)
```



### 在节点中使用任务

如果[节点](https://langchain-doc.cn/v1/python/langgraph/graph-api#nodes)包含多个操作，您可能会发现将每个操作转换为**任务**比将操作重构为单独的节点更容易。

#### Python示例

##### 原始代码

```python
from typing import NotRequired
from typing_extensions import TypedDict
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
import requests

# 定义TypedDict表示状态
class State(TypedDict):
    url: str
    result: NotRequired[str]

def call_api(state: State):
    """示例节点，用于发出API请求。"""
    result = requests.get(state['url']).text[:100]  # 副作用
    return {
        "result": result
    }

# 创建StateGraph构建器并为call_api函数添加节点
builder = StateGraph(State)
builder.add_node("call_api", call_api)

# 连接开始和结束节点到call_api节点
builder.add_edge(START, "call_api")
builder.add_edge("call_api", END)

# 指定checkpointer
checkpointer = InMemorySaver()

# 用checkpointer编译图形
graph = builder.compile(checkpointer=checkpointer)

# 定义带有线程ID的配置。
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

# 调用图形
graph.invoke({"url": "https://www.example.com"}, config)
```

##### 使用任务

```python
from typing import NotRequired
from typing_extensions import TypedDict
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import task
from langgraph.graph import StateGraph, START, END
import requests

# 定义TypedDict表示状态
class State(TypedDict):
    urls: list[str]
    result: NotRequired[list[str]]


@task
def _make_request(url: str):
    """发出请求。"""
    return requests.get(url).text[:100]

def call_api(state: State):
    """示例节点，用于发出API请求。"""
    requests = [_make_request(url) for url in state['urls']]
    results = [request.result() for request in requests]
    return {
        "results": results
    }

# 创建StateGraph构建器并为call_api函数添加节点
builder = StateGraph(State)
builder.add_node("call_api", call_api)

# 连接开始和结束节点到call_api节点
builder.add_edge(START, "call_api")
builder.add_edge("call_api", END)

# 指定checkpointer
checkpointer = InMemorySaver()

# 用checkpointer编译图形
graph = builder.compile(checkpointer=checkpointer)

# 定义带有线程ID的配置。
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

# 调用图形
graph.invoke({"urls": ["https://www.example.com"]}, config)
```



### 恢复工作流

一旦您在工作流中启用了持久执行，您可以在以下场景中恢复执行：

#### Python中

- **暂停和恢复工作流：** 使用`interrupt`函数在特定点暂停工作流，并使用`Command`原语用更新的状态恢复它。有关更多详细信息，请参阅[**中断**](https://langchain-doc.cn/v1/python/langgraph/interrupts)。
- **从失败中恢复：** 在异常（例如，LLM提供商中断）后自动从最后一个成功的检查点恢复工作流。这涉及通过提供`None`作为输入值（请参阅功能API中的[示例](https://langchain-doc.cn/v1/python/langgraph/use-functional-api#resuming-after-an-error)）使用相同的线程标识符执行工作流。

#### JavaScript中

- **暂停和恢复工作流：** 使用`interrupt`函数在特定点暂停工作流，并使用`Command`原语用更新的状态恢复它。有关更多详细信息，请参阅[**中断**](https://langchain-doc.cn/v1/python/langgraph/interrupts)。
- **从失败中恢复：** 在异常（例如，LLM提供商中断）后自动从最后一个成功的检查点恢复工作流。这涉及通过提供`null`作为输入值（请参阅功能API中的[示例](https://langchain-doc.cn/v1/python/langgraph/use-functional-api#resuming-after-an-error)）使用相同的线程标识符执行工作流。

### 工作流恢复的起点

#### Python中

- 如果您使用`StateGraph`（Graph API），起点是执行停止的[**节点**](https://langchain-doc.cn/v1/python/langgraph/graph-api#nodes)的开始。
- 如果您在节点内进行子图调用，起点将是调用被暂停的子图的**父**节点。在子图内部，起点将是执行停止的特定[**节点**](https://langchain-doc.cn/v1/python/langgraph/graph-api#nodes)。
- 如果您使用Functional API，起点是执行停止的[**入口点**](https://langchain-doc.cn/v1/python/langgraph/functional-api#entrypoint)的开始。

#### JavaScript中

- 如果您使用[StateGraph（Graph API）](https://langchain-doc.cn/v1/python/langgraph/graph-api)，起点是执行停止的[**节点**](https://langchain-doc.cn/v1/python/langgraph/graph-api#nodes)的开始。
- 如果您在节点内进行子图调用，起点将是调用被暂停的子图的**父**节点。在子图内部，起点将是执行停止的特定[**节点**](https://langchain-doc.cn/v1/python/langgraph/graph-api#nodes)。
- 如果您使用Functional API，起点是执行停止的[**入口点**](https://langchain-doc.cn/v1/python/langgraph/functional-api#entrypoint)的开始。

## 流式传输

LangGraph实现了一个流式传输系统，用于实时展示更新。流式传输对于增强基于LLM构建的应用程序的响应性至关重要。通过在完整响应准备好之前逐步显示输出，流式传输显著改善了用户体验(UX)，特别是在处理LLM延迟时。

LangGraph流式传输可以实现以下功能：

- **流式传输图状态** — 使用`updates`和`values`模式获取状态更新/值。
- **流式传输子图输出** — 包含父图和任何嵌套子图的输出。
- **流式传输LLM令牌** — 从任何地方捕获令牌流：节点内部、子图或工具中。
- **流式传输自定义数据** — 直接从工具函数发送自定义更新或进度信号。
- **使用多种流式传输模式** — 可以选择`values`（完整状态）、`updates`（状态增量）、`messages`（LLM令牌+元数据）、`custom`（任意用户数据）或`debug`（详细跟踪）。

### 支持的流式传输模式

将以下流式传输模式中的一种或多种作为列表传递给`stream`或`astream`方法：

| 模式       | 描述                                                         |
| ---------- | ------------------------------------------------------------ |
| `values`   | 在图的每个步骤后流式传输状态的完整值。                       |
| `updates`  | 在图的每个步骤后流式传输状态的更新。如果在同一步骤中进行了多次更新（例如，运行了多个节点），这些更新会分别流式传输。 |
| `custom`   | 从图节点内部流式传输自定义数据。                             |
| `messages` | 从调用LLM的任何图节点流式传输2元组（LLM令牌，元数据）。      |
| `debug`    | 在图执行过程中流式传输尽可能多的信息。                       |

### 基本使用示例

LangGraph图暴露了`stream`（同步）和`astream`（异步）方法，以迭代器的形式生成流式输出。

#### Python

```python
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)
```

#### JavaScript

```typescript
for await (const chunk of await graph.stream(inputs, {
  streamMode: "updates",
})) {
  console.log(chunk);
}
```

#### 扩展示例：流式传输更新

##### Python

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    joke: str

def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}

def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)

# stream()方法返回一个迭代器，生成流式输出
for chunk in graph.stream(
    {"topic": "ice cream"},
    # 设置stream_mode="updates"以仅流式传输每个节点后图状态的更新
    # 也可以使用其他流式传输模式。详见支持的流式传输模式
    stream_mode="updates",
):
    print(chunk)
```

##### JavaScript

```python
import { StateGraph, START, END } from "@langchain/langgraph";
import * as z from "zod";

const State = z.object({
  topic: z.string(),
  joke: z.string(),
});

const graph = new StateGraph(State)
  .addNode("refineTopic", (state) => {
    return { topic: state.topic + " and cats" };
  })
  .addNode("generateJoke", (state) => {
    return { joke: `This is a joke about ${state.topic}` };
  })
  .addEdge(START, "refineTopic")
  .addEdge("refineTopic", "generateJoke")
  .addEdge("generateJoke", END)
  .compile();

for await (const chunk of await graph.stream(
  { topic: "ice cream" },
  // 设置streamMode: "updates"以仅流式传输每个节点后图状态的更新
  // 也可以使用其他流式传输模式。详见支持的流式传输模式
  { streamMode: "updates" }
)) {
  console.log(chunk);
}
```

### 流式传输多种模式

#### Python

您可以将列表作为`stream_mode`参数传递，一次流式传输多种模式。

流式输出将是元组`(mode, chunk)`，其中`mode`是流式传输模式的名称，`chunk`是该模式流式传输的数据。

```python
for mode, chunk in graph.stream(inputs, stream_mode=["updates", "custom"]):
    print(chunk)
```

#### JavaScript

您可以将数组作为`streamMode`参数传递，一次流式传输多种模式。

流式输出将是元组`[mode, chunk]`，其中`mode`是流式传输模式的名称，`chunk`是该模式流式传输的数据。\

```typescript
for await (const [mode, chunk] of await graph.stream(inputs, {
  streamMode: ["updates", "custom"],
})) {
  console.log(chunk);
}
```

### 流式传输图状态

使用流式传输模式`updates`和`values`来流式传输图执行时的状态。

- `updates`在图的每个步骤后流式传输状态的**更新**。
- `values`在图的每个步骤后流式传输状态的**完整值**。

**Python**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
  topic: str
  joke: str


def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}


def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
  StateGraph(State)
  .add_node(refine_topic)
  .add_node(generate_joke)
  .add_edge(START, "refine_topic")
  .add_edge("refine_topic", "generate_joke")
  .add_edge("generate_joke", END)
  .compile()
)
```

**JavaScript**

```typescript
import { StateGraph, START, END } from "@langchain/langgraph";
import * as z from "zod";

const State = z.object({
  topic: z.string(),
  joke: z.string(),
});

const graph = new StateGraph(State)
  .addNode("refineTopic", (state) => {
    return { topic: state.topic + " and cats" };
  })
  .addNode("generateJoke", (state) => {
    return { joke: `This is a joke about ${state.topic}` };
  })
  .addEdge(START, "refineTopic")
  .addEdge("refineTopic", "generateJoke")
  .addEdge("generateJoke", END)
  .compile();
```

#### 使用updates模式

使用此模式仅流式传输每个步骤后节点返回的**状态更新**。流式输出包括节点名称和更新内容。

**Python**

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="updates",
):
    print(chunk)
```

**JavaScript**

```typescript
for await (const chunk of await graph.stream(
  { topic: "ice cream" },
  { streamMode: "updates" }
)) {
  console.log(chunk);
}
```

#### 使用values模式

使用此模式流式传输每个步骤后的**完整图状态**。

##### Python

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="values",
):
    print(chunk)
```

##### JavaScript

```typescript
for await (const chunk of await graph.stream(
  { topic: "ice cream" },
  { streamMode: "values" }
)) {
  console.log(chunk);
}
```

### 流式传输子图输出

**Python**

要在流式输出中包含子图的输出，可以在父图的`.stream()`方法中设置`subgraphs=True`。这将流式传输父图和任何子图的输出。

输出将以元组`(namespace, data)`的形式流式传输，其中`namespace`是包含子图调用节点路径的元组，例如`("parent_node:<task_id>", "child_node:<task_id>")`。

```python
for chunk in graph.stream(
    {"foo": "foo"},
    # 设置subgraphs=True以流式传输子图的输出
    subgraphs=True,
    stream_mode="updates",
):
    print(chunk)
```

**JavaScript**

要在流式输出中包含子图的输出，可以在父图的`.stream()`方法中设置`subgraphs: true`。这将流式传输父图和任何子图的输出。

输出将以元组`[namespace, data]`的形式流式传输，其中`namespace`是包含子图调用节点路径的元组，例如`["parent_node:<task_id>", "child_node:<task_id>"]`。

```typescript
for await (const chunk of await graph.stream(
  { foo: "foo" },
  {
    // 设置subgraphs: true以流式传输子图的输出
    subgraphs: true,
    streamMode: "updates",
  }
)) {
  console.log(chunk);
}
```

#### 扩展示例：从子图流式传输

**Python**

```python
from langgraph.graph import START, StateGraph
from typing import TypedDict

# 定义子图
class SubgraphState(TypedDict):
    foo: str  # 注意这个键与父图状态共享
    bar: str

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# 定义父图
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    # 设置subgraphs=True以流式传输子图的输出
    subgraphs=True,
):
    print(chunk)
```

**JavaScript**

```typescript
import { StateGraph, START } from "@langchain/langgraph";
import * as z from "zod";

// 定义子图
const SubgraphState = z.object({
  foo: z.string(), // 注意这个键与父图状态共享
  bar: z.string(),
});

const subgraphBuilder = new StateGraph(SubgraphState)
  .addNode("subgraphNode1", (state) => {
    return { bar: "bar" };
  })
  .addNode("subgraphNode2", (state) => {
    return { foo: state.foo + state.bar };
  })
  .addEdge(START, "subgraphNode1")
  .addEdge("subgraphNode1", "subgraphNode2");
const subgraph = subgraphBuilder.compile();

// 定义父图
const ParentState = z.object({
  foo: z.string(),
});

const builder = new StateGraph(ParentState)
  .addNode("node1", (state) => {
    return { foo: "hi! " + state.foo };
  })
  .addNode("node2", subgraph)
  .addEdge(START, "node1")
  .addEdge("node1", "node2");
const graph = builder.compile();

for await (const chunk of await graph.stream(
  { foo: "foo" },
  {
    streamMode: "updates",
    // 设置subgraphs: true以流式传输子图的输出
    subgraphs: true,
  }
)) {
  console.log(chunk);
}
```

Python**输出**

```
((), {'node_1': {'foo': 'hi! foo'}})
(('node_2:dfddc4ba-c3c5-6887-5012-a243b5b377c2',), {'subgraph_node_1': {'bar': 'bar'}})
(('node_2:dfddc4ba-c3c5-6887-5012-a243b5b377c2',), {'subgraph_node_2': {'foo': 'hi! foobar'}})
((), {'node_2': {'foo': 'hi! foobar'}})
```

**JavaScript输出**

```
[[], {'node1': {'foo': 'hi! foo'}}]
[['node2:dfddc4ba-c3c5-6887-5012-a243b5b377c2'], {'subgraphNode1': {'bar': 'bar'}}]
[['node2:dfddc4ba-c3c5-6887-5012-a243b5b377c2'], {'subgraphNode2': {'foo': 'hi! foobar'}}]
[[], {'node2': {'foo': 'hi! foobar'}}]
```

**注意**我们不仅接收节点更新，还接收命名空间，这些命名空间告诉我们正在从哪个图（或子图）流式传输。

#### 调试

使用`debug`流式传输模式在图执行过程中流式传输尽可能多的信息。流式输出包括节点名称和完整状态。

**Python**

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="debug",
):
    print(chunk)
```

**JavaScript**

```typescript
for await (const chunk of await graph.stream(
  { topic: "ice cream" },
  { streamMode: "debug" }
)) {
  console.log(chunk);
}
```

### LLM令牌

使用`messages`流式传输模式从图的任何部分（包括节点、工具、子图或任务）**逐令牌**流式传输大型语言模型(LLM)的输出。

**Python**

`messages`模式的流式输出是元组`(message_chunk, metadata)`，其中：

- `message_chunk`：来自LLM的令牌或消息段。
- `metadata`：包含图节点和LLM调用详细信息的字典。

> 如果您的LLM没有可用的LangChain集成，可以使用`custom`模式代替流式传输其输出。详见[与任何LLM一起使用](https://langchain-doc.cn/v1/python/langgraph/streaming.html#与任何llm一起使用)部分。

> **Python < 3.11中异步需要手动配置**
> 在使用Python < 3.11的异步代码时，必须显式地将`RunnableConfig`传递给`ainvoke()`以启用正确的流式传输。详见[Python < 3.11中的异步](https://langchain-doc.cn/v1/python/langgraph/streaming.html#async)部分获取详细信息，或升级到Python 3.11+。

```python
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START


@dataclass
class MyState:
    topic: str
    joke: str = ""


model = init_chat_model(model="gpt-4o-mini")

def call_model(state: MyState):
    """调用LLM生成关于某个主题的笑话"""
    # 注意，即使LLM是使用.invoke而不是.stream运行的，也会发出消息事件
    model_response = model.invoke(
        [
            {"role": "user", "content": f"Generate a joke about {state.topic}"}
        ]
    )
    return {"joke": model_response.content}

graph = (
    StateGraph(MyState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)

# "messages"流式传输模式返回元组迭代器(message_chunk, metadata)
# 其中message_chunk是LLM流式传输的令牌，metadata是包含有关LLM调用的图节点信息和其他信息的字典
for message_chunk, metadata in graph.stream(
    {"topic": "ice cream"},
    stream_mode="messages",
):
    if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)
```

**JavaScript**

`messages`模式的流式输出是元组`[message_chunk, metadata]`，其中：

- `message_chunk`：来自LLM的令牌或消息段。
- `metadata`：包含图节点和LLM调用详细信息的字典。

> 如果您的LLM没有可用的LangChain集成，可以使用`custom`模式代替流式传输其输出。详见[与任何LLM一起使用](https://langchain-doc.cn/v1/python/langgraph/streaming.html#与任何llm一起使用)部分。

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, START } from "@langchain/langgraph";
import * as z from "zod";

const MyState = z.object({
  topic: z.string(),
  joke: z.string().default(""),
});

const model = new ChatOpenAI({ model: "gpt-4o-mini" });

const callModel = async (state: z.infer<typeof MyState>) => {
  // 调用LLM生成关于某个主题的笑话
  // 注意，即使LLM是使用.invoke而不是.stream运行的，也会发出消息事件
  const modelResponse = await model.invoke([
    { role: "user", content: `Generate a joke about ${state.topic}` },
  ]);
  return { joke: modelResponse.content };
};

const graph = new StateGraph(MyState)
  .addNode("callModel", callModel)
  .addEdge(START, "callModel")
  .compile();

// "messages"流式传输模式返回元组迭代器[messageChunk, metadata]
// 其中messageChunk是LLM流式传输的令牌，metadata是包含有关LLM调用的图节点信息和其他信息的字典
for await (const [messageChunk, metadata] of await graph.stream(
  { topic: "ice cream" },
  { streamMode: "messages" }
)) {
  if (messageChunk.content) {
    console.log(messageChunk.content + "|");
  }
}
```



#### 按LLM调用过滤

您可以将`tags`与LLM调用关联，以按LLM调用过滤流式传输的令牌。

**Python**

```python
from langchain.chat_models import init_chat_model

# model_1被标记为"joke"
model_1 = init_chat_model(model="gpt-4o-mini", tags=['joke'])
# model_2被标记为"poem"
model_2 = init_chat_model(model="gpt-4o-mini", tags=['poem'])

graph = ... # 定义使用这些LLM的图

# stream_mode设置为"messages"以流式传输LLM令牌
# metadata包含有关LLM调用的信息，包括tags
async for msg, metadata in graph.astream(
    {"topic": "cats"},
    stream_mode="messages",
):
    # 通过metadata中的tags字段过滤流式传输的令牌，仅包含
    # 来自标记为"joke"的LLM调用的令牌
    if metadata["tags"] == ["joke"]:
        print(msg.content, end="|", flush=True)
```

**JavaScript**

```typescript
import { ChatOpenAI } from "@langchain/openai";

// model1被标记为"joke"
const model1 = new ChatOpenAI({
  model: "gpt-4o-mini",
  tags: ['joke']
});
// model2被标记为"poem"
const model2 = new ChatOpenAI({
  model: "gpt-4o-mini",
  tags: ['poem']
});

const graph = // ... 定义使用这些LLM的图

// streamMode设置为"messages"以流式传输LLM令牌
// metadata包含有关LLM调用的信息，包括tags
for await (const [msg, metadata] of await graph.stream(
  { topic: "cats" },
  { streamMode: "messages" }
)) {
  // 通过metadata中的tags字段过滤流式传输的令牌，仅包含
  // 来自标记为"joke"的LLM调用的令牌
  if (metadata.tags?.includes("joke")) {
    console.log(msg.content + "|");
  }
}
```

**扩展示例：按标签过滤**

```python
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph

# joke_model被标记为"joke"
joke_model = init_chat_model(model="gpt-4o-mini", tags=["joke"])
# poem_model被标记为"poem"
poem_model = init_chat_model(model="gpt-4o-mini", tags=["poem"])


class State(TypedDict):
        topic: str
        joke: str
        poem: str


async def call_model(state, config):
        topic = state["topic"]
        print("Writing joke...")
        # 注意：对于python < 3.11，需要显式传递config
        # 因为context var支持在那之前没有添加：https://docs.python.org/3/library/asyncio-task.html#creating-tasks
        # 显式传递config以确保context vars正确传播
        # 当在Python < 3.11中使用异步代码时，这是必需的。请参阅异步部分获取更多详细信息
        joke_response = await joke_model.ainvoke(
              [{"role": "user", "content": f"Write a joke about {topic}"}],
              config,
        )
        print("\n\nWriting poem...")
        poem_response = await poem_model.ainvoke(
              [{"role": "user", "content": f"Write a short poem about {topic}"}],
              config,
        )
        return {"joke": joke_response.content, "poem": poem_response.content}


graph = (
        StateGraph(State)
        .add_node(call_model)
        .add_edge(START, "call_model")
        .compile()
)

# stream_mode设置为"messages"以流式传输LLM令牌
# metadata包含有关LLM调用的信息，包括tags
async for msg, metadata in graph.astream(
        {"topic": "cats"},
        stream_mode="messages",
):
    if metadata["tags"] == ["joke"]:
        print(msg.content, end="|", flush=True)
```

#### 按节点过滤

要仅从特定节点流式传输令牌，请使用`stream_mode="messages"`并按流式传输的metadata中的`langgraph_node`字段过滤输出：

**Python**

```python
# "messages"流式传输模式返回(message_chunk, metadata)元组
# 其中message_chunk是LLM流式传输的令牌，metadata是包含有关LLM调用的图节点信息和其他信息的字典
for msg, metadata in graph.stream(
    inputs,
    stream_mode="messages",
):
    # 通过metadata中的langgraph_node字段过滤流式传输的令牌
    # 仅包含来自指定节点的令牌
    if msg.content and metadata["langgraph_node"] == "some_node_name":
        ...
```

**JavaScript**

```typescript
// "messages"流式传输模式返回[messageChunk, metadata]元组
// 其中messageChunk是LLM流式传输的令牌，metadata是包含有关LLM调用的图节点信息和其他信息的字典
for await (const [msg, metadata] of await graph.stream(
  inputs,
  { streamMode: "messages" }
)) {
  // 通过metadata中的langgraph_node字段过滤流式传输的令牌
  // 仅包含来自指定节点的令牌
  if (msg.content && metadata.langgraph_node === "some_node_name") {
    // ...
  }
}
```

**扩展示例：从特定节点流式传输LLM令牌**

**Python**

```python
from typing import TypedDict
from langgraph.graph import START, StateGraph
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")


class State(TypedDict):
        topic: str
        joke: str
        poem: str


def write_joke(state: State):
        topic = state["topic"]
        joke_response = model.invoke(
              [{"role": "user", "content": f"Write a joke about {topic}"}]
        )
        return {"joke": joke_response.content}


def write_poem(state: State):
        topic = state["topic"]
        poem_response = model.invoke(
              [{"role": "user", "content": f"Write a short poem about {topic}"}]
        )
        return {"poem": poem_response.content}


graph = (
        StateGraph(State)
        .add_node(write_joke)
        .add_node(write_poem)
        # 同时编写笑话和诗歌
        .add_edge(START, "write_joke")
        .add_edge(START, "write_poem")
        .compile()
)

# "messages"流式传输模式返回(message_chunk, metadata)元组
# 其中message_chunk是LLM流式传输的令牌，metadata是包含有关LLM调用的图节点信息和其他信息的字典
for msg, metadata in graph.stream(
    {"topic": "cats"},
    stream_mode="messages",
):
    # 通过metadata中的langgraph_node字段过滤流式传输的令牌
    # 仅包含来自write_poem节点的令牌
    if msg.content and metadata["langgraph_node"] == "write_poem":
        print(msg.content, end="|", flush=True)
```

**JavaScript**

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, START } from "@langchain/langgraph";
import * as z from "zod";

const model = new ChatOpenAI({ model: "gpt-4o-mini" });

const State = z.object({
  topic: z.string(),
  joke: z.string(),
  poem: z.string(),
});

const graph = new StateGraph(State)
  .addNode("writeJoke", async (state) => {
    const topic = state.topic;
    const jokeResponse = await model.invoke([
      { role: "user", content: `Write a joke about ${topic}` }
    ]);
    return { joke: jokeResponse.content };
  })
  .addNode("writePoem", async (state) => {
    const topic = state.topic;
    const poemResponse = await model.invoke([
      { role: "user", content: `Write a short poem about ${topic}` }
    ]);
    return { poem: poemResponse.content };
  })
  // 同时编写笑话和诗歌
  .addEdge(START, "writeJoke")
  .addEdge(START, "writePoem")
  .compile();

// "messages"流式传输模式返回[messageChunk, metadata]元组
// 其中messageChunk是LLM流式传输的令牌，metadata是包含有关LLM调用的图节点信息和其他信息的字典
for await (const [msg, metadata] of await graph.stream(
  { topic: "cats" },
  { streamMode: "messages" }
)) {
  // 通过metadata中的langgraph_node字段过滤流式传输的令牌
  // 仅包含来自writePoem节点的令牌
  if (msg.content && metadata.langgraph_node === "writePoem") {
    console.log(msg.content + "|");
  }
}
```

### 流式传输自定义数据

#### **Python**

要从LangGraph节点或工具内部发送**自定义用户定义数据**，请按照以下步骤操作：

1. 使用`get_stream_writer`访问流写入器并发出自定义数据。
2. 在调用`.stream()`或`.astream()`时设置`stream_mode="custom"`以在流中获取自定义数据。您可以组合多种模式（例如，`["updates", "custom"]`），但至少有一种必须是`"custom"`。

> **Python < 3.11中异步无法使用`get_stream_writer`**
> 在Python < 3.11上运行的异步代码中，`get_stream_writer`将无法工作。
> 相反，请在您的节点或工具中添加`writer`参数并手动传递它。
> 详见[Python < 3.11中的异步](https://langchain-doc.cn/v1/python/langgraph/streaming.html#async)部分获取使用示例。

##### 在节点中流式传输自定义数据

```python
from typing import TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START

class State(TypedDict):
    query: str
    answer: str

def node(state: State):
    # 获取流写入器以发送自定义数据
    writer = get_stream_writer()
    # 发出自定义键值对（例如，进度更新）
    writer({"custom_key": "Generating custom data inside node"})
    return {"answer": "some data"}

graph = (
    StateGraph(State)
    .add_node(node)
    .add_edge(START, "node")
    .compile()
)

inputs = {"query": "example"}

# 设置stream_mode="custom"以在流中接收自定义数据
for chunk in graph.stream(inputs, stream_mode="custom"):
    print(chunk)
```

##### 在工具中流式传输自定义数据

```python
from langchain.tools import tool
from langgraph.config import get_stream_writer

@tool
def query_database(query: str) -> str:
    """查询数据库。"""
    # 访问流写入器以发送自定义数据
    writer = get_stream_writer()
    # 发出自定义键值对（例如，进度更新）
    writer({"data": "Retrieved 0/100 records", "type": "progress"})
    # 执行查询
    # 发出另一个自定义键值对
    writer({"data": "Retrieved 100/100 records", "type": "progress"})
    return "some-answer"


graph = ... # 定义使用此工具的图

# 设置stream_mode="custom"以在流中接收自定义数据
for chunk in graph.stream(inputs, stream_mode="custom"):
    print(chunk)
```

### 与任何LLM一起使用

#### Python

您可以使用`stream_mode="custom"`从**任何LLM API**流式传输数据——即使该API**不**实现LangChain聊天模型接口。

这使您能够集成原始LLM客户端或提供自己流式传输接口的外部服务，使LangGraph对于自定义设置非常灵活。

```python
from langgraph.config import get_stream_writer

def call_arbitrary_model(state):
    """调用任意模型并流式传输输出的示例节点"""
    # 获取流写入器以发送自定义数据
    writer = get_stream_writer()
    # 假设您有一个生成块的流式客户端
    # 使用自定义流式客户端生成LLM令牌
    for chunk in your_custom_streaming_client(state["topic"]):
        # 使用writer将自定义数据发送到流
        writer({"custom_llm_chunk": chunk})
    return {"result": "completed"}

graph = (
    StateGraph(State)
    .add_node(call_arbitrary_model)
    # 根据需要添加其他节点和边
    .compile()
)
# 设置stream_mode="custom"以在流中接收自定义数据
for chunk in graph.stream(
    {"topic": "cats"},
    stream_mode="custom",
):
    # chunk将包含从llm流式传输的自定义数据
    print(chunk)
```

#### JavaScript

您可以使用`streamMode: "custom"`从**任何LLM API**流式传输数据——即使该API**不**实现LangChain聊天模型接口。

这使您能够集成原始LLM客户端或提供自己流式传输接口的外部服务，使LangGraph对于自定义设置非常灵活。

```typescript
import { LangGraphRunnableConfig } from "@langchain/langgraph";

const callArbitraryModel = async (
  state: any,
  config: LangGraphRunnableConfig
) => {
  // 调用任意模型并流式传输输出的示例节点
  // 假设您有一个生成块的流式客户端
  // 使用自定义流式客户端生成LLM令牌
  for await (const chunk of yourCustomStreamingClient(state.topic)) {
    // 使用writer将自定义数据发送到流
    config.writer({ custom_llm_chunk: chunk });
  }
  return { result: "completed" };
};

const graph = new StateGraph(State)
  .addNode("callArbitraryModel", callArbitraryModel)
  // 根据需要添加其他节点和边
  .compile();

// 设置streamMode: "custom"以在流中接收自定义数据
for await (const chunk of await graph.stream(
  { topic: "cats" },
  { streamMode: "custom" }
)) {
  // chunk将包含从llm流式传输的自定义数据
  console.log(chunk);
}
```

#### 扩展示例：流式传输任意聊天模型

```python
import operator
import json

from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START

from openai import AsyncOpenAI

openai_client = AsyncOpenAI()
model_name = "gpt-4o-mini"


async def stream_tokens(model_name: str, messages: list[dict]):
    response = await openai_client.chat.completions.create(
        messages=messages, model=model_name, stream=True
    )
    role = None
    async for chunk in response:
        delta = chunk.choices[0].delta

        if delta.role is not None:
            role = delta.role

        if delta.content:
            yield {"role": role, "content": delta.content}


# 这是我们的工具
async def get_items(place: str) -> str:
    """使用此工具列出您询问的地方可能找到的物品。"""
    writer = get_stream_writer()
    response = ""
    async for msg_chunk in stream_tokens(
        model_name,
        [
            {
                "role": "user",
                "content": (
                    "Can you tell me what kind of items "
                    f"i might find in the following place: '{place}'. "
                    "List at least 3 such items separating them by a comma. "
                    "And include a brief description of each item."
                ),
            }
        ],
    ):
        response += msg_chunk["content"]
        writer(msg_chunk)

    return response


class State(TypedDict):
    messages: Annotated[list[dict], operator.add]


# 这是工具调用图节点
async def call_tool(state: State):
    ai_message = state["messages"][-1]
    tool_call = ai_message["tool_calls"][-1]

    function_name = tool_call["function"]["name"]
    if function_name != "get_items":
        raise ValueError(f"Tool {function_name} not supported")

    function_arguments = tool_call["function"]["arguments"]
    arguments = json.loads(function_arguments)

    function_response = await get_items(**arguments)
    tool_message = {
        "tool_call_id": tool_call["id"],
        "role": "tool",
        "name": function_name,
        "content": function_response,
    }
    return {"messages": [tool_message]}


graph = (
    StateGraph(State)
    .add_node(call_tool)
    .add_edge(START, "call_tool")
    .compile()
)
```

让我们使用包含工具调用的`AIMessage`调用图：
```python
inputs = {
    "messages": [
        {
            "content": None,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "arguments": '{"place":"bedroom"}',
                        "name": "get_items",
                    },
                    "type": "function",
                }
            ],
        }
    ]
}

async for chunk in graph.astream(
    inputs,
    stream_mode="custom",
):
    print(chunk["content"], end="|", flush=True)
```

### 为特定聊天模型禁用流式传输

如果您的应用程序混合使用支持流式传输和不支持流式传输的模型，则可能需要明确为不支持流式传输的模型禁用流式传输。

在初始化模型时设置`disable_streaming=True`。

**使用init_chat_model**

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    # 设置disable_streaming=True以禁用聊天模型的流式传输
    disable_streaming=True
)
```

**使用聊天模型接口**

```python
from langchain_openai import ChatOpenAI

# 设置disable_streaming=True以禁用聊天模型的流式传输
model = ChatOpenAI(model="o1-preview", disable_streaming=True)
```

#### Python < 3.11中的异步

在Python版本< 3.11中，[asyncio任务](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)不支持`context`参数。
这限制了LangGraph自动传播上下文的能力，并在两个关键方面影响LangGraph的流式传输机制：

1. 您**必须**显式地将[`RunnableConfig`](https://python.langchain.com/docs/concepts/runnables/#runnableconfig)传递给异步LLM调用（例如，`ainvoke()`），因为回调不会自动传播。
2. 您**不能**在异步节点或工具中使用`get_stream_writer`——您必须直接传递`writer`参数。

##### 扩展示例：带有手动配置的异步LLM调用

```python
todo
```

## 中断

略



## 使用状态回溯

