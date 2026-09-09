# Langgraph Checkpointer底层原理

[Langgraph Checkpointer底层原理](https://bryce.wang/posts/2025-11-13-langgraph-checkpointer%E5%BA%95%E5%B1%82%E5%8E%9F%E7%90%86.html)

## 核心概念

> 官方文档：[LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)

LangGraph 内置了持久层，通过 **Checkpoint** 实现。当编译 graph 时传入 checkpointer，它会在每个 **Super Step** 处保存一份 graph state 的 checkpoint，这些 checkpoint 会关联到对应的 **thread**。
![img](Langgraph Checkpointer底层原理.assets/file-20251113224409005.png)

- **Thread ID** <=> Session ID、Chat ID
- **Checkpoint** <=> Snapshot

使用 checkpointer 时必须传入的唯一配置就是：

```python
{"configurable": {"thread_id": "1"}}
```



## Checkpoint 数据结构

在 Graph API 层面表现为 `StateSnapshot`：

```python
class StateSnapshot(NamedTuple):
    values: dict[str, Any] | Any          # 当前 channel 值
    next: tuple[str, ...]                  # 下一步要执行的节点
    config: RunnableConfig                 # 配置
    metadata: CheckpointMetadata | None    # 元数据
    created_at: str | None                 # 创建时间
    parent_config: RunnableConfig | None   # 父 checkpoint 配置
    tasks: tuple[PregelTask, ...]          # 待执行任务
    interrupts: tuple[Interrupt, ...]      # 待处理中断
```

**关键点**：在非并行情况下，checkpoint 会在每个node执行之前创建，然后在node执行之后更新。

有了 checkpoint，LangGraph 才能支持这些高级特性：

- **Human-in-the-loop**：人类可随时查看/修改状态
- **Memory**：记忆能力
- **Time Travel**：重放特定步骤，从任意 checkpoint 分叉
- **Fault-tolerance**：从上一个 SuperStep 重新执行
- **Pending writes**：不重新运行已成功的节点

### Pregel Model

> 官方文档：[LangGraph Pregel](https://docs.langchain.com/oss/python/langgraph/pregel)

LangGraph 底层使用 **Pregel** 作为运行基础，取名源自 Google 的经典图计算框架最早是为大规模图计算（如 PageRank、广度优先搜索等）设计的，核心思想是：

> **Think Like a Vertex**

每个顶点是独立的计算单元（actor），顶点之间通过有向边（channel）通信。整个计算由一系列 **Super Step** 组成：

1. **Plan**：决定哪些 actor 将执行
2. **Execution**：所有 actor 并行执行，互相看不到对方的计算过程
3. **Update**：根据执行结果更新对应的 channel

重复执行直到没有 actor 被选中或达到循环上限。

### Pregel Channel

Graph API 和 Functional API 构建的 graph 最终都会编译为 `Pregel` 对象。核心是 `PregelNode`（也叫`Actor`，有点像LangChain 的 `Runnable` 类似） 和 `BaseChannel`。

LangGraph 提供的 channel 类型：（可能不太全？）

- **LastValue**：存储最新值，每个 superstep 只能收到 1 个值
- **Topic**：PubSub Topic，可接收多值，广播给多接收者
- **BinaryOperatorAggregate**：存储一个持久值，通过将二元运算符应用于当前值和发送到通道的每个更新来更新该值，适用于在多个步骤中计算聚合

举个例子，假设有这样的 State：

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class ChatFlowState(MessagesState):
    node_results: List[NodeExecutionResult]
    current_node_name: Optional[str]
    execution_path: List[str]
    extra_info: Optional[StateExtraInfo]
```

编译后的 channels：

```python
{
    'messages': BinaryOperatorAggregate,  # 因为有 Annotated
    'node_results': LastValue,
    'current_node_name': LastValue,
    'execution_path': LastValue,
    'extra_info': LastValue,
    '__start__': EphemeralValue,
    '__pregel_tasks': Topic,
    'branch:to:xxx': EphemeralValue,  # 各节点的分支
}
```

**⚠️ 踩坑提醒**：LangGraph **不会解析嵌套的 Annotated 字段**！

```python
class SubState(TypedDict):
    cnt: Annotated[int, operator.add]  # 这个不会生效！
    res: Annotated[str, operator.concat]  # 这个也不会！

class State(TypedDict):
    sub_state: SubState  # 整个会被当作 LastValue
```

实测 `sub_state` 内部的累加和拼接完全不起作用。

### 持久化模式

> 官方文档：[LangGraph Durable Execution](https://docs.langchain.com/oss/python/langgraph/durable-execution)

**三种持久化模式**：

- **exit**：仅当 graph 执行完成（成功/错误/interrupt）时才保存
- **async**：执行过程中异步提交
- **sync**：执行过程中同步提交

**序列化**：默认使用 `JsonPlusSerializer`（底层是 [ormsgpack](https://github.com/aviramha/ormsgpack)，Rust 实现，性能很强）

### CheckpointSaver

所有 checkpoint 持久化操作封装在 `BaseCheckpointSaver`：

| 方法             | 功能                                              |
| ---------------- | ------------------------------------------------- |
| `.put`           | 写入 checkpoint                                   |
| `.put_writes`    | 写入中间结果（pending writes）                    |
| `.get_tuple`     | 获取 checkpoint tuple（`graph.get_state()` 底层） |
| `.list`          | 批量查询（`graph.get_state_history()` 底层）      |
| `.delete_thread` | 删除 thread 关联的所有 checkpoints                |



### PostgresSaver 存储结构

PostgresSaver 使用三张核心表：

**checkpoints 表**：每个 checkpoint 一条记录

```sql
CREATE TABLE "checkpoints" (
  "thread_id" text NOT NULL,
  "checkpoint_ns" text NOT NULL DEFAULT '',
  "checkpoint_id" text NOT NULL,
  "parent_checkpoint_id" text,
  "type" text,
  "checkpoint" jsonb NOT NULL,
  "metadata" jsonb NOT NULL DEFAULT '{}',
  PRIMARY KEY ("thread_id", "checkpoint_ns", "checkpoint_id")
);
```

**checkpoint_blobs 表**：非基础类型值序列化存储

```sql
CREATE TABLE "checkpoint_blobs" (
  "thread_id" text NOT NULL,
  "checkpoint_ns" text NOT NULL DEFAULT '',
  "channel" text NOT NULL,
  "version" text NOT NULL,
  "type" text NOT NULL,
  "blob" bytea,
  PRIMARY KEY ("thread_id", "checkpoint_ns", "channel", "version")
);
```

**checkpoint_writes 表**：运行时 channel 中间数据

```sql
CREATE TABLE "checkpoint_writes" (
  "thread_id" text NOT NULL,
  "checkpoint_ns" text NOT NULL DEFAULT '',
  "checkpoint_id" text NOT NULL,
  "task_id" text NOT NULL,
  "idx" int4 NOT NULL,
  "channel" text NOT NULL,
  "type" text,
  "blob" bytea NOT NULL,
  "task_path" text NOT NULL DEFAULT '',
  PRIMARY KEY ("thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx")
);
```



### 实测对比

用一个简单的图测试：`START → node_a → node_user(interrupt) → node_b → node_end`

**async 模式**（执行到 interrupt）：

- checkpoints 表：3 条记录
- checkpoint_blobs 表：3 条记录
- checkpoint_writes 表：7 条记录

**exit 模式**（执行到 interrupt）：

- checkpoints 表：**1 条记录**
- checkpoint_blobs 表：**1 条记录**
- checkpoint_writes 表：**1 条记录**（仅 interrupt）

差距还是很明显的！

### 总结

1. LangGraph **只解析 StateSchema 第一层**，嵌套的 Annotated 不起作用
2. `compile()` 操作将图解析成 Pregel 构建的图
3. StateSchema 第一层字段都会被解析为 channel
4. 每个 SuperStep 会并发处理当前轮次操作（但我们目前没有并行场景）
5. 使用 GraphAPI 时，进入 START 节点和用户定义节点之前会生成 checkpoint
6. 非基础类型字段序列化后存 `checkpoint_blobs`，基础字段直接存 `checkpoints`
7. SuperStep 执行中每个 channel 写入会保存到 `checkpoint_writes`
8. **exit 模式**写入次数大幅下降，代价是无法从中间状态恢复
9. 极致优化应减少 State 字段数量，控制写入数据量
10. PostgreSQL 性能很强，存的都是 byte/jsonb 高效格式

## 实际示例

### **代码**

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from typing import TypedDict

DB_URI = "postgresql://postgres:123456@localhost:5432/postgres?sslmode=disable"

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

with PostgresSaver.from_conn_string(DB_URI) as saver:
    saver.setup()  # 初始化表结构

    # 编译图时启用检查点
    graph = graph.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "2"}}
    graph.invoke({"foo": "", "bar": []}, config=config)

    # 获取最新的状态快照
    latest_state = graph.get_state(config)
    print("最新状态:", latest_state.values)
    print("checkpoint_id:", latest_state.config['configurable']['checkpoint_id'])

    # 获取特定检查点 ID 的状态快照
    config = {
      "configurable": {
        "thread_id": "2",
        "checkpoint_id": latest_state.config['configurable']['checkpoint_id'],
      },
    }
    print("\n特定检查点状态:", graph.get_state(config))
```

这是一个简单的 checkpoint 链，图执行结束时 checkpoints 表里会有四条记录。

```
START
  |
 node_a
  |
 node_b
  |
 END
```



### **第1条**

```python
{
	"thread_id": "2",
	"checkpoint_ns": "",
	"checkpoint_id": "1f18e5fd-3d2f-64de-bfff-329c9014a46e",
	"parent_checkpoint_id": null,
	"type": "",
	"checkpoint": {
		"v": 4,
		"id": "1f18e5fd-3d2f-64de-bfff-329c9014a46e",
		"ts": "2026-08-02T10:49:25.205116+00:00",
		"versions_seen": {
			"__input__": {}
		},
		"channel_values": {},
		"channel_versions": {
			"__start__": "00000000000000000000000000000001.0.17069529317274867"
		},
		"updated_channels": [
			"__start__"
		]
	},
	"metadata": {
		"step": -1,
		"source": "input",
		"parents": {}
	}
}
```

这第一条记录是 **LangGraph 执行开始前的初始化 checkpoint**。

它不是某个节点执行后的结果，而是：

> 用户输入 graph 后，Pregel 创建的第一个 checkpoint，用来记录“初始状态”和启动事件。

可以理解为：

```
用户输入
   ↓
创建初始 checkpoint (step=-1)
   ↓
开始第一个 SuperStep
```

#### 外层字段

##### 1. thread_id

```
"thread_id": "2"
```

含义：

> 当前会话线程 ID。

LangGraph 中一个运行实例对应一个 thread。

例如：

```
config = {
    "configurable": {
        "thread_id": "2"
    }
}
```

所有 checkpoint 都挂在这个 thread 下。

数据库：

```
thread_id = 2
```

表示：这是同一次对话/任务执行产生的历史。

例如：

```
thread_id=2

checkpoint_1
checkpoint_2
checkpoint_3
```

形成一条状态演进链。

##### 2. checkpoint_ns

```
"checkpoint_ns": ""
```

含义：

> checkpoint 命名空间。

namespace 用于区分不同子图（subgraph）。

这里为空表示：**主 graph，没有子 graph**。

##### 3. checkpoint_id

```
"checkpoint_id": "1f18e5fd-3d2f-64de-bfff-329c9014a46e"
```

含义：

> 当前 checkpoint 唯一 ID。

类似：

```
git commit hash
```

每一次状态保存都会生成一个新的 checkpoint_id。

例如：

```
checkpoint A

id:
1f18e5fd...


checkpoint B

id:
1f18e5fd-3d34...


checkpoint C

id:
1f18e5fd-3d39...
```

形成：

```
A
|
B
|
C
```

##### 4. parent_checkpoint_id

```
"parent_checkpoint_id": null
```

含义：

> 父 checkpoint。

表示：

这个 checkpoint 是从哪个 checkpoint 演化来的。

第一条：

```
parent=null
```

因为它是起点。

后面例如第二条：

```
"parent_checkpoint_id":
"1f18e5fd-3d2f-64de-bfff-329c9014a46e"
```

表示：

```
checkpoint1
      |
      |
checkpoint2
```

形成 checkpoint 链。

类似 Git：

```
commit A

    ↓

commit B
```

##### 5. type

```
"type": ""
```

这个字段通常和 checkpoint 类型相关。

LangGraph 内部支持不同 checkpoint 类型。

这里为空表示：

> 普通 checkpoint。

通常不用关注。

##### 6. checkpoint 对象

这个是真正保存的 Pregel 状态。

###### 6.1 v

```
"v":4
```

含义：

> checkpoint 数据结构版本。

LangGraph 内部 checkpoint schema 版本。

类似：

```
{
 "version":4
}
```

用于未来兼容，不是你的 graph 版本。

###### 6.2 id

```
"id":
"1f18e5fd-3d2f-64de-bfff-329c9014a46e"
```

和外层：

```
checkpoint_id
```

一样。内部 checkpoint 自己也保存 ID，方便序列化恢复。

关系：

```
checkpoint 表字段

checkpoint_id
       |
       |
       ↓

checkpoint JSON

id
```

###### 6.3 ts

```
"ts":
"2026-08-02T10:49:25.205116+00:00"
```

timestamp，checkpoint 创建时间。

###### 6.4 versions_seen

```
"versions_seen": {
    "__input__": {}
}
```

表示：

> 当前只是用户输入进入 graph，还没有任何节点开始执行，所以没有记录任何 Actor 已经读取过的 channel 版本。

流程：

```
graph.invoke(input)
        |
        ↓
创建 checkpoint(step=-1)
        |
        ↓
Plan 调度节点
        |
        ↓
Actor 执行时读取 channel
        |
        ↓
versions_seen 记录读取过的版本
```

所以第 1 条时：

- 用户输入已经写入 graph
- 但 `node_a`、`node_b` 都还没运行
- 因此没有“哪个节点读取了哪个 channel”的记录

后面的第 2 条：

```
"versions_seen": {
  "__start__": {
    "__start__": "000...01"
  }
}
```

表示入口 Actor 已经读取过 `__start__` channel 的第 1 个版本。



###### 6.5. channel_values

```
"channel_values": {}
```

含义：

> 当前没有任何 channel 存储业务状态值。

原因：

这是 graph 刚启动的初始化 checkpoint（`step=-1`），还没有节点执行，也没有 State 数据写入 channel。



###### 6.6. channel_versions

```
"channel_versions": {
    "__start__":
    "00000000000000000000000000000001.0.17069529317274867"
}
```

含义：

> 记录当前各个 channel 的版本号。

这里：

- `__start__: "000...01"`
   表示 `__start__` channel 已创建，当前版本为第 1 个版本。

因为这是 graph 刚启动的 checkpoint，只产生了启动 channel，所以只有 `__start__` 的版本记录。



###### 6.7 updated_channels

```
"updated_channels":[
    "__start__"
]
```

含义：

> 表示这一轮只有 `__start__` channel 被更新。

这里：

- `__start__`：LangGraph 内部入口 channel，用于启动 graph 执行。
- 说明当前只是写入了启动信号，还没有业务节点执行。



##### 7. metadata

```
"metadata":{
    "step":-1,
    "source":"input",
    "parents":{}
}
```

这是 checkpoint 的元信息。含义：

- `step: -1`
   表示这是 graph 执行前的初始化 checkpoint，还未进入 Pregel 循环。
- `source: "input"`
   表示该 checkpoint 是由用户输入 graph 时创建的。
- `parents: {}`
   表示没有父级 graph/subgraph 信息。



##### 整体翻译成人话

这一条 checkpoint：

```
{
 thread_id:"2",

 checkpoint_id:"xxx",

 step:-1,

 source:"input"
}
```

意思是：

> thread=2 的一次 graph 执行刚开始，LangGraph 收到了用户输入，创建了第一个状态快照。目前没有节点执行，没有业务数据，只有启动 channel (**start**) 被写入，用于触发后续 Pregel 调度。

对应流程：

```
用户输入
   |
   ↓
checkpoint(step=-1)
   |
   ↓
__start__ channel 写入
   |
   ↓
Pregel 开始调度
   |
   ↓
生成 step=0 checkpoint
```

所以这条记录本质上就是：

**LangGraph 执行链的“创世区块（genesis checkpoint）”。**



### **第2条**

```json
{
	"thread_id": "2",
	"checkpoint_ns": "",
	"checkpoint_id": "1f18e5fd-3d34-630e-8000-815667353dde",
	"parent_checkpoint_id": "1f18e5fd-3d2f-64de-bfff-329c9014a46e",
	"type": "",
	"checkpoint": {
		"v": 4,
		"id": "1f18e5fd-3d34-630e-8000-815667353dde",
		"ts": "2026-08-02T10:49:25.207118+00:00",
		"versions_seen": {
			"__input__": {},
			"__start__": {
				"__start__": "00000000000000000000000000000001.0.17069529317274867"
			}
		},
		"channel_values": {
			"foo": "",
			"branch:to:node_a": null
		},
		"channel_versions": {
			"bar": "00000000000000000000000000000002.0.9091400702205034",
			"foo": "00000000000000000000000000000002.0.9091400702205034",
			"__start__": "00000000000000000000000000000002.0.9091400702205034",
			"branch:to:node_a": "00000000000000000000000000000002.0.9091400702205034"
		},
		"updated_channels": [
			"bar",
			"branch:to:node_a",
			"foo"
		]
	},
	"metadata": {
		"step": 0,
		"source": "loop",
		"parents": {}
	}
}
```

这第二条 checkpoint 是 **第 0 个 SuperStep 执行完成后的状态快照**。

它和第一条最大的区别：

第一条：

```
step=-1
source=input
```

只是 **初始化输入阶段**。

第二条：

```
step=0
source=loop
```

表示：

> Pregel 开始运行，第一个调度周期完成，`__start__` 触发了 `node_a`，并且更新了一些 channel。

整体流程：

```
checkpoint(step=-1)
        |
        | __start__ 信号
        ↓
SuperStep 0
        |
        | 调度 node_a
        |
        ↓
checkpoint(step=0)
```



#### 外层字段

##### thread_id

```
"thread_id": "2"
```

还是同一次 graph 执行。

##### checkpoint_ns

```
"checkpoint_ns": ""
```

表示：当前 checkpoint 属于 main graph，没有子图 。

##### parent_checkpoint_id

```
"parent_checkpoint_id": "1f18e5fd-3d2f-64de-bfff-329c9014a46e"
```

说明，这个 checkpoint 是从上一条 checkpoint 演化来的。

关系：

```
step=-1

checkpoint A
1f18e5fd-3d2f


        ↓


step=0

checkpoint B
1f18e5fd-3d34
```



##### checkpoint 内容

###### versions_seen

```
"versions_seen": {
    "__input__": {},
    "__start__": {
        "__start__":
        "00000000000000000000000000000001.0.17069529317274867"
    }
}
```

表示：

> 记录各个 Actor 已经读取过的 channel 版本。

含义：

- `__input__:{}`
   表示输入阶段没有读取任何 channel。
- `__start__:{ "__start__": "000...01" }`
   表示内部入口 Actor `__start__` 已经读取过 `__start__ channel` 的第 1 个版本。

作用：

> LangGraph 通过 `versions_seen` 判断 Actor 是否已经消费过某个 channel 的数据，避免同一个版本的数据被重复处理。



###### channel_values

```
"channel_values": {
  "foo": "",
  "branch:to:node_a": null
}
```

含义：

- `foo: ""`
   表示当前 `foo` channel 的值为空字符串，来自 graph 初始输入。
- `branch:to:node_a: null`
   表示生成了发送给 `node_a` 的调度信号，用于触发 `node_a` 执行，不携带业务数据。



######  channel_versions

```json
"channel_versions": {
    "bar":
      "00000000000000000000000000000002...",
      
    "foo":
      "00000000000000000000000000000002...",

    "__start__":
      "00000000000000000000000000000002...",

    "branch:to:node_a":
      "00000000000000000000000000000002..."
}
```

表示：

> 每个 channel 当前的版本号。

含义：

- `foo: 000...02`：`foo` channel 在这一轮被写入，版本从 1 升到 2
- `bar: 000...02`：`bar` channel 在这一轮被写入，版本升到 2
- `__start__: 000...02`：启动 channel 状态推进到版本 2
- `branch:to:node_a: 000...02`：产生了发送给 `node_a` 的调度消息，版本升到 2

版本号主要用于 **Pregel 判断节点是否需要读取新的 channel 数据**。

###### updated_channels

```
"updated_channels":[
    "bar",
    "branch:to:node_a",
    "foo"
]
```

表示这一轮有 3 个 channel 被更新：

- `bar`：输入状态里的 `bar=[]` 被写入 channel
- `foo`：输入状态里的 `foo=""` 被写入 channel
- `branch:to:node_a`：`START -> node_a` 的边产生了节点调度消息，触发 `node_a` 执行

注意：此时 **node_a 还没有执行完成**，这里只记录 channel 更新。

##### metadata

```
"metadata": {
    "step": 0,
    "source": "loop",
    "parents": {}
}
```

含义：

- `step: 0`
   表示这是第 0 个 Pregel SuperStep 产生的 checkpoint。
- `source: "loop"`
   表示该 checkpoint 是 LangGraph 执行循环（Pregel loop）过程中自动生成的，不是用户输入生成的。
- `parents: {}`
   表示当前没有父级 graph/subgraph 信息。

### **第3条**

```python
{
	"thread_id": "2",
	"checkpoint_ns": "",
	"checkpoint_id": "1f18e5fd-3d39-611c-8001-32f6fcf4ea2f",
	"parent_checkpoint_id": "1f18e5fd-3d34-630e-8000-815667353dde",
	"type": "",
	"checkpoint": {
		"v": 4,
		"id": "1f18e5fd-3d39-611c-8001-32f6fcf4ea2f",
		"ts": "2026-08-02T10:49:25.209116+00:00",
		"versions_seen": {
			"node_a": {
				"branch:to:node_a": "00000000000000000000000000000002.0.9091400702205034"
			},
			"__input__": {},
			"__start__": {
				"__start__": "00000000000000000000000000000001.0.17069529317274867"
			}
		},
		"channel_values": {
			"foo": "a",
			"branch:to:node_b": null
		},
		"channel_versions": {
			"bar": "00000000000000000000000000000003.0.013864109602768737",
			"foo": "00000000000000000000000000000003.0.013864109602768737",
			"__start__": "00000000000000000000000000000002.0.9091400702205034",
			"branch:to:node_a": "00000000000000000000000000000003.0.013864109602768737",
			"branch:to:node_b": "00000000000000000000000000000003.0.013864109602768737"
		},
		"updated_channels": [
			"bar",
			"branch:to:node_b",
			"foo"
		]
	},
	"metadata": {
		"step": 1,
		"source": "loop",
		"parents": {}
	}
}
```

#### 外层字段

##### checkpoint 内容

###### versions_seen

```
"versions_seen": {
  "node_a": {
    "branch:to:node_a": "000...02"
  },
  "__input__": {},
  "__start__": {
    "__start__": "000...01"
  }
}
```

表示各 Actor 已读取过的 channel 版本。

- `node_a`：已经消费过 `branch:to:node_a` 的第 2 个版本，说明 node_a 已被触发执行。
- `__start__`：已经读取过启动 channel 第 1 个版本。
- `__input__:{}`：输入阶段没有读取 channel。



###### channel_values

```
"channel_values": {
  "foo": "a",
  "branch:to:node_b": null
}
```

说明：

- `foo: "a"`
   表示 `node_a` 执行后返回的状态更新。
- `bar: ["a"]`
   表示 `node_a` 返回的 `bar` 状态也被更新，但没有直接展示在 `channel_values` 中（可能存储在 checkpoint 的 blob 数据中）。
- `branch:to:node_b: null`
   表示生成了发送给 `node_b` 的调度信号，用于触发 `node_b` 执行，不携带业务数据。

###### channel_versions

```
"channel_versions": {
  "bar": "000...03",
  "foo": "000...03",
  "__start__": "000...02",
  "branch:to:node_a": "000...03",
  "branch:to:node_b": "000...03"
}
```

含义：

> 记录当前所有 channel 的版本号。

- `foo: 000...03`
   `node_a` 更新了 `foo`，版本从 2 升到 3。
- `bar: 000...03`
   `node_a` 更新了 `bar`，版本从 2 升到 3。
- `branch:to:node_b: 000...03`
   生成了触发 `node_b` 的调度消息，版本升到 3。
- `branch:to:node_a: 000...03`
   `node_a` 对应的触发 channel 版本推进到 3。
- `__start__: 000...02`
   启动 channel 当前保持第 2 个版本。

作用：

> LangGraph 用 channel 版本号判断哪些节点需要读取新的 channel 数据。

###### updated_channels

```
"updated_channels": [
  "bar",
  "branch:to:node_b",
  "foo"
]
```

含义：

> 表示这一轮 SuperStep 中被更新的 channel。

- `bar`：`node_a` 返回 `{"bar": ["a"]}`，更新了 bar channel。
- `foo`：`node_a` 返回 `{"foo": "a"}`，更新了 foo channel。
- `branch:to:node_b`：生成了发送给 `node_b` 的调度消息，准备执行 node_b。



### **第4条**

```python
 {
    "thread_id": "2",
    "checkpoint_ns": "",
    "checkpoint_id": "1f18e5fd-3d3b-6868-8002-7baf67be79fe",
    "parent_checkpoint_id": "1f18e5fd-3d39-611c-8001-32f6fcf4ea2f",
    "type": "",
    "checkpoint": {
      "v": 4,
      "id": "1f18e5fd-3d3b-6868-8002-7baf67be79fe",
      "ts": "2026-08-02T10:49:25.210122+00:00",
      "versions_seen": {
        "node_a": {
          "branch:to:node_a": "00000000000000000000000000000002.0.9091400702205034"
        },
        "node_b": {
          "branch:to:node_b": "00000000000000000000000000000003.0.013864109602768737"
        },
        "__input__": {},
        "__start__": {
          "__start__": "00000000000000000000000000000001.0.17069529317274867"
        }
      },
      "channel_values": {
        "foo": "b"
      },
      "channel_versions": {
        "bar": "00000000000000000000000000000004.0.16520078818256767",
        "foo": "00000000000000000000000000000004.0.16520078818256767",
        "__start__": "00000000000000000000000000000002.0.9091400702205034",
        "branch:to:node_a": "00000000000000000000000000000003.0.013864109602768737",
        "branch:to:node_b": "00000000000000000000000000000004.0.16520078818256767"
      },
      "updated_channels": [
        "bar",
        "foo"
      ]
    },
    "metadata": {
      "step": 2,
      "source": "loop",
      "parents": {}
    }
  }
```

#### 外层字段

##### checkpoint

###### versions_seen

```
{
  "node_a": {
    "branch:to:node_a": "000...02"
  },
  "node_b": {
    "branch:to:node_b": "000...03"
  }
}
```

表示各 Actor 已读取过的 channel 版本：

- `node_a`：已消费 `branch:to:node_a` 第 2 个版本。
- `node_b`：已消费 `branch:to:node_b` 第 3 个版本，说明 node_b 已执行。



###### channel_values

```
{
  "foo": "b"
}
```

当前 channel 保存的值：

- `foo="b"`：`node_b` 执行后返回的结果。
- `bar=["b"]` 也被更新，但未直接显示在这里。



###### channel_versions

含义：

> 记录当前所有 channel 的版本号。

- `foo: ...04`：foo 被 node_b 更新。
- `bar: ...04`：bar 被 node_b 更新。
- `branch:to:node_b: ...04`：node_b 的调度 channel 更新。
- `branch:to:node_a: ...03`：保持之前版本。
- `__start__`：启动 channel 版本。



###### updated_channels

```
[
 "bar",
 "foo"
]
```

表示这一轮更新的 channel：

- `bar`：node_b 返回的新值。
- `foo`：node_b 返回的新值。

没有新的节点调度，因此没有 `branch:to:*`。



### checkpoint_blobs

上面代码执行后，checkpoint_blobs 表里会有四条数据

```python
[
  {
    "thread_id": "2",
    "checkpoint_ns": "",
    "channel": "__start__",
    "version": "00000000000000000000000000000001.0.17069529317274867",
    "type": "msgpack",
    "blob": "binary data"
  },
  {
    "thread_id": "2",
    "checkpoint_ns": "",
    "channel": "bar",
    "version": "00000000000000000000000000000002.0.9091400702205034",
    "type": "msgpack",
    "blob": "binary data"
  },
  {
    "thread_id": "2",
    "checkpoint_ns": "",
    "channel": "bar",
    "version": "00000000000000000000000000000003.0.013864109602768737",
    "type": "msgpack",
    "blob": "binary data"
  },
  {
    "thread_id": "2",
    "checkpoint_ns": "",
    "channel": "bar",
    "version": "00000000000000000000000000000004.0.16520078818256767",
    "type": "msgpack",
    "blob": "binary data"
  }
]
```

`checkpoint_blobs` **不是和 checkpoint 一一对应的表，而是用于保存需要单独存储的 channel 数据版本。**

你的：

```
bar: list[str]
```

是一个复杂类型（list），LangGraph 会将它序列化后存储到 `checkpoint_blobs`。

这 4 条记录表示：

1. `__start__`
    保存 graph 启动时 `__start__` channel 的版本和值。

2. `bar` version `000...02`
    保存初始输入时：

   ```
   bar = []
   ```

3. `bar` version `000...03`
    保存 `node_a` 执行后的：

   ```
   bar = ["a"]
   ```

4. `bar` version `000...04`
    保存 `node_b` 执行后的：

   ```
   bar = ["b"]
   ```

总结：

> `checkpoints` 保存 checkpoint 状态和版本信息；`checkpoint_blobs` 保存需要序列化存储的 channel 具体值。这里主要是保存 `bar` 每次更新后的 list 数据。

