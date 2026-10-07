# TongAgents



## TongAgents 工作流框架介绍

| 维度              | TongAgents                                                | LangGraph                                  |
| ----------------- | --------------------------------------------------------- | ------------------------------------------ |
| **图结构编排**    | 用节点（Node）和边（Edge）定义工作流                      | 同样用 Node + Edge 构建 StateGraph         |
| **声明式节点**    | `@node_declare(name=..., edges=[...])` 装饰器声明节点     | `graph.add_node("name", func)` 注册节点    |
| **特殊标记**      | 有 `START` / `END` 标记入口和出口                         | 同样有 `START` / `END` 常量                |
| **条件路由**      | 节点函数返回 `TransferCommand` 决定下一步走向             | 通过 `add_conditional_edges` 实现条件分支  |
| **多 Agent 协作** | 支持 `MasterTowerAgent + SlaveTowerAgent` 的多 Agent 拓扑 | 原生支持 Supervisor、Swarm 等多 Agent 模式 |
| **事件驱动**      | 节点之间通过事件（Event Dict）传递数据                    | State 在节点之间流转                       |

## 面试用语