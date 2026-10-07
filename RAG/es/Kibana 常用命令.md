# Kibana 常用命令

```sh
kibana常用命令：

1. 查询 Elasticsearch 中的索引状态

GET /_cat/indices?v


2. 查看有哪些索引

GET /_cat/indices?v&index=*


3. 查询某个索引下的所有数据，默认只返回前 10 个文档

GET /my_index/_search
{
  "query": {
    "match_all": {}
  }
}


4. 设置返回特定字段的数据

GET /my_index/_search
{
  "query": {
    "match_all": {}
  },
  "_source": ["field1", "field2"]
}
```

