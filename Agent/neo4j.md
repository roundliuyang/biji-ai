# neo4j

## docker 安装

```sh
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7789:7687 \
  -v /home/neo4j/data:/data \
  -v /home/neo4j/logs:/logs \
  -v /home/neo4j/import:/import \
  -v /home/neo4j/plugins:/plugins \
  -e NEO4J_AUTH=neo4j/12345678 \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
  -e NEO4J_dbms_security_procedures_allowlist=apoc.* \
  neo4j:latest
```

