# wsl 配置代理



## 现状

- tiny-agents + huggingface_hub 想用 `"provider": "openai"` → 直接访问 `api.openai.com:443`
- 你在国内 Windows + Anaconda + aiohttp，报错：

```
OSError: [WinError 121] 信号灯超时时间已到
```

- 说明 **aiohttp 的 SSL 连接在 Windows 下被本地网络 / 防火墙 / NAT / 代理拦截了**
- 设置 `HTTP_PROXY/HTTPS_PROXY/AIOHTTP_TRUST_ENV` 后仍然不行 → 代理没被 aiohttp 正确使用
- curl 可以访问 → 说明 **你的代理本身没问题**



## 原因

1. **aiohttp + Windows ProactorEventLoop** 有已知问题，WinError 121 会在 SSL + SOCKS/HTTP CONNECT 代理下频繁出现
2. tiny-agents / huggingface_hub 内部创建的 ClientSession 没法直接指定 SOCKS 代理
3. Windows PowerShell + Anaconda 环境下全局代理环境变量不一定被 asyncio 使用

> 结论：**在 Windows 下，用 tiny-agents + GPT-4o + 国内代理几乎不可能稳定运行**



## 可行方案（换环境）

- 用 **WSL2 / Linux / Docker** 运行 tiny-agents
- 在 Linux 下设置 SOCKS5 / HTTP 代理，aiohttp 可以正确走代理



## wsl 环境配置

### tiny-agents 配置

```sh
tiny-agents run agent.json 
tiny-agents: command not found
```

现在在 **WSL2 / Ubuntu 子系统** 里执行命令，但是系统报：

```
tiny-agents: command not found
```

这说明 **tiny-agents 没有在 WSL2 的环境里安装或 PATH 没包含 tiny-agents**。

**解决步骤**

1️⃣ 确认 Python 和 pip

在 WSL2 里检查：

```
python3 --version
pip3 --version
```

确保你在 WSL2 下有 Python 3.10+。

------

2️⃣ 安装 tiny-agents

在 WSL2 里执行：

```
pip3 install "smolagents[mcp]" "gradio[mcp]" mcp fastmcp
```

> 这和你 Windows 里安装的不是同一个环境，WSL2 里要单独安装。

------

3️⃣ 检查 tiny-agents 是否在 PATH

安装完成后：

```
which tiny-agents
```

- 会输出类似 `/home/username/.local/bin/tiny-agents`
- 如果没有输出，需要加 PATH：

```
export PATH=$PATH:~/.local/bin
```

- 可以把上面命令加到 `~/.bashrc` 或 `~/.zshrc` 里永久生效

------

4️⃣ 运行 agent.json

```
tiny-agents run agent.json
```

> 现在应该可以直接在 WSL2 下使用 GPT-4o，且 aiohttp 的代理在 Linux 下更稳定，不会再报 WinError 121。



### 代理配置

2️⃣ 在 Ubuntu 里设置代理（临时，立刻生效）

```
export HTTP_PROXY="socks5h://192.168.1.4:4781"
export HTTPS_PROXY="socks5h://192.168.1.4:4781"
export ALL_PROXY="socks5h://192.168.1.4:4781"
export NO_PROXY="localhost,127.0.0.1"
export AIOHTTP_TRUST_ENV=1
```

👉 **注意：是 `export`，不是 `$Env:`**

------

3️⃣ 立刻验证（关键）

```
curl https://api.openai.com/v1/models
```

如果看到：

```
Missing bearer authentication
```

🎉 **恭喜，WSL → SOCKS5 → OpenAI 已经通了**

------

✅ 让它「永久生效」（推荐）

编辑 `~/.bashrc` 或 `~/.zshrc`

```
nano ~/.bashrc
```

加到最后：

```
export HTTP_PROXY="socks5h://192.168.1.4:4781"
export HTTPS_PROXY="socks5h://192.168.1.4:4781"
export ALL_PROXY="socks5h://192.168.1.4:4781"
export NO_PROXY="localhost,127.0.0.1"
export AIOHTTP_TRUST_ENV=1
```

生效：

```
source ~/.bashrc
```



### 在 WSL 的 Python 环境里安装 socksio（按需）

1. **在 WSL 的 Python 环境里安装 socksio**

```
# 推荐安装 httpx 及其 socks 支持
pip install "httpx[socks]" --upgrade
```

> 这个命令会安装：
>
> - `httpx` 核心库
> - `socksio`（必需的 SOCKS 支持库）

1. **验证安装成功**

```
python -c "import socksio; print(socksio.__version__)"
```

如果能打印版本号，说明安装成功。

1. **重新设置代理环境变量（WSL 内）**

```
export HTTP_PROXY="socks5h://192.168.1.4:4781"
export HTTPS_PROXY="socks5h://192.168.1.4:4781"
export ALL_PROXY="socks5h://192.168.1.4:4781"
export NO_PROXY="localhost,127.0.0.1"
export AIOHTTP_TRUST_ENV=1
```







## WSL 走宿主机代理（推荐）

```sh
host_ip=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')

export http_proxy=http://$host_ip:4780
export https_proxy=http://$host_ip:4780
export ALL_PROXY=socks5://$host_ip:4781

npm config set proxy http://$host_ip:4780
npm config set https-proxy http://$host_ip:4780

# 测试是否通
curl https://www.google.com -I
```



你需要 **“开机自动生效”方案**

### 方案1（推荐）：写进 bashrc

```sh
echo 'export host_ip=$(awk "/nameserver/ {print \$2}" /etc/resolv.conf)' >> ~/.bashrc

echo 'export http_proxy=http://$host_ip:4780' >> ~/.bashrc
echo 'export https_proxy=http://$host_ip:4780' >> ~/.bashrc
echo 'export ALL_PROXY=socks5://$host_ip:4781' >> ~/.bashrc

# 然后生效：
source ~/.bashrc
```



### 🚀 方案2（更稳）：自动更新 IP + proxy（推荐）

因为 WSL 的 IP 可能会变，所以推荐这样：

```sh
cat << 'EOF' >> ~/.bashrc

update_proxy() {
  export host_ip=$(awk '/nameserver/ {print $2}' /etc/resolv.conf)

  export http_proxy=http://$host_ip:4780
  export https_proxy=http://$host_ip:4780
  export ALL_PROXY=socks5://$host_ip:4781
}

update_proxy
EOF
```

# npm 也要单独持久化

```
npm config set proxy http://$host_ip:4780
npm config set https-proxy http://$host_ip:4780
```

但注意：npm 不会自动刷新 host_ip，所以更稳的方法是：

```
npm config set proxy http://172.24.96.1:4780
npm config set https-proxy http://172.24.96.1:4780
```

（直接写死 IP）
