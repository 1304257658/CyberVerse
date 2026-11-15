# 标准管道工具调用与 720 头像测试说明

## 工具调用（DeepSeek 标准模式）

### 已完成（可在本仓库开发与单测验证）

- `OpenAILLMPlugin` 增加有界工具循环：轮次、单轮/总超时、结果大小、并发上限。
- 首个工具：`web_search`（Tavily 或 Brave），返回真实 URL/摘要；无凭据时默认关闭，不伪造搜索结果。
- 未启用工具时注入系统说明，禁止数字人假装已联网搜索。
- 外部网页/工具文本经 `sanitize_untrusted_text` 过滤常见注入短语；不提供 shell 类工具。
- MCP 仅占位配置（`enabled: false`），未实现远程 MCP 执行。

### 配置步骤

1. 复制 `infra/config` 到本地 `config/`（若尚未复制）。
2. 在 `config/env` 设置 `DEEPSEEK_API_KEY`，可选 `TAVILY_API_KEY` 或 `BRAVE_SEARCH_API_KEY`。
3. 在 `config/llm_models/deepseek.yaml`（或全局 `config/cyberverse.yaml` → `inference.tool_calling`）设置：

```yaml
tool_calling:
    enabled: true
    tools:
        web_search:
            enabled: true
            provider: tavily
```

4. 角色 `components.llm` 使用 `deepseek`，保持 `tts.doubao`、`asr.sensevoice`、`avatar.audiocpp_flashhead` 不变。

### 待你在 QAQ 上验证

- 带真实 API Key 的端到端语音对话 + 联网问答（本云环境无密钥、无外呼验收）。

## FlashHead / audio.cpp 720 画质

### 仓库内结论

- 默认生产仍为 **512×512** 原生生成（非后处理放大）。
- Python 层将 `height`/`width` 传入 audio.cpp；**未在本环境测量** SoulX 是否支持 720 原生 latent，亦**无 GPU** 跑 VRAM/FPS。
- `allow_resolution_test_profile: true` 才可启用 720 候选配置；禁止 1280×720 非正方形扭曲。

### QAQ（RTX 3070 8GB）操作建议

1. 使用 `infra/config/avatar_models/audiocpp_flashhead_720_candidate.yaml` 合并到本地配置。
2. 运行：`python scripts/benchmark_audiocpp_flashhead_resolution.py --config-json /path/to/params.json`
3. 同时执行 `nvidia-smi dmon` 记录 **真实 VRAM 峰值**；脚本内 `peak_rss_mb` 仅为进程 RSS 代理。
4. 若 OOM 或 FPS 不可接受，保持 512 默认；候选配置可随时删/关。

### 状态区分

| 项目 | 状态 |
|------|------|
| 工具调用代码与单测 | 已完成 |
| 720 保护逻辑与基准脚本 | 已完成 |
| QAQ 真机 720 VRAM/FPS/画质对比 | **待测** |

## 配置热加载（audiocpp_flashhead）

### 保存后下一轮生效（无需重启 inference）

在 **`SetAvatar`** 与 **`GenerateStream` 新一句开始** 时从磁盘重读 `load_config` 合并结果：

- `height` / `width` / `allow_resolution_test_profile`
- `num_inference_steps` / `seed` / `color_correction_strength`
- `buffering.*`（保留已测 `gen_fps` / `bias`）
- `metrics_log`
- DeepSeek 等 LLM：`temperature`、`model`、`tool_calling.*`（每轮 `generate_stream` 重读）

即使参考图 sha1 未变，只要上述编码相关字段变化，会重新执行 worker `op=avatar`。

### 仍须重启 inference

变更仅记录告警、不偷偷应用：

- `lib_path`、`model_path`、`exec_plan`、`backend`、`device`、`threads`
- `python`、`ld_library_path`、`CUDA_VISIBLE_DEVICES`、`world_size`
- `plugin_class`、监听端口、TURN、以及 `config/env` 密钥

`flash_head` / `live_act` 等其它头像后端当前仍按原逻辑视为需重启（未在本轮做 GPU 权重热切换）。

**操作建议：** 在 Launch 页改 720 后点保存 → 再点「启动数字人」或开始新一句；无需重启整个 inference 进程。WebRTC 轨道按实际帧尺寸编码，无固定 512 缓存。