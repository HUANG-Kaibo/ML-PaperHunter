# PaperHunter

🔍 机器学习顶会论文搜索工具 | 支持智能筛选、双语摘要、LLM辅助分析 | ICLR/ICML/NeurIPS

## 功能特点

- 🤖 支持多个顶级会议论文搜索
- 🎯 基于LLM的智能论文筛选
- 🌏 自动生成中英双语摘要
- ⚡ 高效批量处理
- 🔄 可自定义筛选模板
- 📊 详细的处理统计

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 基本使用

```bash
# 获取ICLR 2024的论文，使用llm_security模板筛选
python get_paper_data.py -c ICLR -y 2024 -t llm_security

# 获取多个会议的论文
python get_paper_data.py -c "ICLR,ICML" -y "2023,2024"

# 指定输出目录
python get_paper_data.py -c ICLR -y 2024 -o ./output
```

### 命令行参数

```
-c, --conf        会议名称，多个会议用逗号分隔，如'ICLR,NeurIPS'
-y, --year        年份，支持单个年份、多个年份或范围，如'2022-2024'
-p, --proxy       代理服务器地址，格式为'host:port'
-o, --output-dir  输出目录路径
-e, --english-only 只生成英文版本
-n, --num-papers  要翻译的论文数量
-t, --template    过滤模板名称
-f, --filter-papers 要过滤的论文数量
```

## 自定义配置

### 添加新的会议支持

在 `conference_config.py` 中添加新的会议配置：

1. 在 `ConferenceType` 中添加新会议：
```python
class ConferenceType(Enum):
    """支持的会议类型"""
    ICLR = "ICLR"
    ICML = "ICML"
    NEURIPS = "NeurIPS"
    # 添加新会议
    AAAI = "AAAI"  # 示例
```

2. 在 `get_conference_config` 中添加配置：
```python
base_configs = {
    # 现有配置...
    ConferenceType.ICLR: {
        'venue_id': f'ICLR.cc/{year}/Conference',
        'name': 'ICLR',
        'full_name': 'International Conference on Learning Representations',
        'submission_id': 'Submission',  # 如果不同会议的submission_id不同，可以在这里配置
        'website': 'https://iclr.cc'
    }
}
```

3. 在 `normalize_conference_name` 中添加名称映射：
```python
name_mapping = {
    # 现有映射...
    'AAAI': ConferenceType.AAAI,
    'aaai': ConferenceType.AAAI,
}
```

### 自定义筛选模板

在 `prompt_filter.py` 中创建新的筛选模板：

1. 定义模板配置：
```python
TEMPLATE_CONFIGS = {
    # 大语言模型安全相关
        "llm_security": {
            "keywords": ["language model", "LLM", "security", "safety", "privacy", 
                "adversarial", "attack", "defense", "alignment"],
            "description": "大语言模型安全相关论文",
            "criteria": [
                "是否研究大语言模型的内在安全性或对齐问题",
                "是否涉及LLM的公平性、可解释性、偏见等问题",
                "是否研究LLM的对齐、安全性或场景特定安全",
                "是否涉及LLM的越狱、对抗或防御",
                "是否包含多模态LLM或LLM Agent的安全问题",
                "注意：仅使用LLM做传统安全任务（如恶意软件检测）不属于此类"
            ],
            "aspects": [
                "安全问题的定义和分类",
                "问题的发现和分析方法",
                "解决方案和防御机制",
                "评估方法和实验结果",
                "潜在影响和局限性"
            ],
            "impact_factors": [
                "问题的重要性和普遍性",
                "解决方案的有效性",
                "方法的创新性",
                "对LLM安全领域的贡献"
            ]
    },
    # 添加新模板
    "your_template": {
            "keywords": ["关键词1", "关键词2"],
            "description": "你的领域描述",
            "criteria": [
                # 添加判断标准
            ],
            "aspects": [
                # 添加分析角度
            ],
            "impact_factors": [
                # 添加影响力评估因素
            ]
        },
}
```

2. 使用新模板：
```bash
python get_paper_data.py -c ICLR -y 2024 -t your_template
```

## 执行流程

1. 参数解析：
   - 解析命令行参数
   - 设置输出目录和代理

2. 论文获取：
   - 连接OpenReview API
   - 获取指定会议和年份的论文列表
   - 解析论文基本信息

3. 智能筛选：
   - 使用指定模板进行初步关键词筛选
   - 使用ChatGLM进行深度内容分析
   - 生成筛选报告

4. 结果处理：
   - 生成中英双语摘要
   - 保存筛选结果
   - 输出统计信息

## 注意事项

### 代理设置
- 默认代理设置为 `127.0.0.1:8899`，请根据你的实际情况修改：
```bash
# 使用自定义代理
python get_paper_data.py -c ICLR -y 2024 -p "your_proxy_host:port"

# 不使用代理
python get_paper_data.py -c ICLR -y 2024 -p ""
```

### 模型配置
- 默认使用 ChatGLM3-6B 模型，你可以在 `get_paper_data.py` 中修改：
```python
# 在 PaperFetcher 类的 __init__ 方法中
def __init__(self, proxy: str = "127.0.0.1:8899", use_llm: bool = True):
    # ...
    if use_llm:
        # 修改为你想使用的模型
        model_path = "THUDM/chatglm3-6b"  # 可以改为其他模型路径
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16  # 可以根据显存调整精度
        ).to(self.device)
```

### 硬件要求
- ChatGLM3-6B 模型推荐至少 16GB 显存
- 如果显存不足，可以考虑：
  1. 使用更小的模型
  2. 调整为 int8 或 int4 量化版本
  3. 使用 CPU 模式（将 `.to(self.device)` 改为 `.to("cpu")`）

### 性能优化
- 批处理大小可以在 `_filter_papers_with_llm` 方法中调整：
```python
batch_size = 10  # 根据显存大小调整
```
- 处理间隔可以调整：
```python
time.sleep(0.1)  # 可以根据硬件性能调整
```

### 网络相关
- 确保网络稳定，特别是在使用代理时
- OpenReview API 可能有访问限制，建议添加适当延迟
- 翻译功能需要能访问 Google 翻译服务

### 数据处理
- 处理大量论文时建议先用小批量测试
- 使用 `-n` 参数限制翻译论文数量
- 使用 `-f` 参数限制过滤论文数量
- 输出目录需要有足够的存储空间

### 模板定制
- 自定义模板时关键词要够具体和相关
- 过于宽泛的关键词可能导致筛选效果不理想
- 建议先用小规模数据测试新模板

## 实验室快速开始

如果你是实验室的同学，可以直接在实验室服务器上体验本项目：

```bash
# 登录服务器
ssh username@10.XXX.XX.164

# 进入项目目录
cd /data/huangkaibo/ML-PaperHunter

# 激活环境
conda activate kb_paperhunter

# 开始使用，例如：
python get_paper_data.py -c ICLR -y 2024 -t llm_security
```

环境已经配置好所有依赖，可以直接使用。记得使用完后及时清理自己的输出文件。

## 项目说明

本项目是在一天内快速完成的，可能存在一些疏漏和不足。如果在使用过程中发现任何问题或有改进建议，欢迎提交 Issue，感谢大家的支持和帮助！

