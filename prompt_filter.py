"""
论文过滤模板模块

提供了一个通用的模板框架，用于定义特定领域的关键词和判断标准，
用于生成论文过滤的提示词。
"""

from typing import Dict, List
from enum import Enum

class FilterType(Enum):
    """过滤器类型"""
    TEMPLATE = "template"  # 使用预定义模板

class PromptTemplate:
    """论文过滤模板类"""
    '''
        # 用户可以按照相同的框架添加新模板
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
    '''
    # 模板框架
    FILTER_TEMPLATES = {
        # 大语言模型相关
        "llm": {
            "keywords": ["language model", "LLM", "GPT", "transformer", "NLP", "natural language"],
            "description": "大语言模型相关论文",  # 领域描述
            "criteria": [  # 判断论文是否相关的具体标准
                "是否涉及大语言模型的开发或应用",
                "是否使用或改进Transformer架构",
                "是否包含语言理解或生成任务",
                "是否涉及模型训练或微调"
            ],
            "aspects": [  # 论文分析的重要方面
                "模型架构和创新点",
                "训练方法和数据",
                "性能评估和基准",
                "应用场景和限制"
            ],
            "impact_factors": [  # 评估论文影响力的因素
                "技术创新程度",
                "实验结果质量",
                "应用价值",
                "对领域的推动作用"
            ]
        },
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
        }
    }

    def __init__(self, template: str):
        """
        初始化过滤模板
        
        Args:
            template: 使用预定义模板的名称
        """
        if template not in self.FILTER_TEMPLATES:
            raise ValueError(
                f"未知的模板名称: {template}, "
                f"请使用以下模板之一: {', '.join(self.FILTER_TEMPLATES.keys())}, "
                f"或者在FILTER_TEMPLATES中添加新的模板"
            )
            
        self.template = template
        self.config = self.FILTER_TEMPLATES[template]

    @classmethod
    def list_templates(cls) -> Dict[str, Dict]:
        """列出所有可用的预定义模板"""
        return cls.FILTER_TEMPLATES

    def get_template_info(self) -> Dict:
        """获取当前模板的完整配置"""
        return self.config

class PromptGenerator:
    """提示词生成器类"""
    
    # 通用提示词模板
    PROMPT_TEMPLATES = {
        "relevance": """
请分析以下论文是否与{domain}领域相关。

论文信息：
标题：{title}
摘要：{abstract}
关键词：{keywords}

请根据以下标准进行判断：
{criteria}

回答要求：
一定要用"是"或"否"开头
""",
        
        # 论文详细分析
        "analysis": """
请对以下{domain}领域的论文进行分析：

论文信息：
标题：{title}
摘要：{abstract}
关键词：{keywords}

请从以下方面进行分析（每点50字以内）：
{aspects}

最后，请从以下角度评估论文的潜在影响：
{impact_factors}
"""
    }

    def __init__(self, paper_filter: PromptTemplate):
        """
        初始化提示词生成器
        
        Args:
            paper_filter: PromptTemplate实例，包含模板信息
        """
        self.template = paper_filter.template
        self.config = paper_filter.config
        self.domain = self.config["description"]

    def generate_relevance_prompt(self, paper: Dict) -> str:
        """生成论文相关性判断的提示词"""
        criteria_text = "\n".join(
            f"{i+1}. {c}" 
            for i, c in enumerate(self.config.get("criteria", []))
        )
        
        return self.PROMPT_TEMPLATES["relevance"].format(
            domain=self.domain,
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            keywords=", ".join(paper.get("keywords", [])),
            criteria=criteria_text
        )

    def generate_analysis_prompt(self, paper: Dict) -> str:
        """生成论文分析的提示词"""
        aspects_text = "\n".join(
            f"{i+1}. {a}" 
            for i, a in enumerate(self.config.get("aspects", []))
        )
        
        impact_text = "\n".join(
            f"{i+1}. {f}" 
            for i, f in enumerate(self.config.get("impact_factors", []))
        )
        
        return self.PROMPT_TEMPLATES["analysis"].format(
            domain=self.domain,
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            keywords=", ".join(paper.get("keywords", [])),
            aspects=aspects_text,
            impact_factors=impact_text
        )

    def generate_custom_prompt(self, paper: Dict, custom_prompt: str) -> str:
        """生成自定义提示词"""
        return self.PROMPT_TEMPLATES["custom"].format(
            domain=self.domain,
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            keywords=", ".join(paper.get("keywords", [])),
            custom_prompt=custom_prompt
        )

