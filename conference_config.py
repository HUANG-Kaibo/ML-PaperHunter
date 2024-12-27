"""
会议配置管理模块

这个模块提供了会议相关的配置信息，包括：
- 支持的会议类型
- 会议的具体配置（venue_id等）
- 会议名称的标准化
"""

from enum import Enum
from typing import Dict, Any, Optional


class ConferenceType(Enum):
    """支持的会议类型"""
    ICLR = "ICLR"
    ICML = "ICML"
    NEURIPS = "NeurIPS"
    # 在这里添加新的会议类型
    # AAAI = "AAAI"
    # ACL = "ACL"
    # ...



class ConferenceConfig:
    """会议配置管理类"""
    
    @staticmethod
    def get_conference_config(conf_type: ConferenceType, year: int) -> Dict[str, Any]:
        """
        获取指定会议的配置信息
        
        Args:
            conf_type: 会议类型
            year: 年份
            
        Returns:
            Dict: 会议配置信息
        """
        # 基础配置
        base_configs = {
            ConferenceType.ICLR: {
                'venue_id': f'ICLR.cc/{year}/Conference',
                'name': 'ICLR',
                'full_name': 'International Conference on Learning Representations',
                'submission_id': 'Submission',  # 如果不同会议的submission_id不同，可以在这里配置
                'website': 'https://iclr.cc'
            },
            ConferenceType.ICML: {
                'venue_id': f'ICML.cc/{year}/Conference',
                'name': 'ICML',
                'full_name': 'International Conference on Machine Learning',
                'submission_id': 'Submission',
                'website': 'https://icml.cc'
            },
            ConferenceType.NEURIPS: {
                'venue_id': f'NeurIPS.cc/{year}/Conference',
                'name': 'NeurIPS',
                'full_name': 'Neural Information Processing Systems',
                'submission_id': 'Submission',
                'website': 'https://neurips.cc'
            },
            # 在这里添加新的会议配置
        }
        
        return base_configs.get(conf_type, {})
    
    @staticmethod
    def normalize_conference_name(name: str) -> Optional[ConferenceType]:
        """
        标准化会议名称
        
        Args:
            name: 会议名称（支持多种格式）
            
        Returns:
            ConferenceType: 标准化的会议类型
        """
        # 名称映射表
        name_mapping = {
            # ICLR
            'ICLR': ConferenceType.ICLR,
            'iclr': ConferenceType.ICLR,
            'International Conference on Learning Representations': ConferenceType.ICLR,
            
            # ICML
            'ICML': ConferenceType.ICML,
            'icml': ConferenceType.ICML,
            'International Conference on Machine Learning': ConferenceType.ICML,
            
            # NeurIPS
            'NEURIPS': ConferenceType.NEURIPS,
            'NeurIPS': ConferenceType.NEURIPS,
            'neurips': ConferenceType.NEURIPS,
            'NIPS': ConferenceType.NEURIPS,
            'Neural Information Processing Systems': ConferenceType.NEURIPS,
            
            # 添加新的会议名称映射
            # 'AAAI': ConferenceType.AAAI,
            # 'aaai': ConferenceType.AAAI,
            # ...
        }
        
        return name_mapping.get(name)
    
    @staticmethod
    def is_valid_conference(name: str) -> bool:
        """
        检查会议名称是否有效
        
        Args:
            name: 会议名称
            
        Returns:
            bool: 是否是支持的会议
        """
        return ConferenceConfig.normalize_conference_name(name) is not None


# 使用示例
if __name__ == "__main__":
    # 获取会议配置
    conf_type = ConferenceType.ICLR
    config = ConferenceConfig.get_conference_config(conf_type, 2024)
    print(f"ICLR 2024 配置: {config}")
    
    # 标准化会议名称
    name = "neurips"
    normalized = ConferenceConfig.normalize_conference_name(name)
    print(f"标准化 '{name}' -> {normalized}")
    
    # 检查会议名称是否有效
    print(f"'ICLR' 是否有效: {ConferenceConfig.is_valid_conference('ICLR')}")
    print(f"'xyz' 是否有效: {ConferenceConfig.is_valid_conference('xyz')}") 