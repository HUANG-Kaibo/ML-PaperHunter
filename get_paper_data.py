"""
学术会议论文数据获取工具

这个模块提供了一个简单的命令行工具，用于获取各大学术会议的论文数据。
支持批量获取多个会议和多个年份的论文。

使用示例:
    # 命令行方式
    python get_data.py -c ICLR -y 2024
    python get_data.py -c "ICLR,NeurIPS" -y "2022-2024"
    python get_data.py -c "ICLR,NeurIPS" -y "2022-2024" -e
    
    # 代码方式
    >>> from get_data import PaperFetcher
    >>> fetcher = PaperFetcher()
    >>> fetcher.fetch_papers("ICLR", [2024])
"""

import json
import requests
import openreview
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from pathlib import Path
import logging
from datetime import datetime
import time
import random
import os
import argparse
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import shutil
from prompt_filter import PromptTemplate, FilterType, PromptGenerator
import re
from conference_config import ConferenceType, ConferenceConfig

# ! 配置基础路径


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PaperFetcher:
    """论文获取器主类"""
    # Note 我的实验室的代理是127.0.0.1:8899
    def __init__(self, proxy: str = "127.0.0.1:8899", use_llm: bool = True, output_dir: Optional[str] = None):
        """
        初始化论文获取器
        
        Args:
            proxy: 代理服务器地址
            use_llm: 是否使用大模型辅助筛选
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)  
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置代理
        self.setup_proxy(proxy)
        # 初始化openreview客户端
        self.openreview_client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net'
        )
        # 初始化翻译器
        self.translator = GoogleTranslator(source='en', target='zh-CN')
        # 设置翻译延迟
        self.translation_delay = 1
        
        # 初始化大模型
        if use_llm:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 使用 ChatGLM3-6B，适合 V100 32G
            self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                "THUDM/chatglm3-6b",
                trust_remote_code=True,
                torch_dtype=torch.float16  # 使用 FP16 以节省显存
            ).to(self.device)
            self.model.eval()

    @staticmethod
    def setup_proxy(proxy: str):
        """设置HTTP/HTTPS代理"""
        if proxy:
            os.environ['http_proxy'] = f'http://{proxy}'
            os.environ['https_proxy'] = f'http://{proxy}'

    def fetch_papers(
        self,
        conferences: Union[str, List[str]],
        years: Union[int, List[int], str],
        translate: bool = True,
        num_papers: Optional[int] = None,
        template: Optional[str] = None,  # 添加模板参数
        filter_papers: Optional[int] = None  # 添加过滤论文数量参数
    ) -> Dict[str, List[Dict]]:
        """
        获取指定会议和年份的论文
        
        Args:
            conferences: 会议名称
            years: 年份
            translate: 是否生成中文翻译
            num_papers: 要翻译的论文数量
            template: 过滤模板名称，如果指定则进行过滤
            filter_papers: 要过滤的论文数量，None表示处理全部论文
        """
        # 处理会议参数
        if isinstance(conferences, str):
            conferences = [conf.strip() for conf in conferences.split(",")]
        
        # 处理年份参数
        years_set = self._parse_years(years)
        
        results = {}
        for conf in conferences:
            conf_enum = ConferenceType[conf.upper()]
            for year in years_set:
                try:
                    logger.info(f"开始获取 {conf} {year} 论文数据...")
                    papers = self._fetch_single_conference(
                        conf_enum, 
                        year, 
                        template
                    )
                    if papers:
                        key = f"{conf.lower()}{year}"
                        
                        # 如果指定了模板，进行论文过滤
                        if template:
                            logger.info(f"使用 {template} 模板过滤论文...")
                            # 如果指定了过滤数量，只处理部分论文
                            if filter_papers:
                                papers_to_filter = papers[:filter_papers]
                                logger.info(f"将只过滤前 {filter_papers} 篇论文")
                            else:
                                papers_to_filter = papers
                            
                            filtered_papers = self._filter_papers_with_llm(papers_to_filter, template)
                            
                            # NOTE 我们只关心过滤后的论文
                            # # 将处理的论文添加到结果中
                            # if filter_papers:
                            #     filtered_papers.extend(papers[filter_papers:])
                            papers = filtered_papers
                        
                        results[key] = papers
                        
                        # 保存双语数据
                        if translate:
                            logger.info(f"开始翻译 {conf} {year} 论文数据...")
                            if num_papers:
                                logger.info(f"将只翻译前 {num_papers} 篇论文")
                            self._save_bilingual_papers(papers, key, num_papers)
                        
                except Exception as e:
                    logger.error(f"获取 {conf} {year} 数据时发生错误: {e}")
                    continue
        
        return results

    def _parse_years(self, years: Union[int, List[int], str]) -> Set[int]:
        """解析年份参数"""
        if isinstance(years, int):
            return {years}
        elif isinstance(years, list):
            return set(years)
        elif isinstance(years, str):
            if "-" in years:
                start, end = map(int, years.split("-"))
                return set(range(start, end + 1))
            else:
                return {int(year.strip()) for year in years.split(",")}
        raise ValueError("不支持的年份格式")

    def _fetch_conference_papers(
        self,
        conf_type: ConferenceType,
        year: int,
        template: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        api_delay: float = 1.0  # 添加API调用延迟参数
    ) -> List[Dict]:
        """
        获取会议论文的通用方法
        
        Args:
            conf_type: 会议类型
            year: 年份
            template: 过滤模板名称
            max_retries: 最大重试次数
            retry_delay: 重试间隔(秒)
            api_delay: API调用间隔(秒)
        """
        for attempt in range(max_retries):
            try:
                # 获取会议配置
                conf_config = ConferenceConfig.get_conference_config(conf_type, year)
                if not conf_config:
                    raise ValueError(f"不支持的会议类型: {conf_type}")
                
                # 添加API调用前的延迟
                time.sleep(api_delay)
                conf_config['parser'] = self._parse_submission
                venue_id = conf_config['venue_id']   
                
                # 获取会议信息
                venue_group = self.openreview_client.get_group(venue_id)
                submission_name = venue_group.content['submission_name']['value']
                # 添加API调用间隔
                time.sleep(api_delay)
                
                # 获取所有论文
                logger.info(f"正在获取论文数据 (第{attempt + 1}次尝试)...")
                submissions = self.openreview_client.get_all_notes(
                    invitation=f'{venue_id}/-/{submission_name}',
                    details='directReplies'
                )
                
                submissions_count = len(submissions)
                logger.info(f"API 获取论文数量: {submissions_count}")
                
                if submissions_count == 0:
                    if attempt < max_retries - 1:
                        logger.warning(f"未获取到论文，{retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise Exception("多次尝试后仍未获取到论文数据")
                
                # 根据模板进行过滤
                if template:
                    paper_filter = PromptTemplate(template)
                    search_keywords = self._get_search_keywords(paper_filter.config)
                    logger.info(f"使用关键词进行初步筛选: {search_keywords}")
                    
                    # 使用正则表达式过滤论文
                    filtered_submissions = []
                    for submission in submissions:
                        if self._filter_by_keywords(submission, search_keywords):
                            filtered_submissions.append(submission)
                        # 每处理50篇论文添加一个小延迟，避免过于频繁的处理
                        if len(filtered_submissions) % 50 == 0:
                            time.sleep(0.5)
                    
                    submissions = filtered_submissions
                    logger.info(f"初步筛选找到 {len(submissions)} 篇可能相关的论文")
                
                # 解析论文数据
                papers = []
                for idx, submission in enumerate(submissions, 1):
                    try:
                        paper = conf_config['parser'](submission, idx, year)
                        papers.append(paper)
                        # 每解析10篇论文添加一个小延迟
                        if idx % 10 == 0:
                            time.sleep(0.2)
                    except Exception as e:
                        logger.error(f"解析论文 {submission.id} 时发生错误: {e}")
                        continue
                
                return papers
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # 增加重试延迟时间，采用指数退避策略
                    current_delay = retry_delay * (2 ** attempt)
                    logger.warning(f"获取论文数据失败 ({e})，{current_delay}秒后重试...")
                    time.sleep(current_delay)
                else:
                    logger.error(f"获取 {conf_type.value} {year} 数据时发生错误: {e}")
                    raise

    def _fetch_single_conference(
        self,
        conf_type: ConferenceType,
        year: int,
        template: Optional[str] = None
    ) -> List[Dict]:
        """获取单个会议的论文数据"""
        return self._fetch_conference_papers(conf_type, year, template)

    def _filter_by_keywords(self, submission: Any, search_keywords: List[str]) -> bool:
        """使用关键词过滤论文"""
        try:
            # 获取标题和摘要
            content = submission.content
            if isinstance(content.get('title'), dict):
                title = content['title'].get('value', '').lower()
                abstract = content['abstract'].get('value', '').lower()
            else:
                title = content.get('title', '').lower()
                abstract = content.get('abstract', '').lower()
            
            text = f"{title} {abstract}"
            
            # 检查是否包含任何关键词
            for keyword in search_keywords:
                if keyword.lower() in text:
                    return True
            return False
            
        except Exception as e:
            logger.error(f"关键词过滤时发生错误: {e}")
            return False

    def _get_search_keywords(self, config: Dict) -> List[str]:
        """从配置中获取搜索关键词"""
        # 确保返回的是列表
        keywords = config.get('keywords', [])
        if isinstance(keywords, str):
            keywords = [keywords]
        return keywords

    def _parse_submission(self, submission: Any, idx: int, year: int) -> Dict:
        """解析提交数据"""
        try:
            # 检查并获取内容
            content = submission.content
            # 处理不同的数据格式
            if isinstance(content.get('title'), dict):
                # ICLR 2024 格式
                title = content['title'].get('value', '')
                abstract = content['abstract'].get('value', '')
                keywords = content.get('keywords', {}).get('value', [])
            else:
                # ICML 2024 格式
                title = content.get('title', '')
                abstract = content.get('abstract', '')
                # ICML可能没有keywords字段
                keywords = content.get('keywords', []) if content.get('keywords') else []
            
            return {
                "id": idx,
                "title": title,
                "abstract": abstract,
                "keywords": keywords,
                "pdf_url": f"https://openreview.net/pdf?id={submission.id}",
                "forum_url": f"https://openreview.net/forum?id={submission.id}",
                "year": year
            }
            
        except Exception as e:
            logger.error(f"解析论文 {submission.id} 时发生错误: {str(e)}")
            # 打印出内容结构以便调试
            logger.debug(f"论文内容结构: {submission.content}")
            raise


    def _save_formatted(self, papers: List[Dict], key: str) -> None:
        """保存格式化的论文数据"""
        formatted_text = self._format_papers(papers)
        output_path = self.output_dir / f"{key}_formatted.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        logger.info(f"格式化数据已保存至: {output_path}")

    @staticmethod
    def _format_papers(papers: List[Dict]) -> str:
        """将论文数据格式化为文本"""
        formatted_papers = []
        for paper in papers:
            paper_text = (
                f"{paper['id']}. Title: {paper['title']}\n"
                f"Abstract: {paper['abstract']}\n"
                f"Decision: {paper['decision']}\n"
                f"Rating: {paper['rating']} ({paper['num_reviews']} reviews)\n"
                f"URL: {paper['forum_url']}"
            )
            formatted_papers.append(paper_text)
        return "\n\n" + "-"*80 + "\n\n".join(formatted_papers)

    def _translate_text(self, text: str, retries: int = 3) -> Tuple[str, str]:
        """
        翻译文本到中文
        
        Args:
            text: 要翻译的文本
            retries: 重试次数

        Returns:
            Tuple[str, str]: (原文, 译文)
        """
        if not text:
            return text, text
            
        for attempt in range(retries):
            try:
                time.sleep(self.translation_delay)
                # 处理过长的文本
                if len(text) > 5000:  # Google Translate API 限制
                    chunks = self._split_text(text)
                    translated_chunks = []
                    for chunk in chunks:
                        if chunk.strip():
                            translated_chunk = self.translator.translate(chunk)
                            translated_chunks.append(translated_chunk)
                            time.sleep(0.5)
                    return text, " ".join(translated_chunks)
                else:
                    if text.strip():
                        translated = self.translator.translate(text)
                        return text, translated
                    return text, text
            except Exception as e:
                if attempt == retries - 1:
                    logger.warning(f"翻译失败: {e}")
                    return text, text
                time.sleep(2 ** attempt)
                continue
        return text, text

    @staticmethod
    def _split_text(text: str, max_length: int = 4500) -> List[str]:
        """
        将长文本分割成较小的块
        
        Args:
            text: 要分割的文本
            max_length: 每个块的最大长度

        Returns:
            List[str]: 文本块列表
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        # 按句子分割
        sentences = text.split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 添加句号和空格
            sentence = sentence + '. '
            
            if current_length + len(sentence) > max_length:
                # 当前块已满，保存并开始新块
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                # 添加到当前块
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks

    def _translate_paper(self, paper: Dict) -> Dict:
        """翻译论文信息"""
        translated_paper = paper.copy()
        
        # 翻译标题
        _, translated_paper['title_cn'] = self._translate_text(paper['title'])
        
        # 翻译摘要
        _, translated_paper['abstract_cn'] = self._translate_text(paper['abstract'])
        
        # 翻译关键词
        if isinstance(paper.get('keywords', []), list):
            keywords = ', '.join(paper['keywords'])
            _, translated_keywords = self._translate_text(keywords)
            translated_paper['keywords_cn'] = translated_keywords.split('，')
        
        return translated_paper

    def _save_bilingual_papers(
        self,
        papers: List[Dict],
        key: str,
        num_papers: Optional[int] = None
    ) -> None:
        """保存双语论文数据"""
        if num_papers:
            papers_to_translate = papers[:min(num_papers, len(papers))]
            if len(papers) < num_papers:
                logger.info(f"论文总数({len(papers)})小于指定翻译数量({num_papers})，将翻译所有论文")
        else:
            papers_to_translate = papers
        
        translated_papers = []
        for paper in papers_to_translate:
            try:
                translated_paper = self._translate_paper(paper)
                translated_papers.append(translated_paper)
                logger.info(f"完成论文翻译: {paper['id']}")
            except Exception as e:
                logger.error(f"翻译论文时发生错误 {paper['id']}: {e}")
                translated_papers.append(paper)
        
        if num_papers:
            translated_papers.extend(papers[num_papers:])
        
        # 创建子文件夹
        conf_dir = self.output_dir / key
        conf_dir.mkdir(parents=True, exist_ok=True)
        output_path = conf_dir / f"{key}_bilingual.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(translated_papers, f, ensure_ascii=False, indent=2)
        logger.info(f"双语论文数据已保存至: {output_path}")
        
        # 保存格式化的双语文本
        self._save_formatted_bilingual(translated_papers, key, conf_dir)

    def _save_formatted_bilingual(self, papers: List[Dict], key: str, output_dir: Optional[Path] = None) -> None:
        """保存格式化的双语论文数据"""
        formatted_text = self._format_bilingual_papers(papers)
        output_dir = output_dir if output_dir is not None else self.output_dir
        output_path = output_dir / f"{key}_bilingual.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        logger.info(f"格式化双语数据已保存至: {output_path}")

    @staticmethod
    def _format_bilingual_papers(papers: List[Dict]) -> str:
        """将论文数据格式化为双语文本"""
        formatted_papers = []
        for paper in papers:
            paper_text = (
                f"{paper['id']}. Title / 标题:\n"
                f"EN: {paper['title']}\n"
                f"CN: {paper.get('title_cn', '翻译失败')}\n\n"
                f"Abstract / 摘要:\n"
                f"EN: {paper['abstract']}\n"
                f"CN: {paper.get('abstract_cn', '翻译失败')}\n\n"
                f"Keywords / 关键词:\n"
                f"EN: {', '.join(paper['keywords']) if isinstance(paper.get('keywords', []), list) else ''}\n"
                f"CN: {', '.join(paper.get('keywords_cn', [])) if isinstance(paper.get('keywords_cn', []), list) else ''}\n\n"
                f"Decision / 决定: {paper['decision']}\n"
                f"Rating / 评分: {paper['rating']} ({paper['num_reviews']} reviews)\n"
                f"URL: {paper['forum_url']}"
            )
            formatted_papers.append(paper_text)
        return "\n\n" + "="*80 + "\n\n".join(formatted_papers)

    def _filter_papers_with_llm(self, papers: List[Dict], template: str) -> List[Dict]:
        """使用大模型根据模板过滤论文"""
        filtered_papers = []
        total_papers = len(papers)
        
        # 初始化过滤器和提示词生成器
        paper_filter = PromptTemplate(template)
        prompt_generator = PromptGenerator(paper_filter)
        
        # 批处理参数
        batch_size = 10  # 每批处理的论文数量
        
        # 记录开始时间
        start_time = time.time()
        last_batch_time = start_time
        
        for i in range(0, total_papers, batch_size):
            batch_start_time = time.time()
            batch_papers = papers[i:i+batch_size]
            current_batch_size = len(batch_papers)
            
            logger.info(f"正在处理第 {i+1}-{min(i+batch_size, total_papers)} 篇论文 (共 {total_papers} 篇)")
            
            for paper in batch_papers:
                try:
                    paper_start_time = time.time()
                    
                    # 生成相关性判断的提示词
                    prompt = prompt_generator.generate_relevance_prompt(paper)
                    
                    # 使用模型进行判断
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=2048,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # 解析响应
                    if "是" in response[:3] or "yes" in response.lower():
                        filtered_papers.append(paper)
                        logger.info(f"论文 {paper['id']} 通过过滤")
                    else:
                        logger.debug(f"论文 {paper['id']} 未通过过滤")
                    
                    # 计算单篇论文处理时间
                    paper_time = time.time() - paper_start_time
                    logger.debug(f"处理论文 {paper['id']} 耗时: {paper_time:.2f}秒")
                    
                    # 添加短暂延迟，避免GPU过载
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"过滤论文 {paper['id']} 时发生错误: {e}")
                    continue
            
            # 每批处理完后清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 计算批次处理时间
            batch_time = time.time() - batch_start_time
            avg_paper_time = batch_time / current_batch_size
            
            # 计算预估剩余时间
            papers_remaining = total_papers - (i + current_batch_size)
            estimated_time = papers_remaining * avg_paper_time
            
            # 输出时间统计
            logger.info(
                f"批次处理时间: {batch_time:.2f}秒, "
                f"平均每篇: {avg_paper_time:.2f}秒, "
                f"预计剩余时间: {estimated_time/60:.1f}分钟"
            )
            
            # 每批次后暂停一下，避免GPU过热
            time.sleep(1)
            
            # 输出进度和通过率
            passed_count = len(filtered_papers)
            processed_count = i + current_batch_size
            pass_rate = (passed_count / processed_count) * 100
            logger.info(
                f"已过滤: {processed_count}/{total_papers} 篇, "
                f"通过: {passed_count} 篇, "
                f"通过率: {pass_rate:.1f}%"
            )
        
        # 计算总耗时
        total_time = time.time() - start_time
        avg_time = total_time / total_papers
        
        # 输出最终统计信息
        logger.info(
            f"\n过滤完成统计:\n"
            f"- 总耗时: {total_time/60:.1f}分钟\n"
            f"- 平均每篇: {avg_time:.2f}秒\n"
            f"- 处理论文: {total_papers}篇\n"
            f"- 通过论文: {len(filtered_papers)}篇\n"
            f"- 总通过率: {(len(filtered_papers)/total_papers)*100:.1f}%"
        )
        
        return filtered_papers

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="获取学术会议论文数据")
    parser.add_argument(
        "-c", "--conf",
        type=str,
        default="ICLR",
        help="会议名称，多个会议用逗号分隔，如'ICLR,NeurIPS'"
    )
    parser.add_argument(
        "-y", "--year",
        type=str,
        default="2024",
        help="年份，可以是单个年份、多个年份（逗号分隔）或年份范围（如'2022-2024'）"
    )
    parser.add_argument(
        "-p", "--proxy",
        type=str,
        default="127.0.0.1:8899",
        help="代理服务器地址，格式为'host:port'"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=Path("/data/huangkaibo/ML-PaperHunter/data"), 
        help="输出目录路径"
    )
    parser.add_argument(
        "-e", "--english-only",
        action="store_true",
        help="是否只生成英文版本（默认生成双语版本）"
    )
    parser.add_argument(
        "-n", "--num-papers",
        type=int,
        default=2,
        help="要翻译的论文数量，如果设置None，则翻译全部论文"
    )
    parser.add_argument(
        "-t", "--template",
        type=str,
        default="llm_security",
        help="过滤模板名称，默认使用llm_security模板"
    )
    parser.add_argument(
        "-f", "--filter-papers",
        type=int,
        default=None,
        help="要过滤的论文数量，默认处理全部论文"
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return args

def main():
    """主函数"""
    args = parse_args()
    print(args)
    try:
        fetcher = PaperFetcher(
            proxy=args.proxy,
            output_dir=args.output_dir
        )
        results = fetcher.fetch_papers(
            conferences=args.conf,
            years=args.year,
            translate=not args.english_only,
            num_papers=args.num_papers,
            template=args.template,
            filter_papers=args.filter_papers  # 添加过滤论文数量参数
        )
        
        logger.info("数据获取完成！")
        for key, papers in results.items():
            logger.info(f"{key}: 获取到 {len(papers)} 篇论文")
            if not args.english_only:
                if args.num_papers:
                    logger.info(f"{key}: 已翻译前 {min(args.num_papers, len(papers))} 篇论文")
                else:
                    logger.info(f"{key}: 已生成全部双语版本")
            
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

if __name__ == "__main__":
    main()