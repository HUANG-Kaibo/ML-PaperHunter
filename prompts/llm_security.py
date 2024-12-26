PROMPT = """Review these numbered papers and list ONLY the numbers of papers related to LLM security.
LLM security is defined as research on the intrinsic safety or alignment issues of large language models 
(including multimodal LLMs、 LLM Agents and other LLM-related topics), covering fairness, interpretability, bias, alignment, security, scenario-specific safety, jailbreak, and other related topics.

Notice LLM security is not include use LLM to do traditional security tasks, such as malware detection, intrusion detection, etc.

If the paper is related to security or safety, and not related to LLM, you should not return it.

If no papers are relevant, return "None found."

Papers to analyze:
{papers}"""
