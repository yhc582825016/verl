import os, sys
sys.path.append('/mnt/code/yehangcheng/verl/recipe/rllm')
from tools.code_tools import E2BPythonInterpreter, PythonInterpreter, LCBPythonInterpreter
from tools.math_tools import CalculatorTool
from tools.web_tools import GoogleSearchTool, FirecrawlTool, TavilyTool


TOOL_REGISTRY = {
    'calculator': CalculatorTool,
    'e2b_python': E2BPythonInterpreter,
    'local_python': PythonInterpreter,
    'python': LCBPythonInterpreter, # Make LCBPythonInterpreter the default python tool for CodeExec.
    'google_search': GoogleSearchTool,
    'firecrawl': FirecrawlTool,
    'tavily': TavilyTool,
}

__all__ = [
    'CalculatorTool',
    'PythonInterpreter',
    'E2BPythonInterpreter',
    'LCBPythonInterpreter',
    'GoogleSearchTool',
    'FirecrawlTool',
    'TavilyTool',
    'TOOL_REGISTRY',
]
