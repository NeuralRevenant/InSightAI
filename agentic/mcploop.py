"""
Metacognitive control: iterate Planner↔Executor until done or exhausted.
Stores full trace for transparency.
"""
from typing import List, Dict
from PIL import Image
from ..config import MAX_MCP_ITERS
from .executor import Executor

class MCPLoop:
    def __init__(self, max_iters: int = MAX_MCP_ITERS):
        self.exec = Executor()
        self.max_iters = max_iters

    def infer(self, image: Image.Image, question: str) -> (Dict, List[Dict]):
        trace: List[Dict] = []
        feedback = None
        for i in range(self.max_iters):
            out = self.exec.run(image, question, feedback)
            trace.append(out)
            if out["done"]:
                break
            # Provide critique as feedback for next round
            feedback = f"The answer '{out['answer']}' seems incomplete."
        return trace[-1], trace
