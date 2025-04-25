"""
Executor: runs the OpenSearch DSL, reformats hits into context, feeds MM model,
and decides if answer is 'good enough'.  GPU code paths are amp-enabled.
"""
import torch, json
from PIL import Image
from typing import Dict, List
from ..models.multimodal_model import InSightAIModel
from ..retrieval.agent_retriever import RetrievalAgent
from ..agentic.planner import QueryPlanner
from ..config import ANSWER_OK_TOKENS

class Executor:
    def __init__(self):
        self.retriever = RetrievalAgent()
        self.mm        = InSightAIModel().eval().cuda()
        self.planner   = QueryPlanner()

    # ── private helpers ──────────────────────────────────────────────────────
    def _hits_to_context(self, hits: List[Dict]) -> str:
        lines = []
        for h in hits:
            q = h.get("question", "")
            a = h.get("answers", "")
            lines.append(f'Q: "{q}" → A: "{a}" (score={h["_score"]:.2f})')
        return "\n".join(lines)

    def _answer_good(self, ans: str) -> bool:
        return any(ans.strip().startswith(tok) for tok in ANSWER_OK_TOKENS)

    # ── public API ───────────────────────────────────────────────────────────
    def run(self, image: Image.Image, question: str,
            feedback: str | None = None) -> Dict[str, str | List[str]]:
        # 1. Plan → DSL
        dsl = self.planner.make_query(question, feedback)
        # 2. Retrieve
        hits = self.retriever.search(dsl)
        # 3. Build enriched prompt
        ctx = self._hits_to_context(hits)
        enriched_q = f"CONTEXT:\n{ctx}\nQUESTION: {question}"
        # 4. Multimodal answer (GPU, bf16, torch.compile)
        img_t  = torch.tensor(image.resize((224,224))).permute(2,0,1).float()/255
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            answer = self.mm(img_t.unsqueeze(0).cuda(), [enriched_q])[0]
        # 5. Decide done?
        done = self._answer_good(answer)
        return {
            "dsl": json.dumps(dsl, indent=2),
            "answer": answer,
            "done": done,
            "context_used": ctx.splitlines(),
        }
