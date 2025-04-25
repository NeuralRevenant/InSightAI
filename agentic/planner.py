"""
GPT-4o-powered module that turns (question, optional feedback) → OpenSearch DSL.
"""
from __future__ import annotations
import json, re, logging, openai
from ..config import OPENAI as OA

logger = logging.getLogger(__name__)
openai.api_key = OA["api_key"]

_SYSTEM_MSG = (
    "You are an expert OpenSearch engineer.  "
    "Given a user's visual-question and optional critique of a previous answer, "
    "produce ONE *valid* OpenSearch JSON DSL body.  "
    "It may combine knn_vector and textual clauses (match, bool, filter…).  "
    "Return ONLY the JSON — no markdown ``` fences."
)

class QueryPlanner:
    """Stateless wrapper around GPT-4o chat-completion."""

    def _chat(self, user_prompt: str) -> str:
        resp = openai.chat.completions.create(
            model       = OA["model"],
            temperature = OA["temperature"],
            max_tokens  = OA["max_tokens"],
            messages = [
                {"role": "system", "content": _SYSTEM_MSG},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()

    # public -----------------------------------------------------------------
    def make_query(self, question: str, feedback: str | None = None) -> dict:
        user_prompt = (
            f"QUESTION:\n{question}\n"
            f"FEEDBACK:\n{feedback or 'None'}\n"
            "JSON DSL:"
        )
        txt = self._chat(user_prompt)

        # grab first JSON object in answer ------------------------------------
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            raise ValueError("Planner did not return JSON:\n" + txt)
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as e:
            logger.error("Bad JSON from GPT-4o planner: %s\nText:\n%s", e, txt)
            raise
