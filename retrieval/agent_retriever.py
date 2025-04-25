"""
Executor calls this helper with an OpenSearch DSL body → returns hits.
"""
from typing import List, Dict, Any
from opensearchpy import OpenSearch
from ..config import OPENSEARCH as OS

class RetrievalAgent:
    def __init__(self):
        self.es = OpenSearch(
            hosts=[OS["hosts"]],
            http_auth=(OS["user"], OS["password"]),
            timeout=60,
        )
        self.index = OS["index"]

    def search(self, dsl: Dict[str, Any]) -> List[Dict[str, Any]]:
        resp = self.es.search(index=self.index, size=OS["top_k"], body=dsl)
        hits = [h["_source"] | {"_score": h["_score"]} for h in resp["hits"]["hits"]]
        return hits
    