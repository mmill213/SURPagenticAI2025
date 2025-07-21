from typing import TypedDict, List, Tuple
from sar_project.knowledge.knowledge_base import KnowledgeBase #BLACKBOARD/REDIS
from sar_project.agents.base_agent import SARBaseAgent
import math

# ClueMessage Types
class ClueMessage(TypedDict, total=False):
    get_clues: bool
    get_status: bool
    update_path: List[Tuple[float, float]]
    rank_clues: bool

# Clue type scoring (DISCUSS POINT AMOUNTS WITH YAYUN AND OTHERS)     
# Other clues could be: Personal Item (determined based on last seen and interview agent), Non-personal Item (worth much less), Campsite

CLUE_TYPE_POINTS = {
    "IR signature": 30,
    "red hat": 20,
    "footprint": 10,
    "unknown": 5
}

class ClueMeisterAgent(SARBaseAgent):
    def __init__(self, kb: KnowledgeBase, name: str = "clue_meister"):
        super().__init__(
            name=name,
            role="Clue Meister",
            system_message="Deterministic SAR agent that scores and ranks clues based on type and proximity.",
            knowledge_base=kb
        )
        self.kb = kb
        self.path: List[Tuple[float, float]] = []
        self.status = "initialized"

    def process_request(self, message: ClueMessage):
        try:
            if message.get("get_clues"):
                return self.clues_to_text()

            if message.get("get_status"):
                return {"status": self.status}

            if "update_path" in message:
                self.path = message["update_path"]
                return {"status": "path updated", "path_length": len(self.path)}

            if message.get("rank_clues"):
                return self.rank_clues()

            return {"error": "Unknown request type"}
        except Exception as e:
            return {"error": str(e)}

    def clues_to_text(self):
        clues = self.kb.get_clues()
        return {"clues": clues}

    def rank_clues(self):
        """Scores and ranks clues based on type and proximity to the path."""
        if not self.path:
            return {"error": "Path is not set"}

        clues = self.kb.get_clues()  # Assume structure: {id: "IR signature at X,Y"}
        ranked = []

        for cid, desc in clues.items():
            clue_type, coord = self._parse_clue(desc)
            score = CLUE_TYPE_POINTS.get(clue_type, CLUE_TYPE_POINTS["unknown"])
            distance = self._min_distance_to_path(coord)
            proximity_score = max(0, 100 - distance)  # closer = more points (DISCUSS TO DETERMINE POSSIBLE CHANGES SUCH AS MAX DISTANCE BASED ON SITUATION)
            total_score = score + proximity_score
            ranked.append((cid, desc, total_score))

        ranked.sort(key=lambda x: x[2], reverse=True)
        return {
            "ranked_clues": [
                {"id": cid, "description": desc, "score": score}
                for cid, desc, score in ranked
            ]
        }

    def _parse_clue(self, desc: str) -> Tuple[str, Tuple[float, float]]:
        """Parses clue description like 'red hat at 45.2,-121.3'."""
        parts = desc.split(" at ")
        clue_type = parts[0].strip() if len(parts) > 1 else "unknown"
        try:
            lat_str, lon_str = parts[1].split(",")
            coord = (float(lat_str.strip()), float(lon_str.strip()))
        except Exception:
            coord = (0.0, 0.0)
        return clue_type, coord

    def _min_distance_to_path(self, coord: Tuple[float, float]) -> float:
        """Finds the minimum Euclidean distance between a point and the path."""
        return min(self._euclidean_distance(coord, p) for p in self.path)

    @staticmethod
    def _euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
