from typing import TypedDict
from sar_project.knowledge.knowledge_base import KnowledgeBase
import os
import dotenv
import google.generativeai as genai

from sar_project.agents.base_agent import SARBaseAgent
import re

# Load environment
dotenv.load_dotenv()
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY', ''))

class ClueMessage(TypedDict, total=False):
    flag_clue: int
    get_clues: bool
    get_status: bool
    ask_human_query: str
    cluster_clues: bool
    get_clusters: bool

class ClueMeisterAgent(SARBaseAgent):
    def __init__(self, kb: KnowledgeBase, photo_agent, drone_ir, name: str = "clue_meister"):
        super().__init__(
            name=name,
            role="Clue Meister",
            system_message="""
You are a Clue Meister for SAR operations. Your role is to:
1. Sort clues by criteria
2. Identify patterns in clue sets
3. Initiate further inquiries
4. Group related clues into clusters
""",
            knowledge_base=kb
        )
        # Dependencies
        self.kb = kb
        self.photo_agent = photo_agent
        self.drone_ir = drone_ir
        self.status = "initialized"
        self.clusters: dict[int, list[int]] = {}

        # MODEL SELECTION: change to desired LLM
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def process_request(self, message: ClueMessage):
        """Dispatch incoming messages to handlers."""
        try:
            if message.get("flag_clue") is not None:
                cid = message['flag_clue']
                self.kb.add_clue_tag(cid, "ai_flagged")
                return {"clue_id": cid}

            if message.get("get_clues"):
                return self.clues_to_text()

            if message.get("get_status"):
                return {"status": self.get_status()}

            if message.get("ask_human_query"):
                q = message['ask_human_query']
                self.kb.add_query(q)
                return {"response": "Added Query"}

            if message.get("cluster_clues"):
                return self.cluster_clues()

            if message.get("get_clusters"):
                return self.get_clusters()

            return {"error": "Unknown request type"}
        except Exception as e:
            return {"error": str(e)}

    def extract_clue(self, raw_text: str):
        """Extracts clues from raw text via the LLM."""
        clues_found = []
        while True:
            prompt = f"""
You are a Clue Meister for SAR operations. Extract a clue from the text below.

Please list each clue on its own line prefaced by 'Clue:'. If no new clues, reply 'No New Clues'.

Raw Text:
{raw_text}

Clues Already Found:
{clues_found}
"""
            import time; time.sleep(1)
            response = self.model.generate_content(prompt).text

            if 'Clue:' not in response:
                for clue in clues_found:
                    self.kb.add_clue(clue)
                return {"clues": clues_found}

            for line in response.splitlines():
                if line.startswith('Clue:'):
                    txt = line.replace('Clue:', '').strip()
                    if txt and txt not in clues_found:
                        clues_found.append(txt)

    def flag_clues(self):
        """Flags related clues based on LLM reasoning."""
        clue_block = self.clues_to_text()['clue_text']
        prompt = f"""
You are a Clue Meister for SAR operations.
Apply rules to flag related clues. Surround their IDs with exclamation marks, e.g. !3!.

{clue_block}
"""
        response = self.model.generate_content(prompt).text
        matches = re.findall(r'!(\d+)!', response)
        for mid in matches:
            self.kb.add_clue_tag(int(mid), 'ai_flagged')
        return {"info": f"Flagged {len(matches)} clues"}

    def clues_to_text(self):
        """Returns all clues in a human-readable block."""
        clues = self.kb.get_clues()
        flagged = set(self.kb.clue_tags.get('ai_flagged', []))
        text = "Clues:\n"
        for cid, txt in clues.items():
            tag = ' (Already Flagged)' if cid in flagged else ''
            text += f"Clue ID #{cid}: {txt}{tag}\n"
        return {"clue_text": text}

    def cluster_clues(self):
        """Groups related clues into clusters via LLM reasoning."""
        clues = self.kb.get_clues()
        block = '\n'.join(f"{cid}: {txt}" for cid, txt in clues.items())
        prompt = f"""
You are a Clue Meister. Group the following clues into clusters by similarity. List as:
Cluster 1: [1,2,5]
Cluster 2: [3,4]

Clues:
{block}
"""
        response = self.model.generate_content(prompt).text
        clusters: dict[int, list[int]] = {}
        for line in response.splitlines():
            m = re.match(r"Cluster (\d+): \[(.*?)\]", line)
            if m:
                cid = int(m.group(1))
                ids = [int(i) for i in m.group(2).split(',') if i.strip().isdigit()]
                clusters[cid] = ids
        self.clusters = clusters
        return {"clusters": clusters}

    def get_clusters(self):
        """Retrieves the last computed clusters."""
        return {"clusters": self.clusters}

    def _apply_rules(self, clue) -> bool:
        """Deterministic SAR rule application."""
        if not self.kb.is_within_grid(clue.coordinates):
            return False
        if self.photo_agent.path_intersects(clue.coordinates):
            clue.priority += 10
        if self.drone_ir.overlaps(clue.coordinates):
            clue.priority += 20
        if self.kb.interview_supports_path(clue):
            clue.priority += 5
        return True

    def update_status(self, status: str):
        """Updates the agent's status."""
        self.status = status
        return {"status": "updated", "new_status": status}

    def get_status(self):
        """Retrieves current status."""
        return getattr(self, 'status', 'unknown')
