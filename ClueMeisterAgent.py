from typing import TypedDict
from sar_project.knowledge.knowledge_base import KnowledgeBase
import os
import dotenv
import google.generativeai as genai

dotenv.load_dotenv()
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

from sar_project.agents.base_agent import SARBaseAgent

class ClueMessage(TypedDict):
    flag_clue: int
    get_clues: bool
    get_status: bool
    ask_human_query: str

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
""",
            knowledge_base=kb
        )
        # Store dependencies and initialize
        self.kb = kb
        self.photo_agent = photo_agent
        self.drone_ir = drone_ir
        self.status = "initialized"

        # MODEL SELECTION: change this line to switch LLM models
        self.model = genai.GenerativeModel('gemini-1.5-flash')       ////ai model: SWAP

    def process_request(self, message: ClueMessage):
        """Dispatch incoming messages to appropriate handlers."""
        try:
            if "flag_clue" in message:
                clue_id = message['flag_clue']
                assert isinstance(clue_id, int)
                self.kb.add_clue_tag(clue_id, "ai_flagged")
                return {"clue_id": clue_id}

            elif message.get("get_clues", False):
                return self.clues_to_text()

            elif message.get("get_status", False):
                return {"status": self.get_status()}

            elif "ask_human_query" in message:
                query = message['ask_human_query']
                assert isinstance(query, str)
                self.kb.add_query(query)
                return {"response": "Added Query"}

            else:
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
                    clue_text = line.replace('Clue:', '').strip()
                    if clue_text and clue_text not in clues_found:
                        clues_found.append(clue_text)

    def flag_clues(self):
        """Flags related clues based on LLM reasoning."""
        clue_block = self.clues_to_text().get('clue_text', '')
        prompt = f"""
You are a Clue Meister for SAR operations.
Apply deterministic and LLM-guided rules to flag related clues. Surround their IDs with exclamation marks, e.g. !3!.

{clue_block}
"""
        response = self.model.generate_content(prompt).text
        import re
        matches = re.findall(r'!(\d+)!', response)
        for mid in matches:
            self.kb.add_clue_tag(int(mid), 'ai_flagged')
        return {"info": f"Flagged {len(matches)} clues"}

    def clues_to_text(self):
        """Returns all clues in a human-readable text block."""
        clues = self.kb.get_clues()
        lines = ['Clues:']
        flagged = set(self.kb.clue_tags.get('ai_flagged', []))
        for cid, text in clues.items():
            tag = ' (Already Flagged)' if cid in flagged else ''
            lines.append(f"Clue ID #{cid}: {text}{tag}\n")
        return {"clue_text": '\n'.join(lines)}

    def _apply_rules(self, clue) -> bool:
        """Apply deterministic SAR rules to score or filter a clue."""
        # 1. Inside search grid?
        if not self.kb.is_within_grid(clue.coordinates):
            return False

        # 2. On photo-agent's path? boost priority
        if self.photo_agent.path_intersects(clue.coordinates):
            clue.priority += 10

        # 3. Drone IR overlap? boost priority
        if self.drone_ir.overlaps(clue.coordinates):
            clue.priority += 20

        # 4. Interview support? slight boost
        if self.kb.interview_supports_path(clue):
            clue.priority += 5

        return True

    def update_status(self, status: str):
        """Updates the agent's status."""
        self.status = status
        return {"status": "updated", "new_status": status}

    def get_status(self):
        """Retrieves current agent status."""
        return getattr(self, 'status', 'unknown')
