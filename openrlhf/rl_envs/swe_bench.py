from openrlhf.utils.interface import AgentInterface
from typing import *

type Message = Dict[str, str]
type AgentState = Any

class SweBenchEnv(AgentInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def init_state(self, data: dict) -> AgentState:
        pass
    
    def get_next_prompt(self, messages: List[Message], state: AgentState) -> Tuple[Message, AgentState]:
        pass
    
    def is_done(self, messages: List[Message], state: AgentState) -> bool:
        pass
    
    def get_reward(self, state: AgentState) -> float:
        pass