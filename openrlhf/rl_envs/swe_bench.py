from openrlhf.utils.interface import AgentInterface
from typing import *

Message = Dict[str, str]
AgentState = Any

class SweBenchEnv(AgentInterface):
    async def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def init_state(self, data: dict) -> AgentState:
        pass
    
    async def get_next_prompt(self, messages: List[Message], state: AgentState) -> Tuple[Message, AgentState]:
        pass
    
    async def is_done(self, messages: List[Message], state: AgentState) -> bool:
        pass
    
    async def get_reward(self, messages: List[Message], state: AgentState) -> float:
        pass