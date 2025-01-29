from abc import ABC, abstractmethod
from typing import *
import vllm
from vllm import SamplingParams

type Message = Dict[str, str]
type Reward = float
type AgentState = Any  # State needed to track conversation progress

class AgentInterface(ABC):
    def __init__(
        self, 
        full_data: List[dict],
        sampling_params: SamplingParams, 
        vllm_engine: vllm.LLM, 
        **kwargs
    ):
        self.num_envs = len(full_data)
        self.full_data = full_data
        self.sampling_params = sampling_params
        self.vllm_engine = vllm_engine
    
    def generate_many(self) -> List[Tuple[List[Message], Reward]]:
        # Initialize states for all conversations
        states = [self.init_state(data) for data in self.full_data]
        all_messages = [[] for _ in range(self.num_envs)]
        active_indices = list(range(self.num_envs))
        
        # Continue until all conversations are complete
        while active_indices:
            # Get next prompts for all active conversations
            active_conversations = []
            for idx in active_indices:
                prompt, states[idx] = self.get_next_prompt(all_messages[idx], states[idx])
                all_messages[idx].append(prompt)
                active_conversations.append(all_messages[idx])
            
            
            # Batch generate responses
            # TODO: Maybe use their tool API instead of handrolling?
            outputs = self.vllm_engine.chat(
                messages=active_conversations,
                sampling_params=self.sampling_params
            )
            
            # Process outputs and update states
            new_active_indices = []
            for i, output in enumerate(outputs):
                output_message = {"role": "assistant", "content": output.outputs[0].text}
                real_idx = active_indices[i]
                all_messages[real_idx].append(output_message)
                
                if not self.is_done(all_messages[real_idx], states[real_idx]):
                    new_active_indices.append(real_idx)
            
            active_indices = new_active_indices
        
        # Calculate rewards for completed conversations
        results = []
        for messages, state in zip(all_messages, states):
            reward = self.get_reward(state)
            results.append((messages, reward))
        
        return results

    @abstractmethod
    def init_state(self, data: dict) -> AgentState:
        """Initialize the state for a new RL env, given a dict of the info in one element of the dataset"""
        pass

    @abstractmethod
    def get_next_prompt(self, messages: List[Message], state: AgentState) -> Tuple[Message, AgentState]:
        """Get the next prompt to send to the model and updated state.

        In this function, you should (1) use the model's last message to update the state. 
        Then (2) create the prompt to send to the model, which should incorporate observations about the environment.
        Finally, (3) return the next prompt for the model to send, along with the updated state."""
        pass

    @abstractmethod
    def is_done(self, messages: List[Message], state: AgentState) -> bool:
        """Determine if the conversation is complete"""
        pass

    @abstractmethod
    def get_reward(self, state: AgentState) -> Reward:
        pass