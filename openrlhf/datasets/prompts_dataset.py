from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
            
    if data.get("repo", None) is not None:
        # We're working with SWE-bench!
        full_data = {
            "repo": data["repo"],
            "instance_id": data["instance_id"],
            "base_commit": data["base_commit"],
            "patch": data["patch"],
            "test_patch": data["test_patch"],
            "problem_statement": data["problem_statement"],
            "hints_text": data["hints_text"],
            "created_at": data["created_at"],
            "version": data["version"],
            "FAIL_TO_PASS": data["FAIL_TO_PASS"],
            "PASS_TO_PASS": data["PASS_TO_PASS"],
            "environment_setup_commit": data["environment_setup_commit"],
        }
    elif data.get("input", None) is not None:
        # My dummy dataset
        full_data = {
            "input_prompt": data["input"],
            "solution": data["answer"],
        }
    else:
        full_data = None
    return prompt, data.get("test_cases", None), full_data


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, test_cases, full_data = preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append({"prompt": prompt, "test_cases": test_cases, "full_data": full_data})

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
