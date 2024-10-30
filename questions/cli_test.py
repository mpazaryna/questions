from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml
from langchain.prompts import PromptTemplate
from openai import OpenAI


@dataclass
class Config:
    model_name: str = "gpt-3.5-turbo"


@dataclass
class QuestionProcessingContext:
    question: str
    expert_type: str
    client: OpenAI
    config: Config


def load_prompts(prompts_file: Path) -> Dict[str, PromptTemplate]:
    """Load prompt templates from local file"""
    namespace = {}
    exec(prompts_file.read_text(), namespace)
    return namespace.get("local_prompts", {})


def get_prompt(expert_type: str, prompts: Dict[str, PromptTemplate]) -> str:
    """Get formatted prompt for expert type"""
    template = prompts.get(expert_type)
    if not template:
        raise ValueError(f"No prompt template found for expert type: {expert_type}")
    return template.format()


async def generate_related_questions(
    context: QuestionProcessingContext,
    prompts: Dict[str, PromptTemplate],
) -> List[str]:
    """Generate related questions based on user input"""
    expert_prompt = get_prompt(context.expert_type, prompts)

    messages = [
        {"role": "system", "content": expert_prompt},
        {"role": "user", "content": context.question},
    ]

    response = context.client.chat.completions.create(
        model=context.config.model_name,
        messages=messages,
    )

    return [
        q.strip()
        for q in response.choices[0].message.content.split("\n")
        if q.strip() and any(q.strip().startswith(str(i)) for i in range(1, 6))
    ]


async def main():
    # Initialize OpenAI client
    client = OpenAI()
    config = Config()

    # Load prompts
    prompts = load_prompts(Path("prompts.py"))

    # Example usage
    context = QuestionProcessingContext(
        question="I want to invest $10,000 in stocks",
        expert_type="financial advisor",
        client=client,
        config=config,
    )

    questions = await generate_related_questions(context, prompts)
    print("\nRelated questions:")
    for q in questions:
        print(q)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
