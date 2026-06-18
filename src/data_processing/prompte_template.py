from dataclasses import dataclass
from loguru import logger

@dataclass(frozen=True)
class QATemplate:
    """Text templates used to build query inputs and define which prefix is loss-masked.
    
    A complete example consists of:
        context + question + prompt(answer prefix)
    """

    # Context (compressed) region formatting
    context_prefix: str = ""

    # Query region formatting
    question_prefix: str = "Question: "
    question_suffix: str = ""
    # answer_prefix: str = "Output only the answer span copied from the context. No full sentence, no extra words. Answer: "
    answer_prefix: str = "Answer the question directly with a short span only, no explanation. Answer:"
    answer_suffix: str = ""

    logger.info(f"QATemplate: context_prefix: {context_prefix}, question_prefix: {question_prefix}, question_suffix: {question_suffix}, answer_prefix: {answer_prefix}, answer_suffix: {answer_suffix}")

    def build_prefix_text(self, question: str) -> str:
        # Prefix tokens (all masked to -100 in labels)
        return f"{self.question_prefix}{question}{self.question_suffix}{self.answer_prefix}"

    def build_answer_text(self, answer: str) -> str:
        # Answer tokens (supervised, except padding/bos if you choose to mask)
        return f"{answer}{self.answer_suffix}"

    def build_context_text(self, context: str) -> str:
        return f"{self.context_prefix}{context}"

    # --- For evaluation ---
    def build_question_text(self, question: str) -> str:
        return f"{self.question_prefix}{question}{self.question_suffix}"
    
    @property
    def prompt_text(self) -> str:
        return self.answer_prefix