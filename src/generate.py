"""
generate.py
-----------
Generator module for LearnMate RAG.

Handles:
- Prompt formatting with cited passages and strict grounding instructions.
- LLM generation via Ollama (or HuggingFace / mock runtime).
- Robust JSON response extraction:
    {"answer": "<short answer or NO_ANSWER>", "cited_passage_ids": ["<id>", ...]}
- Support for token log-probability collection when supported by the runtime.

CLI:
  python -m src.generate test --config configs/b1_standard_rag.yaml --query "What is evidence grounding?"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

from src.metrics import is_no_answer as check_is_no_answer
from src.retrieve import RetrievedPassage
from src.utils import get_logger, load_config, seed_everything

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GeneratorOutput:
    answer: str
    cited_passage_ids: List[str] = field(default_factory=list)
    raw_response: str = ""
    is_no_answer: bool = False
    parse_success: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_logprobs: List[float] = field(default_factory=list)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, document-grounded question-answering assistant for university courses.\n"
    "Rules:\n"
    "1. Answer the question using ONLY the facts stated in the provided passages.\n"
    "2. Do NOT use any external knowledge, assumptions, or extrapolations.\n"
    "3. If the provided passages do not contain adequate evidence to answer, return 'NO_ANSWER'.\n"
    "4. Return your answer strictly as a valid JSON object matching this schema:\n"
    '   {"answer": "<short factual answer or NO_ANSWER>", "cited_passage_ids": ["<id1>", ...]}\n'
    "5. Cite ONLY the passage IDs that directly contain the facts used in your answer.\n"
    "Do not include any explanation or extra text outside the JSON object."
)


def format_passages_context(passages: List[Union[RetrievedPassage, dict]]) -> str:
    """Format a list of passages for inclusion in the prompt."""
    lines = []
    for p in passages:
        if isinstance(p, RetrievedPassage):
            pid = str(p.passage_id)
            txt = p.text.strip()
        elif isinstance(p, dict):
            pid = str(p.get("passage_id", p.get("id", "")))
            txt = str(p.get("text", p.get("context", ""))).strip()
        else:
            continue
        lines.append(f"[Passage ID: {pid}]\n{txt}")
    return "\n\n".join(lines)


def build_prompt(
    question: str,
    passages: List[Union[RetrievedPassage, dict]],
    system_prompt: Optional[str] = None,
) -> tuple[str, str]:
    """
    Build system prompt and user prompt.

    Returns
    -------
    (system_prompt, user_prompt)
    """
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    context_str = format_passages_context(passages)
    user_prompt = (
        f"Provided Course Passages:\n"
        f"-------------------------\n"
        f"{context_str}\n"
        f"-------------------------\n\n"
        f"Question: {question.strip()}\n\n"
        f"Answer in JSON:"
    )
    return sys_prompt, user_prompt


# ---------------------------------------------------------------------------
# Robust JSON response parser
# ---------------------------------------------------------------------------

def parse_generator_response(raw_text: str) -> tuple[str, List[str], bool]:
    """
    Extract answer and cited_passage_ids from LLM output.

    Returns
    -------
    (answer, cited_passage_ids, parse_success)
    """
    text = raw_text.strip()
    if not text:
        return "NO_ANSWER", [], False

    # 1. Clean markdown code fences if present
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    # 2. Try direct JSON parsing
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            ans = str(obj.get("answer", "")).strip()
            cites = obj.get("cited_passage_ids", [])
            if not isinstance(cites, list):
                cites = [str(cites)]
            cites = [str(c).strip() for c in cites if str(c).strip()]
            return ans, cites, True
    except Exception:
        pass

    # 3. Try to locate JSON object via regex `{ ... }`
    json_candidate = re.search(r"(\{.*\})", text, re.DOTALL)
    if json_candidate:
        try:
            obj = json.loads(json_candidate.group(1))
            if isinstance(obj, dict):
                ans = str(obj.get("answer", "")).strip()
                cites = obj.get("cited_passage_ids", [])
                if not isinstance(cites, list):
                    cites = [str(cites)]
                cites = [str(c).strip() for c in cites if str(c).strip()]
                return ans, cites, True
        except Exception:
            pass

    # 4. Fallback heuristics: check if NO_ANSWER is declared
    if check_is_no_answer(text) or bool(re.search(r"\bno[_\s]answer\b", text, re.IGNORECASE)):
        return "NO_ANSWER", [], False

    # Extract any cited passage IDs using regex
    cited_ids = re.findall(r"Passage\s*(?:ID:?|\#)?\s*([A-Za-z0-9_\-]+)", raw_text, re.IGNORECASE)
    cleaned_ans = re.sub(r"\[Passage ID:[^\]]+\]", "", raw_text).strip()
    return cleaned_ans, cited_ids, False


# ---------------------------------------------------------------------------
# LLM Generator class
# ---------------------------------------------------------------------------

class LLMGenerator:
    """
    RAG Generator supporting Ollama, Transformers, and Mock backends.
    """

    def __init__(
        self,
        runtime: str = "ollama",
        model: str = "llama3.2:3b",
        temperature: float = 0.0,
        max_tokens: int = 512,
        num_ctx: int = 4096,
        system_prompt: Optional[str] = None,
        ollama_host: str = "http://127.0.0.1:11434",
    ):
        self.runtime = runtime.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.ollama_host = ollama_host

        self._ollama_client = None
        if self.runtime == "ollama":
            try:
                import ollama
                self._ollama_client = ollama.Client(host=self.ollama_host)
            except Exception as e:
                log.warning("Ollama client could not be initialized directly: %s", e)

    @classmethod
    def from_config(cls, cfg) -> "LLMGenerator":
        runtime = str(cfg.generator.get("runtime", "ollama"))
        model = str(cfg.generator.get("model", "llama3.2:3b"))
        temperature = float(cfg.generator.get("temperature", 0.0))
        max_tokens = int(cfg.generator.get("max_tokens", 512))
        sys_prompt = str(cfg.generator.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
        ollama_host = str(cfg.generator.get("ollama_host", "http://127.0.0.1:11434"))
        num_ctx = int(cfg.generator.get("num_ctx", 4096))

        return cls(
            runtime=runtime,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            system_prompt=sys_prompt,
            ollama_host=ollama_host,
        )

    # ------------------------------------------------------------------
    def _call_ollama(self, system_prompt: str, user_prompt: str) -> dict:
        """Execute request against Ollama."""
        if self._ollama_client is None:
            import ollama
            self._ollama_client = ollama.Client(host=self.ollama_host)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self._ollama_client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.num_ctx,
            },
            format="json",
        )
        return response

    def _call_mock(self, question: str, passages: List[Union[RetrievedPassage, dict]]) -> dict:
        """Deterministic mock response generator for testing."""
        # If passages empty or no question, return NO_ANSWER
        if not passages or not question.strip():
            return {
                "message": {
                    "content": json.dumps({"answer": "NO_ANSWER", "cited_passage_ids": []})
                },
                "prompt_eval_count": 20,
                "eval_count": 10,
            }

        first_p = passages[0]
        pid = first_p.passage_id if isinstance(first_p, RetrievedPassage) else first_p.get("passage_id", "p0")
        ptxt = first_p.text if isinstance(first_p, RetrievedPassage) else first_p.get("text", "")

        # Toy heuristic for mock: return first sentence as answer
        ans_cand = ptxt.split(".")[0].strip() if ptxt else "Mock answer"
        return {
            "message": {
                "content": json.dumps({"answer": ans_cand, "cited_passage_ids": [str(pid)]})
            },
            "prompt_eval_count": 45,
            "eval_count": 15,
        }

    # ------------------------------------------------------------------
    def generate(
        self,
        question: str,
        passages: List[Union[RetrievedPassage, dict]],
    ) -> GeneratorOutput:
        """Generate grounded answer for a single question and list of passages."""
        sys_prompt, user_prompt = build_prompt(
            question=question,
            passages=passages,
            system_prompt=self.system_prompt,
        )

        t0 = time.perf_counter()
        raw_text = ""
        prompt_tokens = 0
        comp_tokens = 0

        if self.runtime == "mock":
            res = self._call_mock(question, passages)
            raw_text = res["message"]["content"]
            prompt_tokens = res.get("prompt_eval_count", 0)
            comp_tokens = res.get("eval_count", 0)
        elif self.runtime == "ollama":
            res = self._call_ollama(sys_prompt, user_prompt)
            raw_text = res["message"]["content"]
            prompt_tokens = res.get("prompt_eval_count", 0)
            comp_tokens = res.get("eval_count", 0)
        else:
            raise ValueError(f"Unsupported generator runtime: {self.runtime}")

        latency_ms = (time.perf_counter() - t0) * 1000.0

        ans, cites, success = parse_generator_response(raw_text)
        is_no_ans = check_is_no_answer(ans)

        return GeneratorOutput(
            answer=ans,
            cited_passage_ids=cites,
            raw_response=raw_text,
            is_no_answer=is_no_ans,
            parse_success=success,
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            latency_ms=latency_ms,
        )

    def generate_batch(
        self,
        questions: List[str],
        passages_batch: List[List[Union[RetrievedPassage, dict]]],
    ) -> List[GeneratorOutput]:
        """Generate answers sequentially for a batch of questions."""
        outputs = []
        for q, p in zip(questions, passages_batch):
            outputs.append(self.generate(q, p))
        return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LearnMate Grounded Generator")
    parser.add_argument("cmd", choices=["test"], help="Command")
    parser.add_argument("--config", default="configs/b1_standard_rag.yaml")
    parser.add_argument("--query", default="What is supervised learning?")
    parser.add_argument("--runtime", default=None, choices=["ollama", "mock"])

    args = parser.parse_args()
    seed_everything(42)

    cfg = load_config(args.config)
    if args.runtime:
        cfg.generator.runtime = args.runtime

    gen = LLMGenerator.from_config(cfg)

    # Provide a mock passage for CLI demonstration
    sample_passages = [
        RetrievedPassage(
            passage_id="p_sample_01",
            text="Supervised learning is a machine learning paradigm where models are trained on input-output pairs.",
            score=0.92,
            rank=1,
        )
    ]

    out = gen.generate(args.query, sample_passages)
    print("\n--- Generation Result ---")
    print(f"Answer:            {out.answer}")
    print(f"Cited Passages:    {out.cited_passage_ids}")
    print(f"Is No-Answer:      {out.is_no_answer}")
    print(f"Parse Success:     {out.parse_success}")
    print(f"Latency:           {out.latency_ms:.1f} ms")
    print(f"Raw Output:\n{out.raw_response}\n")


if __name__ == "__main__":
    main()
