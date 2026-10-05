"""Behavioral feedback: the refinement prompt p_e that the prompt updater U evolves during training."""

import re
from collections import Counter

from . import prompts as P
from .monitor import LLM


class PromptEvolver:
    """Behavioral feedback loop: refinement prompt p_e evolved by the prompt updater U."""

    def __init__(self, dataset, updater):
        self.prompt = P.SECOND_ATTEMPT_VISUAL if dataset == "virl" else P.SECOND_ATTEMPT_SAFETY
        self.llm = LLM(updater)
        self.history = []

    def summary(self, k=16):
        h = sorted(self.history, key=lambda r: r[2])
        picked = h[:k // 2] + h[-(k // 2):] if len(h) > k else h
        return "\n\n".join(f"[Question]: {q[:400]}\n[Attempt]: {a[:800]}\n[Reward]: {r:.2f}" for q, a, r in picked)

    def quality(self, top=5):
        texts = [a for _, a, _ in self.history]
        n = max(len(texts), 1)
        openers = Counter()
        for t in texts:
            sents = re.split(r"(?<=[.!?])\s+|\n+", t)
            openers.update({" ".join(x.split()[:4]).lower() for x in sents if len(x.split()) >= 4})
        lines = [f"Attempts: {len(texts)}, mean length: {sum(len(t.split()) for t in texts) / n:.0f} words"]
        lines += [f'"{o} ..." opens a sentence in {c / n:.0%} of attempts' for o, c in openers.most_common(top) if c > 1]
        return "\n".join(lines)

    def update(self):
        out = self.llm(P.UPDATER_PROMPT.format(current_prompt=self.prompt, summary=self.summary(),
                                               quality=self.quality()),
                       temperature=0.7, max_tokens=2048)
        m = re.search(r"<prompt>(.*?)</prompt>", out, re.S)
        if m and m.group(1).strip():
            self.prompt = m.group(1).strip()
        self.history = []

    def system_for(self, question, response, reward):
        traj = P.TRAJECTORY_TEMPLATE.format(question=question, response=response, reward=reward)
        return self.prompt + P.ATTEMPT_CONTEXT.format(trajectory=traj)

    def system(self, r):
        return self.system_for(r.sample["question"], r.text, r.reward)
