"""Runs every prompting technique against an LLM and returns one uniform result object."""
from collections import Counter
from dataclasses import dataclass, field

from src import prompts as P
from src.llm import LLM


@dataclass
class Result:
    technique: str
    overall: str | None = None          # positive / neutral / negative, or None if the output was unusable
    aspects: list = field(default_factory=list)   # [{"feature", "sentiment"}]
    ok: bool = True                     # did the model follow the requested output format?
    raw: list = field(default_factory=list)       # every raw model output (for inspection / error analysis)
    prompts: list = field(default_factory=list)   # every messages list sent (shown in the Prompt Lab)
    extra: dict = field(default_factory=dict)     # agreement, summary, action, chain steps, retries ...


class PromptAnalyzer:
    def __init__(self, llm: LLM, guard: bool = True):
        self.llm = llm
        self.guard = guard

    # ------------------------------------------------------------------ single-label techniques
    def _single(self, technique: str, review: str, max_new_tokens: int, prefer: str = "first") -> Result:
        messages = P.TECHNIQUES[technique](review, guard=self.guard)
        raw = self.llm.generate(messages, max_new_tokens=max_new_tokens)[0]
        label = P.parse_label(raw, prefer=prefer)
        return Result(technique, label, [], label is not None, [raw], [messages])

    def zero_shot(self, review):
        return self._single("zero_shot", review, max_new_tokens=24)

    def role(self, review):
        return self._single("role", review, max_new_tokens=12)

    def few_shot(self, review):
        return self._single("few_shot", review, max_new_tokens=12)

    def cot(self, review):
        return self._single("cot", review, max_new_tokens=320, prefer="last")

    # ------------------------------------------------------------------ self-consistency
    def self_consistency(self, review: str, n: int = 5, temperature: float = 0.7) -> Result:
        """Sample the chain-of-thought `n` times and majority-vote the final labels.

        The share of samples agreeing with the winner is a cheap confidence score:
        low agreement flags reviews a human should look at."""
        messages = P.cot(review, guard=self.guard)
        raws = self.llm.generate(messages, max_new_tokens=320, temperature=temperature, n=n)
        votes = [lab for lab in (P.parse_label(r, prefer="last") for r in raws) if lab]
        if not votes:
            return Result("self_consistency", None, [], False, raws, [messages], {"votes": {}, "agreement": 0.0})
        counts = Counter(votes)
        winner, top = counts.most_common(1)[0]
        return Result("self_consistency", winner, [], True, raws, [messages],
                      {"votes": dict(counts), "agreement": top / len(raws), "n": len(raws)})

    # ------------------------------------------------------------------ structured JSON + repair loop
    def structured(self, review: str, retries: int = 1) -> Result:
        """JSON output with response priming. On invalid JSON, feed the error back and retry."""
        messages = P.structured(review, guard=self.guard)
        raws, sent, attempt_messages = [], [messages], messages
        for attempt in range(retries + 1):
            raw = self.llm.generate(attempt_messages, max_new_tokens=320, prefill=P.PRIME)[0]
            raws.append(raw)
            try:
                parsed = P.parse_structured(raw)
                return Result("structured", parsed["overall_sentiment"], parsed["aspects"], True, raws, sent,
                              {"summary": parsed["summary"], "retries_used": attempt})
            except P.ParseError as e:
                if attempt == retries:
                    return Result("structured", None, [], False, raws, sent,
                                  {"error": str(e), "retries_used": attempt})
                attempt_messages = P.json_repair(messages, raw, str(e))
                sent.append(attempt_messages)

    # ------------------------------------------------------------------ prompt chaining
    def chain(self, review: str) -> Result:
        """Three dependent prompts; each step's output feeds the next prompt."""
        raws, sent, steps = [], [], {}

        m1 = P.chain_extract(review, guard=self.guard)
        r1 = self.llm.generate(m1, max_new_tokens=100, prefill=P.PRIME)[0]
        features = P.parse_features(r1)
        raws.append(r1); sent.append(m1); steps["1_extract"] = features

        judged = []
        for feat in features[:8]:
            m2 = P.chain_judge(review, feat, guard=self.guard)
            r2 = self.llm.generate(m2, max_new_tokens=8)[0]
            lab = P.parse_label(r2)
            raws.append(r2); sent.append(m2)
            if lab:
                judged.append({"feature": feat, "sentiment": lab})
        steps["2_judge"] = judged

        m3 = P.chain_summarize(review, judged, guard=self.guard)
        r3 = self.llm.generate(m3, max_new_tokens=140)[0]
        raws.append(r3); sent.append(m3)
        summary = P.parse_chain_summary(r3)
        steps["3_summarize"] = summary

        overall = summary["overall"] or self._rule_based_overall(judged)
        return Result("chain", overall, judged, overall is not None, raws, sent,
                      {"steps": steps, "summary": summary["summary"], "action": summary["action"]})

    @staticmethod
    def _rule_based_overall(judged: list):
        """Fallback when step 3 is unparsable: balanced praise/complaints -> neutral, else majority."""
        if not judged:
            return None
        c = Counter(j["sentiment"] for j in judged)
        pos, neg = c["positive"], c["negative"]
        if pos and neg and abs(pos - neg) <= 1:
            return "neutral"
        return c.most_common(1)[0][0]

    # ------------------------------------------------------------------ dispatcher
    def run(self, technique: str, review: str, **kw) -> Result:
        if technique in ("zero_shot", "role", "few_shot", "cot"):
            return getattr(self, technique)(review)
        if technique == "self_consistency":
            return self.self_consistency(review, **kw)
        if technique == "structured":
            return self.structured(review, **kw)
        if technique == "chain":
            return self.chain(review)
        raise ValueError(f"unknown technique: {technique}")
