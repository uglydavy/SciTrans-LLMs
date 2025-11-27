"""
Candidate reranking for translation quality.

This module provides:
- Multi-candidate generation and selection
- LLM-based reranking with context
- Quality scoring (fluency, adequacy, terminology)
- Consensus selection from multiple candidates

Thesis Contribution #2: Candidate reranking as part of
document-level refinement pipeline.

Based on WMT24 approaches using LLM reranking.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from scitrans_llms.translate.base import Translator, TranslationContext, TranslationResult
from scitrans_llms.translate.glossary import Glossary


@dataclass
class CandidateScore:
    """Score for a translation candidate."""
    candidate: str
    fluency: float = 0.0  # How natural the translation reads
    adequacy: float = 0.0  # How well it preserves meaning
    terminology: float = 0.0  # Glossary adherence
    placeholder_preservation: float = 0.0  # Placeholder integrity
    overall: float = 0.0  # Combined score
    
    def compute_overall(self, weights: dict[str, float] = None):
        """Compute weighted overall score."""
        weights = weights or {
            "fluency": 0.25,
            "adequacy": 0.35,
            "terminology": 0.25,
            "placeholder_preservation": 0.15,
        }
        self.overall = (
            self.fluency * weights["fluency"] +
            self.adequacy * weights["adequacy"] +
            self.terminology * weights["terminology"] +
            self.placeholder_preservation * weights["placeholder_preservation"]
        )


@dataclass
class RerankingResult:
    """Result of candidate reranking."""
    best_candidate: str
    all_candidates: list[str]
    scores: list[CandidateScore]
    metadata: dict = field(default_factory=dict)


class CandidateReranker:
    """Reranks translation candidates to select the best one.
    
    Scoring approach:
    1. Check placeholder preservation (binary)
    2. Check glossary adherence (percentage)
    3. Use LLM to score fluency and adequacy
    4. Combine scores with configurable weights
    """
    
    def __init__(
        self,
        use_llm_scoring: bool = True,
        api_key: Optional[str] = None,
    ):
        self.use_llm_scoring = use_llm_scoring
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None and self.use_llm_scoring:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                self.use_llm_scoring = False
        return self._client
    
    def rerank(
        self,
        source_text: str,
        candidates: list[str],
        context: Optional[TranslationContext] = None,
        glossary: Optional[Glossary] = None,
        source_masked: Optional[str] = None,
    ) -> RerankingResult:
        """Rerank candidates and select the best one.
        
        Args:
            source_text: Original source text
            candidates: List of translation candidates
            context: Translation context for LLM scoring
            glossary: Glossary for terminology checking
            source_masked: Masked source for placeholder checking
            
        Returns:
            RerankingResult with best candidate and scores
        """
        if not candidates:
            return RerankingResult(
                best_candidate="",
                all_candidates=[],
                scores=[],
            )
        
        if len(candidates) == 1:
            score = self._score_candidate(
                candidates[0], source_text, glossary, source_masked
            )
            return RerankingResult(
                best_candidate=candidates[0],
                all_candidates=candidates,
                scores=[score],
            )
        
        # Score all candidates
        scores = []
        for candidate in candidates:
            score = self._score_candidate(
                candidate, source_text, glossary, source_masked
            )
            scores.append(score)
        
        # LLM-based comparison if enabled
        if self.use_llm_scoring and self._get_client():
            self._add_llm_scores(source_text, candidates, scores, context)
        
        # Compute overall scores
        for score in scores:
            score.compute_overall()
        
        # Select best
        best_idx = max(range(len(scores)), key=lambda i: scores[i].overall)
        
        return RerankingResult(
            best_candidate=candidates[best_idx],
            all_candidates=candidates,
            scores=scores,
            metadata={
                "best_index": best_idx,
                "used_llm": self.use_llm_scoring,
            },
        )
    
    def _score_candidate(
        self,
        candidate: str,
        source_text: str,
        glossary: Optional[Glossary],
        source_masked: Optional[str],
    ) -> CandidateScore:
        """Score a single candidate on non-LLM metrics."""
        score = CandidateScore(candidate=candidate)
        
        # Placeholder preservation
        if source_masked:
            source_ph = set(re.findall(r'<<[A-Z]+_\d{3}>>', source_masked))
            cand_ph = set(re.findall(r'<<[A-Z]+_\d{3}>>', candidate))
            if source_ph:
                score.placeholder_preservation = len(source_ph & cand_ph) / len(source_ph)
            else:
                score.placeholder_preservation = 1.0
        else:
            score.placeholder_preservation = 1.0
        
        # Terminology adherence
        if glossary:
            from scitrans_llms.translate.glossary import check_glossary_adherence
            adherence = check_glossary_adherence(candidate, source_text, glossary)
            score.terminology = adherence["adherence_rate"]
        else:
            score.terminology = 1.0
        
        # Basic fluency heuristics (without LLM)
        score.fluency = self._basic_fluency_score(candidate)
        
        # Basic adequacy (length ratio as proxy)
        len_ratio = len(candidate) / max(len(source_text), 1)
        # French is typically ~15% longer than English
        expected_ratio = 1.15
        score.adequacy = max(0, 1 - abs(len_ratio - expected_ratio) / expected_ratio)
        
        return score
    
    def _basic_fluency_score(self, text: str) -> float:
        """Basic fluency scoring without LLM."""
        score = 1.0
        
        # Penalize repeated words
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 0.2
        
        # Penalize broken placeholders
        if re.search(r'<<[A-Z]+_\d{3}', text) and not re.search(r'<<[A-Z]+_\d{3}>>', text):
            score -= 0.3
        
        # Penalize missing punctuation at end
        if text.strip() and text.strip()[-1] not in '.!?:;,':
            score -= 0.1
        
        return max(0, score)
    
    def _add_llm_scores(
        self,
        source_text: str,
        candidates: list[str],
        scores: list[CandidateScore],
        context: Optional[TranslationContext],
    ):
        """Add LLM-based fluency and adequacy scores."""
        client = self._get_client()
        if not client:
            return
        
        # Build comparison prompt
        prompt = self._build_comparison_prompt(source_text, candidates)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for scoring
                messages=[
                    {"role": "system", "content": 
                     "You are an expert translation quality evaluator. "
                     "Score translations on fluency (1-10) and adequacy (1-10). "
                     "Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=500,
            )
            
            # Parse response
            content = response.choices[0].message.content or ""
            self._parse_llm_scores(content, scores)
            
        except Exception:
            pass  # Keep default scores if LLM fails
    
    def _build_comparison_prompt(
        self,
        source_text: str,
        candidates: list[str],
    ) -> str:
        """Build prompt for LLM scoring."""
        parts = [
            "Score these French translations of the English source.",
            "",
            f"Source: {source_text[:500]}",
            "",
        ]
        
        for i, cand in enumerate(candidates):
            parts.append(f"Translation {i+1}: {cand[:500]}")
        
        parts.extend([
            "",
            "For each translation, provide scores (1-10) for:",
            "- fluency: How natural and grammatically correct",
            "- adequacy: How well it preserves the source meaning",
            "",
            "Respond with JSON: {\"scores\": [{\"fluency\": N, \"adequacy\": N}, ...]}",
        ])
        
        return "\n".join(parts)
    
    def _parse_llm_scores(self, content: str, scores: list[CandidateScore]):
        """Parse LLM response and update scores."""
        import json
        
        try:
            # Extract JSON from response
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                return
            
            data = json.loads(match.group())
            llm_scores = data.get("scores", [])
            
            for i, llm_score in enumerate(llm_scores):
                if i < len(scores):
                    scores[i].fluency = llm_score.get("fluency", 5) / 10
                    scores[i].adequacy = llm_score.get("adequacy", 5) / 10
                    
        except (json.JSONDecodeError, KeyError):
            pass


class RerankedTranslator(Translator):
    """Translator wrapper that generates and reranks candidates.
    
    Wraps any translator and adds candidate reranking:
    1. Generate N candidates with the base translator
    2. Score each candidate
    3. Select the best one
    
    Thesis Contribution #2: Candidate reranking for better quality.
    """
    
    def __init__(
        self,
        base_translator: Translator,
        num_candidates: int = 3,
        reranker: Optional[CandidateReranker] = None,
    ):
        self.base = base_translator
        self.num_candidates = num_candidates
        self.reranker = reranker or CandidateReranker()
    
    @property
    def name(self) -> str:
        return f"reranked-{self.base.name}"
    
    @property
    def supports_candidates(self) -> bool:
        return True
    
    def translate(
        self,
        text: str,
        context: Optional[TranslationContext] = None,
        num_candidates: int = 1,
    ) -> TranslationResult:
        """Generate candidates and select the best one."""
        
        # Generate candidates
        n = max(self.num_candidates, num_candidates)
        
        if self.base.supports_candidates:
            result = self.base.translate(text, context, n)
            candidates = [result.text] + result.candidates
        else:
            # Generate multiple times
            candidates = []
            for _ in range(n):
                result = self.base.translate(text, context)
                if result.text not in candidates:
                    candidates.append(result.text)
        
        if len(candidates) == 1:
            return TranslationResult(
                text=candidates[0],
                source_text=text,
                metadata={"translator": self.name, "candidates": 1},
            )
        
        # Rerank
        glossary = context.glossary if context else None
        rerank_result = self.reranker.rerank(
            source_text=text,
            candidates=candidates,
            context=context,
            glossary=glossary,
        )
        
        return TranslationResult(
            text=rerank_result.best_candidate,
            source_text=text,
            candidates=rerank_result.all_candidates[1:],  # Exclude best
            metadata={
                "translator": self.name,
                "candidates": len(candidates),
                "best_score": rerank_result.scores[rerank_result.metadata.get("best_index", 0)].overall,
            },
        )

