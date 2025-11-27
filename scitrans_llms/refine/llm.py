"""
LLM-based refinement for translation improvement.

This module provides:
- Document-level coherence refinement
- Style and fluency improvement
- Pronoun and reference resolution
- Consistency checking across segments

Thesis Contribution #2: Document-level LLM refinement
to improve coherence beyond segment-level translation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from scitrans_llms.refine.base import Refiner, RefinementResult
from scitrans_llms.translate.glossary import Glossary
from scitrans_llms.translate.llm import LLMConfig


@dataclass
class RefinementConfig:
    """Configuration for LLM refinement."""
    model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 4096
    focus_areas: list[str] = None  # ['coherence', 'terminology', 'style']
    
    def __post_init__(self):
        if self.focus_areas is None:
            self.focus_areas = ["coherence", "terminology", "fluency"]


class LLMRefiner(Refiner):
    """LLM-based refiner for translation improvement.
    
    Uses an LLM to:
    1. Check coherence across paragraphs
    2. Fix pronoun references
    3. Ensure terminology consistency
    4. Improve fluency while preserving meaning
    
    Thesis Contribution #2: This is the refinement pass that
    distinguishes document-level from segment-level translation.
    """
    
    def __init__(
        self,
        config: Optional[RefinementConfig] = None,
        api_key: Optional[str] = None,
    ):
        self.config = config or RefinementConfig()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
    
    @property
    def name(self) -> str:
        return "llm-refiner"
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI library required. Install with: pip install openai"
                )
            
            if not self.api_key:
                raise ValueError("OpenAI API key required for LLM refinement")
            
            self._client = OpenAI(api_key=self.api_key)
        
        return self._client
    
    def _build_refinement_prompt(
        self,
        translated_text: str,
        source_text: Optional[str],
        context: Optional[str],
        glossary: Optional[Glossary],
    ) -> tuple[str, str]:
        """Build system and user prompts for refinement."""
        
        system_parts = [
            "You are an expert editor specializing in scientific translation quality.",
            "Your task is to refine a French translation of an English scientific text.",
            "",
            "## Your Goals:",
        ]
        
        if "coherence" in self.config.focus_areas:
            system_parts.append("- Ensure coherence: pronouns should have clear references")
        if "terminology" in self.config.focus_areas:
            system_parts.append("- Ensure terminology consistency throughout")
        if "fluency" in self.config.focus_areas:
            system_parts.append("- Improve fluency while preserving technical accuracy")
        
        system_parts.extend([
            "",
            "## Critical Rules:",
            "1. PRESERVE all placeholders exactly (e.g., <<MATH_001>>, <<URL_002>>)",
            "2. Do NOT change the meaning of the translation",
            "3. Keep the same paragraph structure",
            "4. Only output the refined translation, no explanations",
        ])
        
        if glossary and len(glossary) > 0:
            system_parts.append("")
            system_parts.append("## Required Terminology:")
            for entry in list(glossary)[:20]:
                system_parts.append(f"  • {entry.source} → {entry.target}")
        
        system_prompt = "\n".join(system_parts)
        
        # Build user prompt
        user_parts = []
        
        if source_text:
            user_parts.append("## Original English:")
            user_parts.append(source_text)
            user_parts.append("")
        
        if context:
            user_parts.append("## Preceding Context (for reference):")
            user_parts.append(context[:500] + "..." if len(context) > 500 else context)
            user_parts.append("")
        
        user_parts.append("## Current Translation to Refine:")
        user_parts.append(translated_text)
        user_parts.append("")
        user_parts.append("Please provide the refined translation:")
        
        user_prompt = "\n".join(user_parts)
        
        return system_prompt, user_prompt
    
    def refine(
        self,
        translated_text: str,
        source_text: Optional[str] = None,
        context: Optional[str] = None,
        glossary: Optional[Glossary] = None,
    ) -> RefinementResult:
        """Refine a translation using LLM."""
        
        # Skip very short texts
        if len(translated_text.strip()) < 20:
            return RefinementResult(
                original_text=translated_text,
                refined_text=translated_text,
                changes_made=[],
                metadata={"refiner": self.name, "skipped": "text too short"},
            )
        
        client = self._get_client()
        
        system_prompt, user_prompt = self._build_refinement_prompt(
            translated_text, source_text, context, glossary
        )
        
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            refined = response.choices[0].message.content or translated_text
            refined = refined.strip()
            
            # Validate that placeholders are preserved
            import re
            source_placeholders = set(re.findall(r'<<[A-Z]+_\d{3}>>', translated_text))
            refined_placeholders = set(re.findall(r'<<[A-Z]+_\d{3}>>', refined))
            
            if source_placeholders != refined_placeholders:
                # Refinement damaged placeholders - reject it
                return RefinementResult(
                    original_text=translated_text,
                    refined_text=translated_text,
                    changes_made=[],
                    metadata={
                        "refiner": self.name,
                        "rejected": "placeholder mismatch",
                        "missing": list(source_placeholders - refined_placeholders),
                    },
                )
            
            # Determine what changed
            changes = []
            if refined != translated_text:
                # Simple change detection
                if len(refined) != len(translated_text):
                    changes.append(f"length: {len(translated_text)} → {len(refined)}")
                changes.append("text refined")
            
            return RefinementResult(
                original_text=translated_text,
                refined_text=refined,
                changes_made=changes,
                metadata={
                    "refiner": self.name,
                    "model": self.config.model,
                },
            )
            
        except Exception as e:
            return RefinementResult(
                original_text=translated_text,
                refined_text=translated_text,
                changes_made=[],
                metadata={
                    "refiner": self.name,
                    "error": str(e),
                },
            )


class CoherenceRefiner(LLMRefiner):
    """Specialized refiner focusing on document coherence.
    
    Specifically targets:
    - Pronoun resolution (il/elle/ils/elles)
    - Demonstrative consistency (ce/cet/cette/ces)
    - Temporal markers (maintenant/ensuite/puis)
    - Logical connectors (donc/ainsi/par conséquent)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        config = RefinementConfig(
            focus_areas=["coherence"],
            temperature=0.1,  # Lower temperature for consistency
        )
        super().__init__(config, api_key)
    
    @property
    def name(self) -> str:
        return "coherence-refiner"


class StyleRefiner(LLMRefiner):
    """Specialized refiner for scientific writing style.
    
    Focuses on:
    - Formal register appropriate for scientific texts
    - Passive voice where appropriate
    - Precise terminology
    - Avoiding colloquialisms
    """
    
    def __init__(self, api_key: Optional[str] = None):
        config = RefinementConfig(
            focus_areas=["style", "fluency"],
            temperature=0.3,
        )
        super().__init__(config, api_key)
    
    @property
    def name(self) -> str:
        return "style-refiner"

