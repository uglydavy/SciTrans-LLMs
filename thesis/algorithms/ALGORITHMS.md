# Core Algorithms for SciTrans-LLMs

## Algorithm 1: LaTeX-Aware Masking

```
Algorithm: LaTeX-Aware Masking
Input: text (string containing LaTeX)
Output: masked_text, mask_map

1. Initialize mask_map ← empty dictionary
2. Initialize patterns ← [latex_inline, latex_display, citations, references]
3. FOR each pattern in patterns DO
4.     matches ← FindAll(pattern, text)
5.     FOR each match in matches DO
6.         token ← GenerateUniqueToken(match)
7.         mask_map[token] ← match.content
8.         text ← Replace(text, match, token)
9.     END FOR
10. END FOR
11. RETURN text, mask_map
```

### Python Implementation:
```python
import re
import hashlib

def mask_latex(text):
    mask_map = {}
    patterns = [
        (r'\$\$[^$]+\$\$', 'DISPLAY'),
        (r'\$[^$]+\$', 'INLINE'),
        (r'\\cite\{[^}]+\}', 'CITE'),
        (r'\\ref\{[^}]+\}', 'REF'),
    ]
    
    for pattern, prefix in patterns:
        for match in re.finditer(pattern, text):
            content = match.group(0)
            hash_id = hashlib.md5(content.encode()).hexdigest()[:8]
            token = f'__{prefix}_{hash_id}__'
            mask_map[token] = content
            text = text.replace(content, token, 1)
    
    return text, mask_map
```

## Algorithm 2: Context-Aware Translation

```
Algorithm: Context-Aware Translation
Input: document_blocks, context_window
Output: translated_blocks

1. translated ← []
2. FOR i ← 0 to len(document_blocks) DO
3.     context_start ← max(0, i - context_window)
4.     context_end ← min(len(blocks), i + context_window + 1)
5.     context ← blocks[context_start:context_end]
6.     
7.     IF blocks[i].type == EQUATION OR CODE THEN
8.         translated[i] ← blocks[i].text  // Don't translate
9.     ELSE
10.        prompt ← BuildContextPrompt(blocks[i], context)
11.        translation ← TranslateWithContext(prompt)
12.        translated[i] ← translation
13.    END IF
14. END FOR
15. RETURN translated
```

## Algorithm 3: Glossary-Enhanced Translation

```
Algorithm: Glossary-Enhanced Translation
Input: text, glossary, base_translator
Output: enhanced_translation

1. // Phase 1: Identify glossary terms
2. term_positions ← []
3. FOR each term in glossary DO
4.     positions ← FindAll(term.source, text)
5.     term_positions.append((term, positions))
6. END FOR

7. // Phase 2: Protect terms during translation
8. protected_text ← text
9. protection_map ← {}
10. FOR (term, positions) in term_positions DO
11.     placeholder ← GeneratePlaceholder(term)
12.     protection_map[placeholder] ← term.target
13.     protected_text ← Replace(protected_text, term.source, placeholder)
14. END FOR

15. // Phase 3: Translate
16. translation ← base_translator.translate(protected_text)

17. // Phase 4: Restore glossary terms
18. FOR (placeholder, target) in protection_map DO
19.     translation ← Replace(translation, placeholder, target)
20. END FOR

21. RETURN translation
```

## Algorithm 4: Multi-Backend Translation with Voting

```
Algorithm: Multi-Backend Translation Ensemble
Input: text, backends[], voting_strategy
Output: best_translation

1. candidates ← []
2. scores ← []

3. // Get translations from all backends
4. FOR backend in backends DO
5.     translation ← backend.translate(text)
6.     confidence ← backend.get_confidence()
7.     candidates.append(translation)
8.     scores.append(confidence)
9. END FOR

10. // Apply voting strategy
11. IF voting_strategy == "WEIGHTED" THEN
12.     best_translation ← WeightedConsensus(candidates, scores)
13. ELSE IF voting_strategy == "MAJORITY" THEN
14.     best_translation ← MajorityVote(candidates)
15. ELSE
16.     best_translation ← candidates[argmax(scores)]
17. END IF

18. RETURN best_translation
```

## Algorithm 5: Prompt Optimization for LLM Translation

```
Algorithm: Adaptive Prompt Optimization
Input: source_text, target_lang, examples[], model
Output: optimized_prompt

1. base_prompt ← "Translate from English to {target_lang}:"
2. best_score ← 0
3. best_prompt ← base_prompt

4. // Test different prompt strategies
5. strategies ← [
6.     "direct": "{base_prompt}\n{text}",
7.     "role": "You are an expert translator. {base_prompt}\n{text}",
8.     "few_shot": "{base_prompt}\n{examples}\n{text}",
9.     "cot": "{base_prompt} Think step by step.\n{text}",
10.    "technical": "Translate this scientific text {base_prompt}\n{text}"
11. ]

12. FOR strategy in strategies DO
13.     prompt ← FormatPrompt(strategy, base_prompt, text, examples)
14.     translation ← model.generate(prompt)
15.     score ← EvaluateQuality(translation)
16.     
17.     IF score > best_score THEN
18.         best_score ← score
19.         best_prompt ← prompt
20.     END IF
21. END FOR

22. RETURN best_prompt
```

## Algorithm 6: PDF Block Classification

```
Algorithm: Scientific Document Block Classification  
Input: pdf_blocks
Output: classified_blocks

1. FOR each block in pdf_blocks DO
2.     features ← ExtractFeatures(block)
3.     
4.     // Rule-based classification
5.     IF block.text matches r"^\d+\.?\s+[A-Z]" THEN
6.         block.type ← HEADING
7.     ELSE IF block.text contains "\\begin{equation}" THEN
8.         block.type ← EQUATION
9.     ELSE IF len(block.text) < 50 AND block.text.isupper() THEN
10.        block.type ← TITLE
11.    ELSE IF block.text starts with "Figure" OR "Table" THEN
12.        block.type ← CAPTION
13.    ELSE IF CountLatexCommands(block.text) > 5 THEN
14.        block.type ← FORMULA
15.    ELSE
16.        block.type ← PARAGRAPH
17.    END IF
18. END FOR

19. RETURN classified_blocks
```

## Algorithm 7: Reranking Translation Candidates

```
Algorithm: Neural Reranking for Translation
Input: source, candidates[], reranker_model
Output: best_candidate

1. scores ← []
2. features ← []

3. FOR candidate in candidates DO
4.     // Extract features
5.     f1 ← LengthRatio(source, candidate)
6.     f2 ← LexicalOverlap(source, candidate)  
7.     f3 ← SemanticSimilarity(source, candidate)
8.     f4 ← LanguageModelScore(candidate)
9.     f5 ← GrammarScore(candidate)
10.    
11.    feature_vector ← [f1, f2, f3, f4, f5]
12.    features.append(feature_vector)
13. END FOR

14. // Neural scoring
15. scores ← reranker_model.predict(features)

16. // Select best
17. best_idx ← argmax(scores)
18. RETURN candidates[best_idx]
```

## Algorithm 8: Incremental Glossary Learning

```
Algorithm: Incremental Glossary Learning
Input: document_pairs[], existing_glossary
Output: updated_glossary

1. term_candidates ← {}
2. 
3. FOR (source_doc, target_doc) in document_pairs DO
4.     // Extract technical terms
5.     source_terms ← ExtractTechnicalTerms(source_doc)
6.     target_terms ← ExtractTechnicalTerms(target_doc)
7.     
8.     // Align terms using statistical methods
9.     alignments ← AlignTerms(source_terms, target_terms)
10.    
11.    FOR (s_term, t_term, confidence) in alignments DO
12.        IF s_term not in existing_glossary THEN
13.            IF s_term in term_candidates THEN
14.                term_candidates[s_term].add((t_term, confidence))
15.            ELSE
16.                term_candidates[s_term] ← {(t_term, confidence)}
17.            END IF
18.        END IF
19.    END FOR
20. END FOR

21. // Select best translations
22. FOR s_term in term_candidates DO
23.     translations ← term_candidates[s_term]
24.     best_translation ← SelectBestByFrequencyAndConfidence(translations)
25.     existing_glossary.add(s_term, best_translation)
26. END FOR

27. RETURN existing_glossary
```

## Complexity Analysis

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| LaTeX Masking | O(n·p) | O(n) |
| Context-Aware Translation | O(n·w) | O(n) |
| Glossary Enhancement | O(n·g) | O(g) |
| Multi-Backend Voting | O(n·b) | O(b) |
| Prompt Optimization | O(s·m) | O(s) |
| Block Classification | O(n) | O(1) |
| Neural Reranking | O(c·f) | O(c·f) |
| Incremental Glossary | O(d·t²) | O(t) |

Where:
- n = text length
- p = number of patterns
- w = context window size  
- g = glossary size
- b = number of backends
- s = number of strategies
- m = model inference time
- c = number of candidates
- f = feature dimension
- d = number of documents
- t = average terms per document
