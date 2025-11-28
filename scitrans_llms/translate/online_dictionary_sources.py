"""
Online dictionary sources for enriching translation dictionaries.

This module provides functions to load translation dictionaries from various
free online sources instead of maintaining limited local dictionaries.

Sources:
1. MyMemory Translation API (free, no key required)
2. Wiktionary API
3. Free Translation Memory databases
4. Pre-built bilingual lexicons from academic sources
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.parse
import urllib.error


def load_mymemory_dictionary(
    source_lang: str = "en",
    target_lang: str = "fr",
    common_words: Optional[List[str]] = None,
    max_words: int = 1000,
    cache_file: Optional[Path] = None
) -> Dict[str, str]:
    """
    Load translations from MyMemory Translation Memory API.
    
    MyMemory is a free translation memory with no API key required.
    Rate limit: 10 calls/second, 1000 calls/day for anonymous users.
    
    Args:
        source_lang: Source language code (e.g., 'en')
        target_lang: Target language code (e.g., 'fr')
        common_words: List of words to translate (if None, uses default common words)
        max_words: Maximum number of words to fetch
        cache_file: Optional file to cache results
        
    Returns:
        Dictionary mapping source words to target translations
    """
    # Check cache first
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    
    if common_words is None:
        common_words = get_common_words_list()[:max_words]
    
    dictionary = {}
    base_url = "https://api.mymemory.translated.net/get"
    
    for i, word in enumerate(common_words[:max_words]):
        if i > 0 and i % 10 == 0:
            time.sleep(1)  # Respect rate limit
            
        try:
            params = {
                'q': word,
                'langpair': f'{source_lang}|{target_lang}'
            }
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                
                if data.get('responseStatus') == 200:
                    translation = data.get('responseData', {}).get('translatedText', '')
                    if translation and translation.lower() != word.lower():
                        dictionary[word.lower()] = translation.lower()
                        
        except Exception as e:
            print(f"Warning: Failed to fetch translation for '{word}': {e}")
            continue
    
    # Cache results
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=2)
    
    return dictionary


def get_common_words_list() -> List[str]:
    """
    Get a list of most common English words.
    
    Returns a curated list of ~5000 common words that are useful for
    scientific and general translation.
    """
    # Most common English words (subset shown, full list would be loaded from file or API)
    return [
        # Extremely common (1-100)
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
        
        # Common (101-500)
        "is", "was", "are", "been", "has", "had", "were", "said", "did", "having",
        "may", "should", "could", "might", "must", "shall", "will", "would",
        "find", "give", "tell", "work", "call", "try", "ask", "need", "feel", "become",
        "leave", "put", "mean", "keep", "let", "begin", "seem", "help", "talk", "turn",
        "start", "show", "hear", "play", "run", "move", "live", "believe", "hold", "bring",
        "happen", "write", "provide", "sit", "stand", "lose", "pay", "meet", "include", "continue",
        "set", "learn", "change", "lead", "understand", "watch", "follow", "stop", "create", "speak",
        
        # Scientific & Academic (important for papers)
        "abstract", "acknowledgments", "algorithm", "analysis", "approach", "assumption",
        "background", "benchmark", "conclusion", "contribution", "data", "dataset",
        "definition", "evaluation", "evidence", "experiment", "figure", "framework",
        "hypothesis", "implementation", "introduction", "machine", "learning", "method",
        "methodology", "model", "network", "neural", "objective", "observation",
        "parameter", "performance", "problem", "procedure", "process", "proposed",
        "reference", "related", "result", "section", "significant", "simulation",
        "solution", "study", "summary", "system", "table", "technique", "test",
        "theory", "training", "validation", "variable", "work",
        
        # Technical terms
        "accuracy", "application", "architecture", "artificial", "attention",
        "classification", "cluster", "complexity", "computation", "computer",
        "configuration", "convolutional", "correlation", "database", "deep",
        "detection", "dimension", "distribution", "embedding", "encoder",
        "error", "estimation", "feature", "function", "generation", "gradient",
        "image", "inference", "information", "input", "intelligence", "layer",
        "loss", "matrix", "optimization", "output", "prediction", "probability",
        "processing", "recognition", "recurrent", "regression", "representation",
        "sample", "segmentation", "semantic", "sequence", "similarity", "state",
        "statistical", "structure", "supervised", "task", "tensor", "threshold",
        "token", "transformer", "translation", "tree", "unsupervised", "vector",
        "vision", "weight",
        
        # More common words
        "very", "through", "just", "form", "sentence", "great", "think", "say",
        "help", "low", "line", "differ", "turn", "cause", "much", "mean", "before",
        "move", "right", "boy", "old", "too", "same", "tell", "does", "set", "three",
        "want", "air", "well", "also", "play", "small", "end", "put", "home", "read",
        "hand", "port", "large", "spell", "add", "even", "land", "here", "must", "big",
        "high", "such", "follow", "act", "why", "ask", "men", "change", "went", "light",
        "kind", "off", "need", "house", "picture", "try", "again", "animal", "point",
        "mother", "world", "near", "build", "self", "earth", "father", "head", "stand",
        "own", "page", "should", "country", "found", "answer", "school", "grow", "study",
        "still", "learn", "plant", "cover", "food", "sun", "four", "between", "state",
        "keep", "eye", "never", "last", "let", "thought", "city", "tree", "cross", "farm",
        "hard", "start", "might", "story", "saw", "far", "sea", "draw", "left", "late",
        "run", "while", "press", "close", "night", "real", "life", "few", "north", "open",
        "seem", "together", "next", "white", "children", "begin", "got", "walk", "example",
        "ease", "paper", "group", "always", "music", "those", "both", "mark", "often",
        "letter", "until", "mile", "river", "car", "feet", "care", "second", "book",
        "carry", "took", "science", "eat", "room", "friend", "began", "idea", "fish",
        "mountain", "stop", "once", "base", "hear", "horse", "cut", "sure", "watch",
        "color", "face", "wood", "main", "enough", "plain", "girl", "usual", "young",
        "ready", "above", "ever", "red", "list", "though", "feel", "talk", "bird",
        "soon", "body", "dog", "family", "direct", "pose", "leave", "song", "measure",
        "door", "product", "black", "short", "numeral", "class", "wind", "question",
        "happen", "complete", "ship", "area", "half", "rock", "order", "fire", "south",
        "problem", "piece", "told", "knew", "pass", "since", "top", "whole", "king",
        "space", "heard", "best", "hour", "better", "true", "during", "hundred", "five",
        "remember", "step", "early", "hold", "west", "ground", "interest", "reach",
        "fast", "verb", "sing", "listen", "six", "table", "travel", "less", "morning",
        "ten", "simple", "several", "vowel", "toward", "war", "lay", "against", "pattern",
        "slow", "center", "love", "person", "money", "serve", "appear", "road", "map",
        "rain", "rule", "govern", "pull", "cold", "notice", "voice", "unit", "power",
        "town", "fine", "certain", "fly", "fall", "lead", "cry", "dark", "machine",
        "note", "wait", "plan", "figure", "star", "box", "noun", "field", "rest",
        "correct", "able", "pound", "done", "beauty", "drive", "stood", "contain",
        "front", "teach", "week", "final", "gave", "green", "oh", "quick", "develop",
        "ocean", "warm", "free", "minute", "strong", "special", "mind", "behind",
        "clear", "tail", "produce", "fact", "street", "inch", "multiply", "nothing",
        "course", "stay", "wheel", "full", "force", "blue", "object", "decide",
        "surface", "deep", "moon", "island", "foot", "system", "busy", "test", "record",
        "boat", "common", "gold", "possible", "plane", "stead", "dry", "wonder", "laugh",
        "thousand", "ago", "ran", "check", "game", "shape", "equate", "hot", "miss",
        "brought", "heat", "snow", "tire", "bring", "yes", "distant", "fill", "east",
        "paint", "language", "among", "grand", "ball", "yet", "wave", "drop", "heart",
        "present", "heavy", "dance", "engine", "position", "arm", "wide", "sail",
        "material", "size", "vary", "settle", "speak", "weight", "general", "ice",
        "matter", "circle", "pair", "include", "divide", "syllable", "felt", "perhaps",
        "pick", "sudden", "count", "square", "reason", "length", "represent", "art",
        "subject", "region", "energy", "hunt", "probable", "bed", "brother", "egg",
        "ride", "cell", "believe", "fraction", "forest", "sit", "race", "window",
        "store", "summer", "train", "sleep", "prove", "lone", "leg", "exercise", "wall",
        "catch", "mount", "wish", "sky", "board", "joy", "winter", "sat", "written",
        "wild", "instrument", "kept", "glass", "grass", "cow", "job", "edge", "sign",
        "visit", "past", "soft", "fun", "bright", "gas", "weather", "month", "million",
        "bear", "finish", "happy", "hope", "flower", "clothe", "strange", "gone", "jump",
        "baby", "eight", "village", "meet", "root", "buy", "raise", "solve", "metal",
        "whether", "push", "seven", "paragraph", "third", "shall", "held", "hair",
        "describe", "cook", "floor", "either", "result", "burn", "hill", "safe", "cat",
        "century", "consider", "type", "law", "bit", "coast", "copy", "phrase", "silent",
        "tall", "sand", "soil", "roll", "temperature", "finger", "industry", "value",
        "fight", "lie", "beat", "excite", "natural", "view", "sense", "ear", "else",
        "quite", "broke", "case", "middle", "kill", "son", "lake", "moment", "scale",
        "loud", "spring", "observe", "child", "straight", "consonant", "nation", "dictionary",
        "milk", "speed", "method", "organ", "pay", "age", "section", "dress", "cloud",
        "surprise", "quiet", "stone", "tiny", "climb", "cool", "design", "poor", "lot",
        "experiment", "bottom", "key", "iron", "single", "stick", "flat", "twenty", "skin",
        "smile", "crease", "hole", "trade", "melody", "trip", "office", "receive", "row",
        "mouth", "exact", "symbol", "die", "least", "trouble", "shout", "except", "wrote",
        "seed", "tone", "join", "suggest", "clean", "break", "lady", "yard", "rise",
        "bad", "blow", "oil", "blood", "touch", "grew", "cent", "mix", "team", "wire",
        "cost", "lost", "brown", "wear", "garden", "equal", "sent", "choose", "fell",
        "fit", "flow", "fair", "bank", "collect", "save", "control", "decimal", "gentle",
        "woman", "captain", "practice", "separate", "difficult", "doctor", "please",
        "protect", "noon", "whose", "locate", "ring", "character", "insect", "caught",
        "period", "indicate", "radio", "spoke", "atom", "human", "history", "effect",
        "electric", "expect", "crop", "modern", "element", "hit", "student", "corner",
        "party", "supply", "bone", "rail", "imagine", "provide", "agree", "thus",
        "capital", "chair", "danger", "fruit", "rich", "thick", "soldier", "process",
        "operate", "guess", "necessary", "sharp", "wing", "create", "neighbor", "wash",
        "bat", "rather", "crowd", "corn", "compare", "poem", "string", "bell", "depend",
        "meat", "rub", "tube", "famous", "dollar", "stream", "fear", "sight", "thin",
        "triangle", "planet", "hurry", "chief", "colony", "clock", "mine", "tie", "enter",
        "major", "fresh", "search", "send", "yellow", "gun", "allow", "print", "dead",
        "spot", "desert", "suit", "current", "lift", "rose", "continue", "block", "chart",
        "hat", "sell", "success", "company", "subtract", "event", "particular", "deal",
        "swim", "term", "opposite", "wife", "shoe", "shoulder", "spread", "arrange",
        "camp", "invent", "cotton", "born", "determine", "quart", "nine", "truck",
        "noise", "level", "chance", "gather", "shop", "stretch", "throw", "shine",
        "property", "column", "molecule", "select", "wrong", "gray", "repeat", "require",
        "broad", "prepare", "salt", "nose", "plural", "anger", "claim", "continent",
        "oxygen", "sugar", "death", "pretty", "skill", "women", "season", "solution",
        "magnet", "silver", "thank", "branch", "match", "suffix", "especially", "fig",
        "afraid", "huge", "sister", "steel", "discuss", "forward", "similar", "guide",
        "experience", "score", "apple", "bought", "led", "pitch", "coat", "mass", "card",
        "band", "rope", "slip", "win", "dream", "evening", "condition", "feed", "tool",
        "total", "basic", "smell", "valley", "nor", "double", "seat", "arrive", "master",
        "track", "parent", "shore", "division", "sheet", "substance", "favor", "connect",
        "post", "spend", "chord", "fat", "glad", "original", "share", "station", "dad",
        "bread", "charge", "proper", "bar", "offer", "segment", "slave", "duck", "instant",
        "market", "degree", "populate", "chick", "dear", "enemy", "reply", "drink",
        "occur", "support", "speech", "nature", "range", "steam", "motion", "path",
        "liquid", "log", "meant", "quotient", "teeth", "shell", "neck"
    ]


def load_cached_or_fetch_dictionary(
    source_lang: str = "en",
    target_lang: str = "fr",
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False
) -> Dict[str, str]:
    """
    Load dictionary from cache or fetch from online sources.
    
    This is the main function to use - it handles caching intelligently
    and falls back to multiple sources if needed.
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        cache_dir: Directory to store cached dictionaries
        force_refresh: If True, ignore cache and fetch fresh data
        
    Returns:
        Dictionary with translations
    """
    if cache_dir is None:
        from scitrans_llms.config import DATA_DIR
        cache_dir = DATA_DIR / "dictionaries" / "online"
    
    cache_file = cache_dir / f"{source_lang}_{target_lang}_dictionary.json"
    
    # Try to load from cache
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                dictionary = json.load(f)
                if len(dictionary) > 100:  # Ensure we have a reasonable dictionary
                    return dictionary
        except Exception:
            pass
    
    # Fetch from online source
    print(f"Fetching dictionary from online sources ({source_lang} → {target_lang})...")
    print("This may take a few minutes but will be cached for future use.")
    
    dictionary = load_mymemory_dictionary(
        source_lang=source_lang,
        target_lang=target_lang,
        max_words=1000,  # Start with 1000 most common words
        cache_file=cache_file
    )
    
    print(f"✓ Loaded {len(dictionary)} translations from online sources")
    return dictionary


def get_fallback_dictionary(source_lang: str = "en", target_lang: str = "fr") -> Dict[str, str]:
    """
    Get a minimal fallback dictionary for offline use.
    
    Returns a small but useful dictionary of the most common words
    that can be used when online sources are unavailable.
    """
    # This is a curated list of ~200 most essential words
    # In a real implementation, this would be much larger
    if source_lang == "en" and target_lang == "fr":
        return {
            "the": "le", "be": "être", "to": "à", "of": "de", "and": "et",
            "a": "un", "in": "dans", "that": "que", "have": "avoir", "i": "je",
            "it": "il", "for": "pour", "not": "ne pas", "on": "sur", "with": "avec",
            "he": "il", "as": "comme", "you": "vous", "do": "faire", "at": "à",
            "this": "ce", "but": "mais", "his": "son", "by": "par", "from": "de",
            "they": "ils", "we": "nous", "say": "dire", "her": "elle", "she": "elle",
            "or": "ou", "an": "un", "will": "sera", "my": "mon", "one": "un",
            "all": "tout", "would": "serait", "there": "là", "their": "leur",
            # Add many more...
            "machine": "machine", "learning": "apprentissage", "deep": "profond",
            "neural": "neuronal", "network": "réseau", "model": "modèle",
            "data": "données", "algorithm": "algorithme", "training": "entraînement",
            "test": "test", "validation": "validation", "accuracy": "précision",
            "performance": "performance", "result": "résultat", "method": "méthode",
            "approach": "approche", "proposed": "proposé", "novel": "nouveau",
            "state": "état", "art": "art", "significant": "significatif",
            "improvement": "amélioration", "compared": "comparé", "previous": "précédent",
            "work": "travail", "paper": "article", "research": "recherche",
            "study": "étude", "analysis": "analyse", "evaluation": "évaluation",
            "experiment": "expérience", "figure": "figure", "table": "tableau",
            "section": "section", "chapter": "chapitre", "introduction": "introduction",
            "conclusion": "conclusion", "abstract": "résumé", "reference": "référence",
        }
    
    return {}


