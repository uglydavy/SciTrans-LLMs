"""
Online corpus manager for downloadable translation corpora.

Supports:
- Europarl parallel corpus
- OPUS corpora
- WMT datasets
- Custom corpus uploads

Usage:
    from scitrans_llms.translate.corpus_manager import CorpusManager
    
    manager = CorpusManager()
    manager.download("europarl", "en", "fr")
    
    # Use corpus for translation memory / dictionary enhancement
    entries = manager.get_entries("europarl", "en", "fr", limit=10000)
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urljoin


@dataclass
class CorpusInfo:
    """Information about a downloadable corpus."""
    name: str
    description: str
    url_template: str
    languages: list[tuple[str, str]]  # Available language pairs
    format: str = "tmx"  # tmx, tsv, txt
    size_mb: int = 0
    license: str = "unknown"


# Available corpora
AVAILABLE_CORPORA = {
    "europarl": CorpusInfo(
        name="Europarl",
        description="European Parliament proceedings parallel corpus",
        url_template="https://www.statmt.org/europarl/v7/{src}-{tgt}.tgz",
        languages=[
            ("en", "fr"), ("en", "de"), ("en", "es"), ("en", "it"),
            ("en", "pt"), ("en", "nl"), ("en", "el"), ("en", "da"),
            ("fr", "en"), ("de", "en"), ("es", "en"),
        ],
        format="txt",
        size_mb=200,
        license="Public Domain",
    ),
    "opus-euconst": CorpusInfo(
        name="OPUS EU Constitution",
        description="EU Constitution in multiple languages (smaller)",
        url_template="https://opus.nlpl.eu/download.php?f=EUconst/v1/moses/{src}-{tgt}.txt.zip",
        languages=[
            ("en", "fr"), ("en", "de"), ("en", "es"), ("en", "it"),
            ("fr", "en"), ("de", "en"),
        ],
        format="txt",
        size_mb=5,
        license="Public Domain",
    ),
    "tatoeba": CorpusInfo(
        name="Tatoeba",
        description="Crowdsourced sentence pairs from Tatoeba",
        url_template="https://downloads.tatoeba.org/exports/per_language/{src}/{src}-{tgt}_sentences.tsv.bz2",
        languages=[
            ("en", "fr"), ("en", "de"), ("en", "es"), ("en", "ja"),
            ("en", "zh"), ("fr", "en"), ("de", "en"),
        ],
        format="tsv",
        size_mb=50,
        license="CC BY 2.0",
    ),
}


@dataclass
class CorpusEntry:
    """A single entry from a parallel corpus."""
    source: str
    target: str
    metadata: dict = field(default_factory=dict)


class CorpusManager:
    """Manages downloadable translation corpora.
    
    Corpora are cached locally for reuse.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".scitrans" / "corpora"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_corpora: dict[str, list[CorpusEntry]] = {}
    
    def list_available(self) -> list[CorpusInfo]:
        """List all available corpora."""
        return list(AVAILABLE_CORPORA.values())
    
    def list_downloaded(self) -> list[str]:
        """List locally cached corpora."""
        downloaded = []
        for name in AVAILABLE_CORPORA:
            corpus_dir = self.cache_dir / name
            if corpus_dir.exists() and any(corpus_dir.iterdir()):
                downloaded.append(name)
        return downloaded
    
    def is_downloaded(self, corpus_name: str, src_lang: str, tgt_lang: str) -> bool:
        """Check if a corpus is already downloaded."""
        data_file = self._get_data_path(corpus_name, src_lang, tgt_lang)
        return data_file.exists()
    
    def download(
        self,
        corpus_name: str,
        src_lang: str,
        tgt_lang: str,
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """Download a corpus if not already cached.
        
        Args:
            corpus_name: Name of corpus (e.g., 'europarl')
            src_lang: Source language code
            tgt_lang: Target language code
            progress_callback: Optional callback(message, progress)
            
        Returns:
            Path to downloaded corpus data
        """
        if corpus_name not in AVAILABLE_CORPORA:
            raise ValueError(f"Unknown corpus: {corpus_name}. Available: {list(AVAILABLE_CORPORA.keys())}")
        
        corpus_info = AVAILABLE_CORPORA[corpus_name]
        
        # Check if language pair is supported
        if (src_lang, tgt_lang) not in corpus_info.languages:
            # Try reverse
            if (tgt_lang, src_lang) in corpus_info.languages:
                src_lang, tgt_lang = tgt_lang, src_lang
            else:
                raise ValueError(f"Language pair {src_lang}-{tgt_lang} not available for {corpus_name}")
        
        data_path = self._get_data_path(corpus_name, src_lang, tgt_lang)
        
        if data_path.exists():
            if progress_callback:
                progress_callback(f"Corpus already downloaded: {data_path}", 1.0)
            return data_path
        
        # Download
        url = corpus_info.url_template.format(src=src_lang, tgt=tgt_lang)
        
        if progress_callback:
            progress_callback(f"Downloading {corpus_name} ({src_lang}-{tgt_lang})...", 0.1)
        
        try:
            import requests
            
            # Create corpus directory
            corpus_dir = self.cache_dir / corpus_name
            corpus_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Save to temp file
            temp_path = corpus_dir / f"temp_{src_lang}_{tgt_lang}"
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size:
                        pct = 0.1 + 0.7 * (downloaded / total_size)
                        progress_callback(f"Downloading... {downloaded // 1024}KB", pct)
            
            if progress_callback:
                progress_callback("Extracting...", 0.8)
            
            # Extract and process
            self._extract_corpus(temp_path, data_path, corpus_info.format)
            
            # Clean up
            temp_path.unlink(missing_ok=True)
            
            if progress_callback:
                progress_callback(f"Downloaded to {data_path}", 1.0)
            
            return data_path
            
        except ImportError:
            raise ImportError("requests library required. Install with: pip install requests")
        except Exception as e:
            raise RuntimeError(f"Failed to download corpus: {e}")
    
    def get_entries(
        self,
        corpus_name: str,
        src_lang: str,
        tgt_lang: str,
        limit: Optional[int] = None,
        min_length: int = 5,
        max_length: int = 500,
    ) -> Iterator[CorpusEntry]:
        """Get entries from a downloaded corpus.
        
        Args:
            corpus_name: Name of corpus
            src_lang: Source language code
            tgt_lang: Target language code
            limit: Maximum entries to return
            min_length: Minimum source text length
            max_length: Maximum source text length
            
        Yields:
            CorpusEntry objects
        """
        data_path = self._get_data_path(corpus_name, src_lang, tgt_lang)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Corpus not downloaded. Run: corpus.download('{corpus_name}', '{src_lang}', '{tgt_lang}')")
        
        count = 0
        
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if limit and count >= limit:
                    break
                
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    src, tgt = parts[0], parts[1]
                    
                    if min_length <= len(src) <= max_length:
                        yield CorpusEntry(source=src, target=tgt)
                        count += 1
    
    def build_dictionary(
        self,
        corpus_name: str,
        src_lang: str,
        tgt_lang: str,
        limit: int = 50000,
    ) -> dict[str, str]:
        """Build a translation dictionary from corpus.
        
        Extracts unique phrase pairs suitable for use as glossary entries.
        
        Args:
            corpus_name: Name of corpus
            src_lang: Source language
            tgt_lang: Target language
            limit: Maximum entries
            
        Returns:
            Dict mapping source phrases to target translations
        """
        dictionary = {}
        
        for entry in self.get_entries(corpus_name, src_lang, tgt_lang, limit=limit * 2):
            src = entry.source.strip().lower()
            tgt = entry.target.strip()
            
            # Skip too short/long entries
            if len(src) < 10 or len(src) > 100:
                continue
            
            # Skip entries with special characters (URLs, emails, etc.)
            if re.search(r'[<>@:/]', src):
                continue
            
            # Store (prefer shorter translations if duplicate)
            if src not in dictionary or len(tgt) < len(dictionary[src]):
                dictionary[src] = tgt
            
            if len(dictionary) >= limit:
                break
        
        return dictionary
    
    def enhance_glossary(
        self,
        glossary: dict[str, str],
        corpus_name: str,
        src_lang: str,
        tgt_lang: str,
        max_additions: int = 1000,
    ) -> dict[str, str]:
        """Enhance an existing glossary with corpus entries.
        
        Adds entries from the corpus that are not already in the glossary.
        """
        enhanced = dict(glossary)
        additions = 0
        
        for entry in self.get_entries(corpus_name, src_lang, tgt_lang, limit=max_additions * 10):
            if additions >= max_additions:
                break
            
            src_lower = entry.source.strip().lower()
            
            # Only add if not already present
            if src_lower not in {k.lower() for k in enhanced}:
                # Only add "glossary-worthy" entries (short, specific)
                if 5 <= len(entry.source) <= 50 and not re.search(r'[<>@:/]', entry.source):
                    enhanced[entry.source.strip()] = entry.target.strip()
                    additions += 1
        
        return enhanced
    
    def _get_data_path(self, corpus_name: str, src_lang: str, tgt_lang: str) -> Path:
        """Get path to processed corpus data file."""
        return self.cache_dir / corpus_name / f"{src_lang}_{tgt_lang}.tsv"
    
    def _extract_corpus(self, archive_path: Path, output_path: Path, format: str):
        """Extract and process downloaded corpus archive."""
        import tarfile
        import zipfile
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine archive type and extract
        entries = []
        
        try:
            if str(archive_path).endswith('.tgz') or str(archive_path).endswith('.tar.gz'):
                with tarfile.open(archive_path, 'r:gz') as tar:
                    entries = self._process_tar_archive(tar)
            elif str(archive_path).endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    entries = self._process_zip_archive(zf)
            elif str(archive_path).endswith('.gz'):
                with gzip.open(archive_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    entries = self._process_text_file(f)
            else:
                # Assume plain text/tsv
                with open(archive_path, 'r', encoding='utf-8', errors='ignore') as f:
                    entries = self._process_text_file(f)
        except Exception as e:
            # Fallback: try to read as plain text
            try:
                with open(archive_path, 'r', encoding='utf-8', errors='ignore') as f:
                    entries = self._process_text_file(f)
            except Exception:
                raise RuntimeError(f"Failed to extract corpus: {e}")
        
        # Write processed entries
        with open(output_path, 'w', encoding='utf-8') as f:
            for src, tgt in entries:
                f.write(f"{src}\t{tgt}\n")
    
    def _process_tar_archive(self, tar) -> list[tuple[str, str]]:
        """Process a tar archive containing parallel text files."""
        entries = []
        src_lines = []
        tgt_lines = []
        
        for member in tar.getmembers():
            if member.isfile():
                content = tar.extractfile(member)
                if content:
                    text = content.read().decode('utf-8', errors='ignore')
                    lines = text.strip().split('\n')
                    
                    # Guess if it's source or target based on filename
                    name = member.name.lower()
                    if 'en' in name or 'source' in name:
                        src_lines.extend(lines)
                    elif any(lang in name for lang in ['fr', 'de', 'es', 'target']):
                        tgt_lines.extend(lines)
        
        # Pair up lines
        for src, tgt in zip(src_lines, tgt_lines):
            src, tgt = src.strip(), tgt.strip()
            if src and tgt:
                entries.append((src, tgt))
        
        return entries
    
    def _process_zip_archive(self, zf) -> list[tuple[str, str]]:
        """Process a zip archive."""
        entries = []
        
        for name in zf.namelist():
            if name.endswith('.txt') or name.endswith('.tsv'):
                with zf.open(name) as f:
                    text = f.read().decode('utf-8', errors='ignore')
                    for line in text.strip().split('\n'):
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            entries.append((parts[0].strip(), parts[1].strip()))
        
        return entries
    
    def _process_text_file(self, f) -> list[tuple[str, str]]:
        """Process a plain text or TSV file."""
        entries = []
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                entries.append((parts[0].strip(), parts[1].strip()))
        
        return entries


# Convenience functions for CLI/GUI

def list_corpora() -> list[dict]:
    """List available corpora with info."""
    return [
        {
            "name": info.name,
            "key": key,
            "description": info.description,
            "languages": [f"{s}-{t}" for s, t in info.languages],
            "size_mb": info.size_mb,
            "license": info.license,
        }
        for key, info in AVAILABLE_CORPORA.items()
    ]


def download_corpus(
    name: str,
    src_lang: str = "en",
    tgt_lang: str = "fr",
    progress: Optional[callable] = None,
) -> Path:
    """Download a corpus (convenience function).
    
    Args:
        name: Corpus name (europarl, opus-euconst, tatoeba)
        src_lang: Source language
        tgt_lang: Target language
        progress: Optional progress callback
        
    Returns:
        Path to downloaded data
    """
    manager = CorpusManager()
    return manager.download(name, src_lang, tgt_lang, progress)


def get_corpus_dictionary(
    name: str,
    src_lang: str = "en",
    tgt_lang: str = "fr",
    limit: int = 10000,
) -> dict[str, str]:
    """Get a dictionary from a corpus.
    
    Args:
        name: Corpus name
        src_lang: Source language
        tgt_lang: Target language
        limit: Maximum entries
        
    Returns:
        Dictionary mapping source to target
    """
    manager = CorpusManager()
    
    # Download if needed
    if not manager.is_downloaded(name, src_lang, tgt_lang):
        manager.download(name, src_lang, tgt_lang)
    
    return manager.build_dictionary(name, src_lang, tgt_lang, limit)


__all__ = [
    'CorpusManager',
    'CorpusInfo', 
    'CorpusEntry',
    'AVAILABLE_CORPORA',
    'list_corpora',
    'download_corpus',
    'get_corpus_dictionary',
]

