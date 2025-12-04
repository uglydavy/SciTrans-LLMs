"""
Core data models for SciTrans-LLMs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import json


class BlockType(Enum):
    """Types of document blocks."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"  
    EQUATION = "equation"
    CODE = "code"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    UNKNOWN = "unknown"


@dataclass
class Block:
    """A single block in a document."""
    text: str
    block_type: BlockType = BlockType.PARAGRAPH
    metadata: Dict[str, Any] = field(default_factory=dict)
    translation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "block_type": self.block_type.value,
            "metadata": self.metadata,
            "translation": self.translation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Block":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            block_type=BlockType(data.get("block_type", "paragraph")),
            metadata=data.get("metadata", {}),
            translation=data.get("translation")
        )


@dataclass
class Document:
    """A document with blocks."""
    blocks: List[Block] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_block(self, text: str, block_type: BlockType = BlockType.PARAGRAPH, **metadata):
        """Add a block to the document."""
        block = Block(text=text, block_type=block_type, metadata=metadata)
        self.blocks.append(block)
        return block
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "blocks": [block.to_dict() for block in self.blocks],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create from dictionary."""
        doc = cls(metadata=data.get("metadata", {}))
        for block_data in data.get("blocks", []):
            doc.blocks.append(Block.from_dict(block_data))
        return doc
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Document":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class GlossaryEntry:
    """A single glossary entry."""
    source: str
    target: str
    domain: str = "general"
    notes: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "domain": self.domain,
            "notes": self.notes
        }


@dataclass
class Glossary:
    """A collection of glossary entries."""
    entries: List[GlossaryEntry] = field(default_factory=list)
    
    def add_entry(self, source: str, target: str, domain: str = "general", notes: str = ""):
        """Add an entry to the glossary."""
        entry = GlossaryEntry(source, target, domain, notes)
        self.entries.append(entry)
        return entry
    
    def get_translation(self, term: str) -> Optional[str]:
        """Get translation for a term."""
        for entry in self.entries:
            if entry.source.lower() == term.lower():
                return entry.target
        return None
    
    def to_dict(self) -> List[Dict[str, str]]:
        """Convert to dictionary."""
        return [entry.to_dict() for entry in self.entries]
    
    def copy(self) -> "Glossary":
        """Create a copy of the glossary."""
        new_glossary = Glossary()
        for entry in self.entries:
            new_glossary.add_entry(entry.source, entry.target, entry.domain, entry.notes)
        return new_glossary
