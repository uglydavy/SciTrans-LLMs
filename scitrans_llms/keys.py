"""
API key management for SciTrans-LLMs.

Provides secure storage and retrieval of API keys using:
1. Environment variables (preferred for CI/production)
2. OS keychain via keyring (secure local storage)
3. Local config file (fallback)

Usage:
    from scitrans_llms.keys import KeyManager
    
    km = KeyManager()
    km.set_key("openai", "sk-...")
    key = km.get_key("openai")
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


# Supported services and their env var names
SERVICES = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepl": "DEEPL_API_KEY",
    "google": "GOOGLE_API_KEY",
    "comet": "COMET_API_KEY",
}


@dataclass
class KeyInfo:
    """Information about an API key."""
    service: str
    is_set: bool
    source: str  # 'env', 'keyring', 'config', 'none'
    masked_value: str  # e.g., "sk-...abc"


class KeyManager:
    """Manage API keys securely.
    
    Priority order for key retrieval:
    1. Environment variable
    2. OS keychain (via keyring)
    3. Local config file (~/.scitrans/keys.json)
    """
    
    SERVICE_NAME = "SciTrans-LLMs"
    CONFIG_DIR = Path.home() / ".scitrans"
    CONFIG_FILE = CONFIG_DIR / "keys.json"
    
    def __init__(self):
        self._keyring_available = self._check_keyring()
    
    def _check_keyring(self) -> bool:
        """Check if keyring is available."""
        try:
            import keyring
            # Test if we can access keyring
            keyring.get_keyring()
            return True
        except Exception:
            return False
    
    def get_key(self, service: str) -> Optional[str]:
        """Get API key for a service.
        
        Args:
            service: Service name (openai, deepseek, anthropic, etc.)
            
        Returns:
            API key string or None if not found
        """
        service = service.lower()
        
        # 1. Check environment variable
        env_var = SERVICES.get(service, f"{service.upper()}_API_KEY")
        if env_val := os.getenv(env_var):
            return env_val
        
        # 2. Check keyring
        if self._keyring_available:
            try:
                import keyring
                if key := keyring.get_password(self.SERVICE_NAME, service):
                    return key
            except Exception:
                pass
        
        # 3. Check config file
        if self.CONFIG_FILE.exists():
            try:
                config = json.loads(self.CONFIG_FILE.read_text())
                if key := config.get(service):
                    return key
            except Exception:
                pass
        
        return None
    
    def set_key(self, service: str, key: str, use_keyring: bool = True) -> str:
        """Store API key for a service.
        
        Args:
            service: Service name
            key: API key to store
            use_keyring: Whether to use OS keychain (if available)
            
        Returns:
            Storage location used ('keyring' or 'config')
        """
        service = service.lower()
        
        # Try keyring first
        if use_keyring and self._keyring_available:
            try:
                import keyring
                keyring.set_password(self.SERVICE_NAME, service, key)
                return "keyring"
            except Exception:
                pass
        
        # Fallback to config file
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        config = {}
        if self.CONFIG_FILE.exists():
            try:
                config = json.loads(self.CONFIG_FILE.read_text())
            except Exception:
                pass
        
        config[service] = key
        self.CONFIG_FILE.write_text(json.dumps(config, indent=2))
        self.CONFIG_FILE.chmod(0o600)  # Restrict permissions
        
        return "config"
    
    def delete_key(self, service: str) -> bool:
        """Delete stored API key for a service."""
        service = service.lower()
        deleted = False
        
        # Delete from keyring
        if self._keyring_available:
            try:
                import keyring
                keyring.delete_password(self.SERVICE_NAME, service)
                deleted = True
            except Exception:
                pass
        
        # Delete from config
        if self.CONFIG_FILE.exists():
            try:
                config = json.loads(self.CONFIG_FILE.read_text())
                if service in config:
                    del config[service]
                    self.CONFIG_FILE.write_text(json.dumps(config, indent=2))
                    deleted = True
            except Exception:
                pass
        
        return deleted
    
    def get_key_info(self, service: str) -> KeyInfo:
        """Get information about a stored key."""
        service = service.lower()
        env_var = SERVICES.get(service, f"{service.upper()}_API_KEY")
        
        # Check environment
        if env_val := os.getenv(env_var):
            return KeyInfo(
                service=service,
                is_set=True,
                source="env",
                masked_value=self._mask_key(env_val),
            )
        
        # Check keyring
        if self._keyring_available:
            try:
                import keyring
                if key := keyring.get_password(self.SERVICE_NAME, service):
                    return KeyInfo(
                        service=service,
                        is_set=True,
                        source="keyring",
                        masked_value=self._mask_key(key),
                    )
            except Exception:
                pass
        
        # Check config
        if self.CONFIG_FILE.exists():
            try:
                config = json.loads(self.CONFIG_FILE.read_text())
                if key := config.get(service):
                    return KeyInfo(
                        service=service,
                        is_set=True,
                        source="config",
                        masked_value=self._mask_key(key),
                    )
            except Exception:
                pass
        
        return KeyInfo(
            service=service,
            is_set=False,
            source="none",
            masked_value="",
        )
    
    def list_keys(self) -> list[KeyInfo]:
        """List all configured services and their key status."""
        return [self.get_key_info(service) for service in SERVICES]
    
    def _mask_key(self, key: str) -> str:
        """Mask a key for display (show first 4 and last 4 chars)."""
        if len(key) <= 12:
            return "*" * len(key)
        return f"{key[:4]}...{key[-4:]}"
    
    def export_to_env(self) -> dict[str, str]:
        """Export all keys as environment variables dict."""
        env = {}
        for service, env_var in SERVICES.items():
            if key := self.get_key(service):
                env[env_var] = key
        return env
    
    def setup_environment(self):
        """Set environment variables from stored keys."""
        for env_var, value in self.export_to_env().items():
            os.environ[env_var] = value


def get_key(service: str) -> Optional[str]:
    """Convenience function to get an API key."""
    return KeyManager().get_key(service)


def require_key(service: str) -> str:
    """Get API key or raise error if not found."""
    key = get_key(service)
    if not key:
        raise ValueError(
            f"API key for '{service}' not found. "
            f"Set {SERVICES.get(service, service.upper() + '_API_KEY')} environment variable "
            f"or run: scitrans keys set {service}"
        )
    return key

