#!/usr/bin/env python3
"""
Interactive API key setup for SciTrans-Next.

This script helps you configure API keys for LLM backends.

Usage:
    python scripts/setup_keys.py
    python scripts/setup_keys.py --service openai
    python scripts/setup_keys.py --list
"""

import argparse
import sys
import getpass
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm

console = Console()


def list_keys():
    """List all configured API keys."""
    from scitrans_next.keys import KeyManager
    
    km = KeyManager()
    keys = km.list_keys()
    
    table = Table(title="API Key Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Source")
    table.add_column("Value")
    
    for info in keys:
        status = "✓ Set" if info.is_set else "○ Not set"
        style = "green" if info.is_set else "dim"
        table.add_row(
            info.service,
            status,
            info.source if info.is_set else "",
            info.masked_value if info.is_set else "",
            style=style,
        )
    
    console.print(table)


def set_key(service: str):
    """Set API key for a service."""
    from scitrans_next.keys import KeyManager, SERVICES
    
    km = KeyManager()
    
    # Check current status
    info = km.get_key_info(service)
    if info.is_set:
        console.print(f"[yellow]Key already set for {service}: {info.masked_value}[/]")
        if not Confirm.ask("Do you want to replace it?"):
            return
    
    # Get key from user
    env_var = SERVICES.get(service, f"{service.upper()}_API_KEY")
    console.print(f"\n[dim]You can also set this via environment variable: {env_var}[/]")
    
    key = getpass.getpass(f"Enter API key for {service}: ")
    
    if not key.strip():
        console.print("[red]No key provided, aborting.[/]")
        return
    
    # Store key
    storage = km.set_key(service, key.strip())
    console.print(f"[green]✓ Key stored in {storage}[/]")


def delete_key(service: str):
    """Delete API key for a service."""
    from scitrans_next.keys import KeyManager
    
    km = KeyManager()
    
    info = km.get_key_info(service)
    if not info.is_set:
        console.print(f"[yellow]No key set for {service}[/]")
        return
    
    if Confirm.ask(f"Delete key for {service}?"):
        km.delete_key(service)
        console.print(f"[green]✓ Key deleted[/]")


def interactive_setup():
    """Interactive setup wizard."""
    from scitrans_next.keys import SERVICES
    
    console.print("\n[bold]SciTrans-Next API Key Setup[/]")
    console.print("=" * 40)
    console.print("\nThis wizard will help you configure API keys for translation backends.\n")
    
    for service in SERVICES:
        from scitrans_next.keys import KeyManager
        km = KeyManager()
        info = km.get_key_info(service)
        
        if info.is_set:
            console.print(f"[green]✓[/] {service}: {info.masked_value}")
        else:
            if Confirm.ask(f"Configure {service}?", default=False):
                set_key(service)
            else:
                console.print(f"[dim]○ Skipping {service}[/]")
    
    console.print("\n[green]Setup complete![/]")
    list_keys()


def main():
    parser = argparse.ArgumentParser(description="Configure API keys for SciTrans-Next")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List all configured keys")
    parser.add_argument("--service", "-s", type=str,
                       help="Service to configure (openai, deepseek, anthropic, etc.)")
    parser.add_argument("--delete", "-d", action="store_true",
                       help="Delete key for specified service")
    
    args = parser.parse_args()
    
    if args.list:
        list_keys()
    elif args.service:
        if args.delete:
            delete_key(args.service.lower())
        else:
            set_key(args.service.lower())
    else:
        interactive_setup()


if __name__ == "__main__":
    main()

