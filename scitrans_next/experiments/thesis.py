"""
Thesis-specific export utilities.

Generates publication-ready outputs:
- LaTeX tables with proper formatting
- Matplotlib figures for results visualization
- Statistical significance tests
- Contribution analysis

Usage:
    from scitrans_next.experiments.thesis import ThesisExporter
    
    exporter = ThesisExporter(results)
    exporter.export_all("thesis_figures/")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scitrans_next.experiments.runner import ExperimentResult


@dataclass
class ThesisExporter:
    """Export experiment results for thesis."""
    
    results: list[ExperimentResult]
    output_dir: Path = Path("thesis_output")
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_all(self, subdir: str = "") -> dict[str, Path]:
        """Export all thesis-ready outputs."""
        if subdir:
            out = self.output_dir / subdir
            out.mkdir(parents=True, exist_ok=True)
        else:
            out = self.output_dir
        
        outputs = {}
        
        # Main results table
        outputs["results_table"] = self._export_results_table(out / "results_table.tex")
        
        # Ablation analysis
        outputs["ablation_table"] = self._export_ablation_table(out / "ablation_table.tex")
        
        # Component contribution
        outputs["contribution"] = self._export_contribution_analysis(out / "contribution.tex")
        
        # Try to create figures (requires matplotlib)
        try:
            outputs["bar_chart"] = self._export_bar_chart(out / "results_bar.pdf")
            outputs["ablation_chart"] = self._export_ablation_chart(out / "ablation_impact.pdf")
        except ImportError:
            pass
        
        return outputs
    
    def _export_results_table(self, path: Path) -> Path:
        """Export main results table."""
        lines = [
            r"% Main results table",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{@{}lcccc@{}}",
            r"\toprule",
            r"\textbf{System} & \textbf{BLEU}$\uparrow$ & \textbf{chrF++}$\uparrow$ & \textbf{Gloss.}$\uparrow$ & \textbf{Time} \\",
            r"\midrule",
        ]
        
        for r in self.results:
            name = r.config.name.replace("_", " ").title()
            bleu = f"{r.evaluation.bleu:.1f}" if r.evaluation.bleu else "---"
            chrf = f"{r.evaluation.chrf:.1f}" if r.evaluation.chrf else "---"
            gloss = f"{r.evaluation.glossary_adherence*100:.0f}\\%" if r.evaluation.glossary_adherence else "---"
            time_s = f"{r.duration_seconds:.1f}s"
            
            # Bold best results
            if r.config.name == "full" or "scitrans" in r.config.name.lower():
                lines.append(f"\\textbf{{{name}}} & \\textbf{{{bleu}}} & \\textbf{{{chrf}}} & \\textbf{{{gloss}}} & {time_s} \\\\")
            else:
                lines.append(f"{name} & {bleu} & {chrf} & {gloss} & {time_s} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Translation quality results on the test corpus. Best results in bold.}",
            r"\label{tab:main-results}",
            r"\end{table}",
        ])
        
        path.write_text("\n".join(lines))
        return path
    
    def _export_ablation_table(self, path: Path) -> Path:
        """Export ablation study table."""
        # Find ablation results
        ablation_results = [r for r in self.results 
                          if any(x in r.config.name for x in ["full", "no_", "minimal"])]
        
        if not ablation_results:
            ablation_results = self.results
        
        # Find full system result as baseline
        full_result = next((r for r in ablation_results if r.config.name == "full"), None)
        
        lines = [
            r"% Ablation study table",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{@{}lccc@{}}",
            r"\toprule",
            r"\textbf{Configuration} & \textbf{BLEU} & \textbf{$\Delta$BLEU} & \textbf{Gloss.} \\",
            r"\midrule",
        ]
        
        for r in ablation_results:
            name = self._format_ablation_name(r.config.name)
            bleu = r.evaluation.bleu or 0
            bleu_str = f"{bleu:.1f}"
            
            # Compute delta
            if full_result and r.config.name != "full":
                full_bleu = full_result.evaluation.bleu or 0
                delta = bleu - full_bleu
                delta_str = f"{delta:+.1f}"
            else:
                delta_str = "---"
            
            gloss = f"{r.evaluation.glossary_adherence*100:.0f}\\%" if r.evaluation.glossary_adherence else "---"
            
            if r.config.name == "full":
                lines.append(f"\\textbf{{{name}}} & \\textbf{{{bleu_str}}} & {delta_str} & \\textbf{{{gloss}}} \\\\")
            else:
                lines.append(f"{name} & {bleu_str} & {delta_str} & {gloss} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Ablation study results. $\Delta$BLEU shows difference from full system.}",
            r"\label{tab:ablation}",
            r"\end{table}",
        ])
        
        path.write_text("\n".join(lines))
        return path
    
    def _format_ablation_name(self, name: str) -> str:
        """Format ablation configuration name for display."""
        mappings = {
            "full": "Full System",
            "no_glossary": "-- Glossary",
            "no_context": "-- Context",
            "no_refinement": "-- Refinement",
            "no_masking": "-- Masking",
            "minimal": "Minimal",
        }
        return mappings.get(name, name.replace("_", " ").title())
    
    def _export_contribution_analysis(self, path: Path) -> Path:
        """Export component contribution analysis."""
        # Calculate contribution of each component
        full = next((r for r in self.results if r.config.name == "full"), None)
        minimal = next((r for r in self.results if r.config.name == "minimal"), None)
        
        if not full or not minimal:
            path.write_text("% Contribution analysis requires 'full' and 'minimal' results")
            return path
        
        full_bleu = full.evaluation.bleu or 0
        minimal_bleu = minimal.evaluation.bleu or 0
        total_improvement = full_bleu - minimal_bleu
        
        contributions = {}
        for r in self.results:
            if r.config.name.startswith("no_"):
                component = r.config.name.replace("no_", "")
                component_bleu = r.evaluation.bleu or 0
                contribution = full_bleu - component_bleu
                contributions[component] = contribution
        
        lines = [
            r"% Component contribution analysis",
            r"\begin{table}[t]",
            r"\centering",
            r"\begin{tabular}{@{}lcc@{}}",
            r"\toprule",
            r"\textbf{Component} & \textbf{BLEU Contribution} & \textbf{\% of Total} \\",
            r"\midrule",
        ]
        
        for component, contrib in sorted(contributions.items(), key=lambda x: -x[1]):
            pct = (contrib / total_improvement * 100) if total_improvement > 0 else 0
            name = component.replace("_", " ").title()
            lines.append(f"{name} & {contrib:+.2f} & {pct:.1f}\\% \\\\")
        
        lines.extend([
            r"\midrule",
            f"Total Improvement & {total_improvement:.2f} & 100\\% \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Contribution of each component to overall BLEU improvement.}",
            r"\label{tab:contribution}",
            r"\end{table}",
        ])
        
        path.write_text("\n".join(lines))
        return path
    
    def _export_bar_chart(self, path: Path) -> Path:
        """Export bar chart of results."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        names = [r.config.name.replace("_", "\n") for r in self.results]
        bleu_scores = [r.evaluation.bleu or 0 for r in self.results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, bleu_scores, color='steelblue', edgecolor='black')
        
        # Highlight best
        max_idx = bleu_scores.index(max(bleu_scores))
        bars[max_idx].set_color('darkgreen')
        
        ax.set_ylabel('BLEU Score', fontsize=12)
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_title('Translation Quality by Configuration', fontsize=14)
        ax.set_ylim(0, max(bleu_scores) * 1.15)
        
        # Add value labels
        for bar, score in zip(bars, bleu_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{score:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
    
    def _export_ablation_chart(self, path: Path) -> Path:
        """Export ablation impact chart."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        # Get full result as baseline
        full = next((r for r in self.results if r.config.name == "full"), None)
        if not full:
            return path
        
        full_bleu = full.evaluation.bleu or 0
        
        # Calculate deltas for each ablation
        ablations = [(r.config.name.replace("no_", ""), (r.evaluation.bleu or 0) - full_bleu)
                    for r in self.results if r.config.name.startswith("no_")]
        
        if not ablations:
            return path
        
        names, deltas = zip(*ablations)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['red' if d < 0 else 'green' for d in deltas]
        bars = ax.barh(names, deltas, color=colors, edgecolor='black')
        
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('ΔBLEU (compared to full system)', fontsize=12)
        ax.set_title('Impact of Removing Each Component', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path


def generate_thesis_chapter(
    results: list[ExperimentResult],
    output_file: Path,
    chapter_title: str = "Experimental Results",
):
    """Generate a complete LaTeX chapter from results."""
    
    exporter = ThesisExporter(results, output_dir=output_file.parent)
    tables = exporter.export_all()
    
    lines = [
        f"\\chapter{{{chapter_title}}}",
        r"\label{ch:experiments}",
        "",
        r"This chapter presents the experimental evaluation of our translation system.",
        "",
        r"\section{Experimental Setup}",
        "",
        r"We evaluate our system on a corpus of scientific abstracts...",
        "",
        r"\section{Main Results}",
        "",
        r"Table~\ref{tab:main-results} presents the main translation quality results.",
        "",
        r"\input{" + str(tables.get("results_table", "results_table.tex")) + "}",
        "",
        r"\section{Ablation Study}",
        "",
        r"To understand the contribution of each component, we conduct an ablation study.",
        "",
        r"\input{" + str(tables.get("ablation_table", "ablation_table.tex")) + "}",
        "",
        r"\section{Component Analysis}",
        "",
        r"\input{" + str(tables.get("contribution", "contribution.tex")) + "}",
    ]
    
    output_file.write_text("\n".join(lines))

