import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from math import pi
from typing import Dict, List, Optional, Any


class ScientificVisualizer:
    """
    Scientific visualizer for CaDIU: radar charts, Pareto plots, and workflow diagrams.
    """
    
    def __init__(self, style: str = "whitegrid", palette: str = "husl"):
        """Initialize visualizer with publication-ready style."""
        sns.set(style=style)
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
        self.palette = sns.color_palette(palette)

    def plot_radar_charts(
        self,
        results: Dict[str, Dict[str, float]],
        datasets: List[str] = ["VisA", "MVTec-AD", "Real-IAD", "BTAD"],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot radar charts for SOTA comparison across datasets.
        
        Args:
            results: Dict[method][metric] -> value
            datasets: List of dataset names
            save_path: Path to save figure (PNG)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(polar=True))
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            
            # Select metrics
            if dataset in ["VisA", "MVTec-AD"]:
                apf = [results[method].get(f"{dataset}_APF", 0) for method in results]
                priv = [results[method].get(f"{dataset}_IRE", 1) for method in results]
                ylabel = "IRE ↓"
            else:
                apf = [results[method].get(f"{dataset.replace('-','')}_APF", 0) for method in results]
                priv = [results[method].get(f"{dataset.replace('-','')}_CLS", 1) for method in results]
                ylabel = "CLS ↓"
            
            mer = [results[method].get("MER", 1) for method in results]
            methods = list(results.keys())

            # Normalize (higher = better)
            apf_norm = np.array(apf)
            priv_norm = 1 - np.clip(np.array(priv) / max(priv), 0, 1)
            mer_norm = 1 - np.clip(np.array(mer) / max(mer), 0, 1)

            # Radar setup
            categories = ['APF ↑', ylabel, 'MER ↓']
            N = len(categories)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            # Plot each method
            for i, method in enumerate(methods):
                values = [apf_norm[i], priv_norm[i], mer_norm[i]]
                values += values[:1]
                ax.plot(angles, values, linewidth=2, label=method, color=self.palette[i])
                ax.fill(angles, values, color=self.palette[i], alpha=0.1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1.1)
            ax.set_title(dataset, size=14, weight='bold', pad=20)

        # Legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=4)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_pareto_frontier(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot 3D Pareto frontier for the CLPU trilemma.
        
        Args:
            results: Dict[method][metric] -> value
            save_path: Path to save figure (HTML for interactive)
        """
        methods = list(results.keys())
        avg_apf = [results[m]["Avg_APF"] for m in methods]
        avg_priv = [results[m]["Avg_PRIV"] for m in methods]
        mer = [results[m]["MER"] for m in methods]

        # Normalize
        priv_norm = np.array(avg_priv) / max(avg_priv)
        mer_norm = np.array(mer) / max(mer)

        # Create 3D scatter
        fig = go.Figure(data=[go.Scatter3d(
            x=avg_apf,
            y=1 - priv_norm,
            z=1 - mer_norm,
            mode='markers+text',
            marker=dict(
                size=8,
                color=avg_apf,
                colorscale='Viridis',
                showscale=True
            ),
            text=methods,
            textposition="top center"
        )])

        fig.update_layout(
            title="CLPU Trilemma: Transfer vs. Privacy vs. Memory",
            scene=dict(
                xaxis_title='APF ↑',
                yaxis_title='Privacy ↑',
                zaxis_title='Memory Efficiency ↑'
            ),
            width=800,
            height=600
        )

        if save_path:
            fig.write_html(save_path)
        fig.show()

    def generate_workflow_tikz(
        self,
        save_path: str = "workflow_cadiu.tex"
    ) -> None:
        """
        Generate TikZ code for CaDIU workflow diagram.
        
        Args:
            save_path: Path to save TikZ file
        """
        tikz_code = r"""
\documentclass{standalone}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shadows, fit, backgrounds, calc}

% Color definitions
\definecolor{inputcolor}{rgb}{0.95, 0.9, 0.95}
\definecolor{daecolor}{rgb}{0.85, 0.95, 0.85}
\definecolor{dpgcolor}{rgb}{0.75, 0.95, 0.95}
\definecolor{ssmcolor}{rgb}{0.95, 0.85, 0.95}
\definecolor{unlearncolor}{rgb}{0.95, 0.95, 0.85}
\definecolor{losscolor}{rgb}{0.85, 0.85, 0.95}

\tikzset{
    block/.style = {
        rectangle, draw=black, fill=#1, rounded corners, minimum height=1cm, align=center, font=\footnotesize\sffamily, drop shadow
    },
    imgblock/.style = {
        rectangle, draw=none, minimum width=2.5cm, minimum height=2cm, align=center
    },
    arrow/.style = {->, >=Latex, thick, draw=black, rounded corners}
}

\begin{document}
\begin{tikzpicture}[node distance=1.2cm and 1.4cm, every node/.style={align=center}]
    
    % ========== INPUT IMAGE ==========
    \node[imgblock] (input) {\includegraphics[width=2.4cm]{input_image}};
    \node[below=0.0cm of input, font=\footnotesize] {Industrial Image \\ $x \in \mathcal{X}$};
    
    % ========== DAE ==========
    \node[block=daecolor, right=of input, xshift=-0.5cm, yshift=-1.49cm, rotate=90] (dae) {ViT-Base};
    \draw[arrow] (input) -- (dae);
    
    \node[block=daecolor, right=0.6cm and 1.2cm of dae] (prim) {Primitive Branch \\ $z_{\text{prim}}$};
    \node[block=daecolor, below right=1.86cm and 0.6cm of dae] (sem) {Semantic Branch \\ $z_{\text{sem}}$};
    
    \draw[arrow] (dae.south) -- ++(0.2,0) |- (prim.west);
    \draw[arrow] (dae.south) -- ++(0.2,0) |- (sem.west);
    
    % ========== DPG ==========
    \node[block=dpgcolor, below=2.0cm of dae, rotate=90] (dpg) {DPG Manager};
    \draw[arrow] (dae) -- (dpg);
    
    % ========== SSM ==========
    \node[block=ssmcolor, below=2.0cm of dpg, rotate=90] (ssm) {SSM};
    \draw[arrow] (dpg) -- (ssm);
    
    % ========== UNLEARNING ==========
    \node[block=unlearncolor, below=2.0cm of ssm, rotate=90] (unlearn) {Unlearning Protocol};
    \draw[arrow] (ssm) -- (unlearn);
    
    % ========== GROUPING ==========
    \begin{pgfonlayer}{background}
        \node[draw=black, dash dot, rounded corners, inner sep=8pt,
        fit=(input) (dae) (prim) (sem),
        label={[xshift=-2cm, yshift=-0.5cm]:{\textbf{DAE}}}] (daebox) {};
        
        \node[draw=black, dash dot, rounded corners, inner sep=8pt,
        fit=(dpg),
        label={[xshift=-2cm, yshift=-0.5cm]:{\textbf{DPG}}}] (dpgbox) {};
        
        \node[draw=black, dash dot, rounded corners, inner sep=8pt,
        fit=(ssm),
        label={[xshift=-2cm, yshift=-0.5cm]:{\textbf{SSM}}}] (ssmbox) {};
        
        \node[draw=black, dash dot, rounded corners, inner sep=8pt,
        fit=(unlearn),
        label={[xshift=-2cm, yshift=-0.5cm]:{\textbf{Unlearning}}}] (unlearnbox) {};
    \end{pgfonlayer}
    
\end{tikzpicture}
\end{document}
"""
        with open(save_path, 'w') as f:
            f.write(tikz_code)
        print(f"TikZ workflow saved to {save_path}")

    def plot_ablation_study(
        self,
        ablation_data: Dict[str, List[float]],
        param_name: str,
        metric_names: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ablation study for hyperparameter sensitivity.
        
        Args:
            ablation_data: Dict[param_value][metric] -> value
            param_name: Name of hyperparameter (e.g., "lambda")
            metric_names: List of metric names to plot
            save_path: Path to save figure
        """
        param_values = list(ablation_data.keys())
        metrics = {m: [] for m in metric_names}
        
        for val in param_values:
            for m in metric_names:
                metrics[m].append(ablation_data[val][m])

        plt.figure(figsize=(10, 6))
        for metric in metric_names:
            plt.plot(param_values, metrics[metric], 'o-', label=metric, linewidth=2.5)
        
        plt.xlabel(f"{param_name} (Hyperparameter)")
        plt.ylabel("Metric Value")
        plt.title(f"Ablation Study: {param_name} Sensitivity")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()