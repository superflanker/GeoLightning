"""
Visualization Utility – Histogram Plotting with Interpolated PDF and Confidence Interval
----------------------------------------------------------------------------------------

Summary
-------
This module provides a high-resolution plotting utility tailored for visualizing 
probability distributions derived from localization error metrics or similar 
performance indicators.

The `make_histogram_graph` function renders a stylized histogram combined with a 
cubic spline interpolation of the probability density function (PDF). A shaded 
area denotes the confidence interval (typically 90%), enhancing interpretability 
in formal reports and scientific publications.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Intended Use
------------
This utility is designed to support the STELA evaluation pipeline by generating
consistent, high-quality graphical summaries of localization accuracy across test cases.

Contents
--------
- Publication-ready plotting style using the `scienceplots` IEEE theme.
- Automatic cubic interpolation via `scipy.interpolate.interp1d`.
- Configurable quantile-based confidence interval overlay.
- Output exported as a 600 DPI image suitable for inclusion in LaTeX documents.

Notes
-----
This module is part of the academic activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, 
Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- Matplotlib
- SciencePlots
- Scipy (interp1d)

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scienceplots


def make_histogram_graph(hist: np.ndarray,
                         bin_edges: np.ndarray,
                         quantile_0: np.float64,
                         quantile_90: np.float64,
                         xlabel: str,
                         ylabel: str,
                         filename: str):
    """
    Generates and saves a stylized histogram plot with smoothed interpolation 
    and a shaded confidence interval, suitable for publication-quality figures.

    This function uses a scientific plotting style (via `scienceplots`) to 
    render a histogram of localization errors or similar quantities. 
    A cubic interpolation is applied to produce a smoothed curve over 
    the histogram bins. Additionally, a confidence interval (typically 
    the central 90%) is highlighted for visual clarity.

    Parameters
    ----------
    hist : np.ndarray
        Histogram bin counts, typically computed from localization error data.

    bin_edges : np.ndarray
        Edges of the histogram bins. Must be one element longer than `hist`.

    quantile_0 : float
        Lower bound of the desired confidence interval (e.g., 0.05 for 5th percentile).

    quantile_90 : float
        Upper bound of the desired confidence interval (e.g., 0.95 for 95th percentile).

    xlabel: str
        name of x axis

    ylabel: str
        name of y axis

    filename : str
        Path to save the resulting figure. Saved with 600 DPI resolution.

    Notes
    -----
    - The interpolation is performed using `scipy.interpolate.interp1d` 
        with cubic splines for smoothing.
    - The histogram bars and the interpolated PDF are both rendered, along 
        with a shaded region representing the confidence interval.
    - Matplotlib parameters are adjusted for IEEE-compatible formatting 
        using the `scienceplots` style.
    """
    # configuração do matplotlib
    plt.style.use(['science'])

    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.titlesize': 8,
        'legend.loc': 'upper left',   # ou 'best', 'lower right', etc.
        'legend.frameon': False,
        'legend.handlelength': 2.0,
        'legend.borderaxespad': 0.4,
    })

    #: gráficos
    plt.close('all')

    #: histograma

    func_interp = interp1d(bin_edges, hist, 'cubic')

    rv_x_confidence_interval = np.arange(quantile_0,
                                         quantile_90, 0.001)
    rv_x = np.arange(min(bin_edges), max(bin_edges), 0.001)
    rv_values = func_interp(rv_x)
    rv_values_confidence_interval = func_interp(rv_x_confidence_interval)

    plt.bar(bin_edges,
            hist,
            color='#8a8a8a',
            edgecolor="#000000",
            linewidth=0,
            width=0.05,
            align='center',
            label="Histogram")

    plt.plot(rv_x,
             rv_values,
             linestyle="dashed",
             color='#000000',
             label="Distribution")

    plt.fill_between(rv_x_confidence_interval,
                     0,
                     rv_values_confidence_interval,
                     facecolor='#ababab',
                     label=r"$90\%$ Interval")

    # plt.xlabel("Location Error (m)")

    # plt.ylabel("Probability Density Function (PDF)")

    #: plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)

    plt.savefig(filename, dpi=600)

