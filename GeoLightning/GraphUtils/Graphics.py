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
- GeoPandas
- GeoPy
- Shapely
- pyproj
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import box, LineString, Point
from scipy.interpolate import interp1d
import scienceplots
import geopandas as gpd
from geopy.point import Point as GeoPoint
from pyproj import CRS



def make_histogram_graph(hist: np.ndarray,
                         bin_edges: np.ndarray,
                         quantile_90: np.float64,
                         xlabel: str,
                         ylabel: str,
                         xlimit: np.float64,
                         b_size: np.float64,
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

    xlimit: np.float64
        where the x axis should stop in the graph

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

    # plt.style.use(['science', 'ieee'])

    #: histograma

    #: func_interp = UnivariateSpline(bin_edges, hist)
    wbin_edges = list()
    for j in range(np.size(bin_edges) - 1):
        wbin_edges.append(bin_edges[j] + (bin_edges[j + 1] - bin_edges[j]) / 2)
    bin_edges = wbin_edges
    func_interp = interp1d(bin_edges, hist, 'cubic')

    rv_x_confidence_interval = np.linspace(min(bin_edges),
                                         quantile_90, len(bin_edges) * 100)
    rv_x = np.linspace(min(bin_edges), max(bin_edges), len(bin_edges) * 100)
    rv_values = func_interp(rv_x)
    rv_values_confidence_interval = func_interp(rv_x_confidence_interval)

    plt.bar(wbin_edges,
            hist,
            color="#4e8ef5",
            edgecolor="#000000",
            linewidth=0,
            width=b_size,
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
                     facecolor="#b6d1fd",
                     label="90\% interval",
                     alpha=0.5)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.xlim(0, xlimit)

    #: plt.title(title)

    plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)

    plt.savefig(filename , dpi=600)



def generate_nde_report(sensors: np.ndarray,
                        nde: np.ndarray,
                        lightning_area: np.ndarray,
                        report_filename: str,
                        cities_df: gpd.GeoDataFrame,
                        states_df: gpd.GeoDataFrame,
                        title: str,
                        sensors_colors: str = "#fc8505",
                        de_multipliers: np.float64 = 0.95) -> None:
    """
    Generate a geographic report of the Network Detection Efficiency (NDE).

    This function creates a choropleth map representing the spatial distribution of detection 
    efficiency over a lightning-prone region. It visualizes sensors, state and municipal borders, 
    and highlights the effective detection region where the NDE exceeds a given threshold (e.g., 95% of max).

    Parameters
    ----------
    sensors : np.ndarray
        Array of sensor coordinates in the format [[lon, lat], ...] in decimal degrees.
    nde : np.ndarray
        Normalized detection efficiency values (from 0 to 1), corresponding to `lightning_area` points.
    lightning_area : np.ndarray
        Array of coordinates [[lon, lat], ...] where the NDE values are defined.
    report_filename : str
        Output filename (including path) for the generated PNG image.
    cities_df : gpd.GeoDataframe
        cities map GeoDataframe.
    states_df : gpd.GeoDataframe
        states map GeoDataframe  
    title : str
        Title of the plot.
    sensors_colors : str, optional
        Color of the sensor markers in the plot (default: "#fc8505").
    de_multipliers : float, optional
        Relative threshold (0–1) to define the region of effective detection. 
        Only areas with DE ≥ `de_multipliers * max(DE)` will be enclosed (default: 0.95).

    Returns
    -------
    None
        The function saves the figure directly to `report_filename` as a high-resolution PNG.

    Notes
    -----
    - All spatial data is reprojected to EPSG:4326 (WGS 84).
    - The function computes the convex hull of the effective detection region (DE ≥ threshold).
    - Useful for validating coverage of lightning sensor networks and identifying blind spots.

    """

    de_geopoints = [Point(x[1], x[0]) for x in lightning_area]
    de_geo = gpd.GeoDataFrame(geometry=de_geopoints)
    de_geo.set_crs(epsg=4326, inplace=True)
    sensors_geopoints = [Point(x[1], x[0]) for x in sensors]
    sensors_geo = gpd.GeoDataFrame(geometry=sensors_geopoints)
    sensors_geo.set_crs(epsg=4326, inplace=True)

    plt.close('all')

    de_index = 100 * nde
    de_geo["de_index"] = de_index
    max_de_index = np.max(de_index)
    de_area = de_geo.loc[de_geo["de_index"] >= de_multipliers * max_de_index]
    de_area = gpd.GeoDataFrame(geometry=[de_area.unary_union.convex_hull])

    fig, ax = plt.subplots(figsize=(10, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    de_geo.plot(column="de_index",
                ax=ax, 
                cax=cax,
                legend=True, 
                cmap='Blues')
    
    states_df.boundary.plot(ax=ax, 
                            color="#000000",
                            linewidth=0.5, 
                            markersize=0.0, 
                            alpha=0.5)

    sensors_geo.plot(ax=ax, 
                     color=sensors_colors, 
                     markersize=10, 
                     label="Sensores")

    de_area.boundary.plot(ax=ax, color="yellow",
                          label="Limite de Detecção Efetiva $ (DE > 95%)$")
    cities_df.plot(ax=ax, 
                   color="#6a9cb0",
                   linewidth=0.5, 
                   markersize=0.5, 
                   alpha=0.9,
                   label="Linhas de Transmissão")
    
    norm = Normalize(vmin=0, vmax=100)
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("$DE$ $(%)$")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc='best')
    ax.set_title(title)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels() +
                 cax.get_xticklabels() + cax.get_yticklabels()):
        item.set_fontsize(12.0)
    plt.tight_layout()
    plt.savefig(report_filename, dpi=600)


def geo_countour_plot(x_vals: np.ndarray,
                      y_vals: np.ndarray,
                      z_vals: np.ndarray,
                      title: str,
                      mapa_base: gpd.GeoDataFrame,
                      sensor_map: gpd.GeoDataFrame,
                      output_file: str,
                      levels: int = 10,
                      sensors_colors: str = "#fc8505") -> None:
    """
    Save a contour plot representing the mean error landscape of a bivariate algorithm.

    This function generates and saves a level curve (contour) plot from a grid of error values 
    defined over two parameters or spatial dimensions. It is typically used to visualize the 
    performance surface of optimization algorithms or estimation methods, where each point 
    corresponds to the average error associated with a pair (x₁, x₂).

    Parameters
    ----------
    x_vals : np.ndarray
        Longitude or easting (1D array).
    y_vals : np.ndarray
        Latitude or northing (1D array).
    z_vals : np.ndarray
        2D array of scalar values with shape (len(y_vals), len(x_vals)).
    title : str
        Title of the plot.
    mapa_base : gpd.GeoDataFrame
        Base map to overlay the contours (e.g., municipalities or states).    
    sensor_map : gpd.GeoDataFrame
        Sensor's map
    output_file : str
        Path to save the final figure.
    levels : int
        Number of contour levels (default: 10).
    sensors_colors : str, optional
        Color of the sensor markers in the plot (default: "#fc8505").

    Returns
    -------
    None
        The function produces a `.png` file containing the contour plot.

    Notes
    -----
    - The color gradient and contour lines indicate error magnitude: darker regions correspond
      to higher error values.
    - It is recommended to verify the shape of `z_vals` before passing to this function. Use 
      `z_vals.shape == (len(y_vals), len(x_vals))`.

    """

    # Gerar malha
    X, Y = np.meshgrid(x_vals, y_vals)

    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotar mapa base
    mapa_base.plot(ax=ax,
                   color='whitesmoke',
                   edgecolor='gray')

    # Plotar Sensores
    sensor_map.plot(ax=ax,
                    color=sensors_colors,
                    markersize=10,
                    label="Sensores")

    # Gerar curvas de nível
    cs = plt.contour(X, Y, z_vals, levels=levels, cmap='viridis')

    # Extrair contornos como LineStrings
    geoms = []
    valores = []
    for i, collection in enumerate(cs.collections):
        for path in collection.get_paths():
            coords = path.vertices
            if coords.shape[0] > 1:
                line = LineString(coords)
                geoms.append(line)
                valores.append(cs.levels[i])

    # Criar GeoDataFrame das curvas
    gdf_contornos = gpd.GeoDataFrame(
        {'nivel': valores, 'geometry': geoms}, crs='EPSG:4326')
    gdf_contornos.plot(ax=ax, column='nivel', cmap='viridis',
                       linewidth=1.5, legend=True)

    # Título e eixos
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close(fig)


def coverage_plot(mapa: gpd.GeoDataFrame,
                  sensores: np.ndarray,
                  pontos_malha: np.ndarray,
                  raio: float = 120000,
                  epsg_utm: int = 32722,
                  salvar_em: str = "data/area_cobertura.png") -> None:
    """
    Plots the sensor distribution and coverage area on a geographic map (EPSG:4326),
    converting coordinates from UTM.

    Parameters
    ----------
    mapa : gpd.GeoDataFrame
        Base map in EPSG:4326 (e.g., states or municipalities).
    sensores : np.ndarray
        Array of sensor positions in UTM (shape: [n, 2]).
    pontos_malha : np.ndarray
        Grid points to be plotted (also in UTM).
    raio : float
        Radius of coverage in meters (default: 120 km).
    epsg_utm : int
        EPSG code of the UTM coordinate system (default: 32722 = WGS 84 / UTM zone 22S).
    salvar_em : str
        Path to save the resulting figure.
    """
    # Criar GeoDataFrames a partir de arrays UTM
    gdf_sensores = gpd.GeoDataFrame(
        geometry=[Point(xy) for xy in sensores],
        crs=CRS.from_epsg(epsg_utm)
    ).to_crs(epsg=4326)

    gdf_malha = gpd.GeoDataFrame(
        geometry=[Point(xy) for xy in pontos_malha],
        crs=CRS.from_epsg(epsg_utm)
    ).to_crs(epsg=4326)

    # Extrair as coordenadas reprojetadas para plotagem
    sensores_ll = np.array([[p.x, p.y] for p in gdf_sensores.geometry])
    malha_ll = np.array([[p.x, p.y] for p in gdf_malha.geometry])

    # Plotar mapa base
    fig, ax = plt.subplots(figsize=(10, 10))
    mapa.plot(ax=ax, color="lightgray", edgecolor="black")

    # Adicionar círculos de cobertura reprojetados (em graus)
    for point in gdf_sensores.geometry:
        circ = plt.Circle((point.x, point.y), raio / 111320,  # Aproximação: 1 grau ≈ 111.32 km
                          color='red', alpha=0.2)
        ax.add_patch(circ)

    # Plotar sensores e malha
    ax.scatter(sensores_ll[:, 0], sensores_ll[:, 1],
               c="red", edgecolors="black", label="Sensores")
    ax.scatter(malha_ll[:, 0], malha_ll[:, 1],
               c="blue", s=5, alpha=0.4, label="Malha")

    ax.set_title(
        f"Distribuição de Sensores com Cobertura Total ({len(sensores)} sensores)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(salvar_em, dpi=600)
    plt.close(fig)
