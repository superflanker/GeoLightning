"""
Performance Metrics
===================

This module defines the performance metrics used in atmospheric discharge geolocation experiments.

Summary
-------
The functions implemented in this module are designed to evaluate the 
accuracy and consistency of geolocation algorithms applied to atmospheric 
discharges (e.g., lightning events). It supports metrics such as 
positioning error, timing residuals, likelihood values, and statistical 
criteria for evaluating candidate solutions.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- Metric evaluation functions for estimated positions and times.
- Support for spatial and temporal residuals.
- Adapted for use with likelihood-based geolocation pipelines.

Notes
-----
This module is part of the activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, 
Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numpy.linalg.inv
- numba
- GeoLightning.Utils.Constants (SIGMA_D, SIGMA_T)
"""


import numpy as np
from numpy.linalg import inv
from numba import jit
from typing import Tuple
from GeoLightning.Utils.Constants import SIGMA_D, SIGMA_T, AVG_EARTH_RADIUS, AVG_LIGHT_SPEED
from GeoLightning.Utils.Utils import computa_distancias

@jit(nopython=True, cache=True, fastmath=True)
def rmse(deltas: np.ndarray) -> np.float64:
    """
    Calculates the root mean square error (RMSE) between estimated and real values.

    A fundamental parameter for evaluating the accuracy of event location algorithms, the RMSE provides a measure of the average discrepancy between the estimated and real position vectors, and is particularly useful in quantifying spatial accuracy.

    Parameters
    ----------
    deltas : np.ndarray
        Vector of measurement deltas
    real : np.ndarray
        Vector of real positions

    Returns
    -------
    np.float64
        Value of the root mean square error (RMSE) between the given vectors.
    """
    return np.sqrt(np.mean((deltas) ** 2))


@jit(nopython=True, cache=True, fastmath=True)
def mae(deltas: np.ndarray) -> np.float64:
    """
    Calculates the mean absolute error (MAE) between the estimated values ​​and the true values.

    MAE is a widely used metric in the evaluation of prediction models, providing a direct measure of the average magnitude of the errors, without considering their direction.

    Parameters
    ----------
    deltas : np.ndarray
        Vector of estimated values, representing the model predictions.

    Returns
    -------
    np.float64
        Value of the mean absolute error (MAE) between the given vectors.
    """
    return np.mean(np.abs(deltas))


@jit(nopython=True, cache=True, fastmath=True)
def average_mean_squared_error(deltas: np.ndarray) -> np.float64:
    """
    Calculates the mean squared error (AMSE) between estimated values ​​and real values.

    AMSE (Average Mean Squared Error) is a metric that quantifies the average of the squares of the differences between estimated values ​​and real values, penalizing larger errors with greater severity.

    Parameters
    ----------
    deltas : np.ndarray
        Vector of measurement deltas

    Returns
    -------
    np.float64
        Value of the mean squared error (AMSE) between the provided vectors.
    """
    return np.mean((deltas) ** 2)


@jit(nopython=True, cache=True, fastmath=True)
def mean_location_error(deltas: np.ndarray) -> np.float64:
    """
    Calculates the mean localization error (MLE) between the estimated values ​​and the real values.

    The MLE (Mean Localization Error) is defined as the average of the Euclidean distances between pairs of estimated and real positions, and is widely used in geolocation problems to quantify spatial accuracy.

    Parameters
    ----------
    deltas : np.ndarray
        Vector of measurement deltas

    Returns
    -------
    np.float64
        Value of the mean localization error (MLE), expressed in the same unit as the given spatial coordinates.
    """

    return np.mean(deltas)


@jit(nopython=True, cache=True, fastmath=True)
def calcula_prmse(rmse: float,
                  referencia: float) -> float:
    """
    Calculates the percentage root mean square error (PRMSE).

    PRMSE (Percentage Root Mean Square Error) is defined as the percentage ratio between the RMSE value and a reference value (usually the full scale), and is useful for relative performance analysis.

    Parameters
    ----------
    rmse : float
        Absolute value of the root mean square error (RMSE).
    reference : float
        Full scale value or reference adopted for normalization.

    Returns
    -------
    float
        Value of the root mean square error expressed as a percentage (PRMSE), given by: `(rmse / reference) * 100`.
    """

    return 100.0 * rmse / referencia


@jit(nopython=True, cache=True, fastmath=True)
def acuracia_associacao(associacoes_estimadas: np.ndarray,
                        associacoes_reais: np.ndarray) -> np.float64:
    """
    Calculates the accuracy of the association between detections and events.

    The accuracy of the association is defined as the proportion of correct associations between the estimated and real indices, expressing the effectiveness of the algorithm in the task of identifying corresponding events.

    Parameters
    ----------
    estimated_associations : np.ndarray
        One-dimensional vector containing the estimated indices of the associations for each detection.
    real_associations : np.ndarray
        One-dimensional vector containing the real indices of the corresponding associations.

    Returns
    -------
    np.float64
    Value of the accuracy of the association, defined as the ratio between the number of correct associations and the total number of detections.
    """
    return np.mean(associacoes_estimadas == associacoes_reais)


@jit(nopython=True, cache=True, fastmath=True)
def erro_relativo_funcao_ajuste(F_estimado: float,
                                F_referencia: float) -> float:
    """
    Calculates the percentage relative error between the estimated fitting function and a reference.

    The percentage relative error is defined as the percentage difference between the estimated value of the fitting function and a reference value, the latter usually being a benchmark, known optimum value or ideal solution.

    Parameters
    ----------
    F_estimado : float
        Value of the fitting function obtained by the algorithm.
    F_referencia : float
        Reference value used for comparison, such as the theoretical optimum or a benchmark.

    Returns
    -------
    float
    Percentage relative error between the given values, calculated as
    ``100 * abs(F_estimado - F_referencia) / abs(F_referencia)``.
    """

    return np.abs(F_estimado - F_referencia) / np.abs(F_referencia) * 100.0


@jit(nopython=True, cache=True, fastmath=True)
def tempo_execucao(tempo_inicial: float,
                   tempo_final: float) -> float:
    """
    Calculates the total execution time between two time instants.

    This method computes the difference between the end time and the start time,
    returning the total elapsed time in seconds.

    Parameters
    ----------
    start_time : float
        Timestamp of the start of the execution, expressed in seconds.
    end_time : float
        Timestamp of the end of the execution, expressed in seconds.

    Returns
    -------
    float
        Total execution time, in seconds, calculated as the difference
        between `end_time` and `start_time`.
    """
    return tempo_final - tempo_inicial


@jit(nopython=True, cache=True, fastmath=True)
def _inv_3x3(A: np.ndarray, eps_det: np.float64 = 1e-18) -> Tuple[np.ndarray, np.float64]:
    """
    Invert a 3x3 matrix using an explicit adjugate/determinant formula.

    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (3, 3).

    eps_det : np.float64, optional
        Determinant magnitude threshold used to declare near-singularity.

    Returns
    -------
    A_inv : np.ndarray
        Inverse matrix of shape (3, 3). If singular/near-singular, all entries
        are set to ``np.inf``.

    det : np.float64
        Determinant of ``A``.
    """
    a = A[0, 0]
    b = A[0, 1]
    c = A[0, 2]
    d = A[1, 0]
    e = A[1, 1]
    f = A[1, 2]
    g = A[2, 0]
    h = A[2, 1]
    i = A[2, 2]

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    A_inv = np.empty((3, 3), dtype=np.float64)

    if np.abs(det) < eps_det:
        # Near-singular: CRLB not meaningful / unbounded under this geometry.
        for r in range(3):
            for cc in range(3):
                A_inv[r, cc] = np.inf
        return A_inv, det

    inv_det = 1.0 / det

    # Adjugate (cofactor transpose)
    A_inv[0, 0] = (e * i - f * h) * inv_det
    A_inv[0, 1] = (c * h - b * i) * inv_det
    A_inv[0, 2] = (b * f - c * e) * inv_det

    A_inv[1, 0] = (f * g - d * i) * inv_det
    A_inv[1, 1] = (a * i - c * g) * inv_det
    A_inv[1, 2] = (c * d - a * f) * inv_det

    A_inv[2, 0] = (d * h - e * g) * inv_det
    A_inv[2, 1] = (b * g - a * h) * inv_det
    A_inv[2, 2] = (a * e - b * d) * inv_det

    return A_inv, det


@jit(nopython=True, cache=True, fastmath=True)
def fim_crlb_latlon_t0(solucao_deg: np.ndarray,
                       pontos_de_chegada: np.ndarray,
                       sigma_t: np.float64 = SIGMA_T,
                       step_m: np.float64 = 1.0,
                       sistema_cartesiano: bool = False) -> Tuple[np.ndarray,
                                                                  np.ndarray,
                                                                  np.ndarray]:
    """
    Compute the Fisher Information Matrix (FIM) and CRLB for (lat, lon, t0).

    This routine evaluates the Cramér–Rao Lower Bound (CRLB) for a TOA model in
    a 2D geodetic parameterization (latitude/longitude) with unknown origin time.
    The parameter vector is:

        theta = [lat, lon, t0]^T,

    where lat/lon are expressed in degrees and t0 in seconds.

    The measurement model is assumed to be:

        t_i = t0 + d_i(lat, lon) / c + eps_i,

    with i.i.d. zero-mean Gaussian noise:

        eps_i ~ N(0, sigma_t^2),

    where c is ``AVG_LIGHT_SPEED`` and d_i is the propagation distance (meters)
    from the event location to sensor i (computed by the selected distance model).

    The Jacobian of the mean model mu_i(theta) = t0 + d_i/c is computed by forward
    finite differences in latitude and longitude:

        ∂mu_i/∂lat ≈ (d_i(lat+Δlat, lon) - d_i(lat, lon)) / (c * Δlat),
        ∂mu_i/∂lon ≈ (d_i(lat, lon+Δlon) - d_i(lat, lon)) / (c * Δlon),
        ∂mu_i/∂t0  = 1.

    The FIM is then:

        I(theta) = (1/sigma_t^2) * J^T J,

    and the CRLB covariance is:

        Cov(theta) = I(theta)^{-1}.

    Additionally, the routine returns the 2x2 covariance matrix in the local
    tangent plane (North-East, meters) obtained by transforming the lat/lon
    covariance to radians and then applying:

        dN ≈ R * dphi,      dE ≈ R * cos(phi) * dlambda,

    where R is ``AVG_EARTH_RADIUS`` and phi is the event latitude in radians.

    Parameters
    ----------
    solucao_deg : np.ndarray
        Candidate event location in spherical/geodetic form
        ``[latitude, longitude, altitude]`` (degrees, degrees, meters).
        Only latitude and longitude are considered in the CRLB state; altitude is
        treated as fixed.

    pontos_de_chegada : np.ndarray
        Sensor coordinates aligned with the model distances. Each row must be a
        sensor position. In spherical mode, rows must follow
        ``[latitude, longitude, altitude]`` (degrees, degrees, meters). In Cartesian
        mode, rows must be ``[x, y, z]`` (meters), but note that lat/lon finite
        differences are not meaningful under a Cartesian-only state parameterization.

    sigma_t : np.float64, optional
        Standard deviation of the TOA noise in seconds. Must be strictly positive.

    step_m : np.float64, optional
        Finite-difference step expressed as an approximate arc-length in meters.
        The corresponding angular step in degrees is computed as:
        ``Δdeg = (step_m / AVG_EARTH_RADIUS) * (180/pi)``.

    sistema_cartesiano : bool, optional
        If True, uses Cartesian distances inside `computa_distancias`. If False,
        uses spherical distances. Default is False.

    Returns
    -------
    fim : np.ndarray
        Fisher Information Matrix of shape (3, 3) in the parameter units
        (degrees, degrees, seconds).

    crlb : np.ndarray
        CRLB covariance matrix of shape (3, 3) for ``[lat_deg, lon_deg, t0_s]``.
        If the FIM is near-singular, entries may be ``np.inf``.

    cov_ne : np.ndarray
        2x2 covariance matrix in the local tangent plane (North-East), in meters^2,
        corresponding to the (lat, lon) sub-block of the CRLB after conversion to
        radians and then to (N, E).

    Notes
    -----
    - This function requires a non-degenerate sensor geometry for the FIM to be
      invertible. If the determinant is near zero, the CRLB is unbounded.
    - The output CRLB is expressed in the units of the chosen state:
      (degrees, degrees, seconds). Use `cov_ne` for a metric interpretation in meters.
    """
    # Finite-difference step in degrees corresponding to step_m meters
    step_deg = (step_m / AVG_EARTH_RADIUS) * (180.0 / np.pi)
    dlat = step_deg
    dlon = step_deg

    # Distances at the nominal solution
    d0 = computa_distancias(origem=solucao_deg,
                            destinos=pontos_de_chegada,
                            sistema_cartesiano=sistema_cartesiano)

    # Distances under latitude perturbation
    x_lat = solucao_deg.copy()
    x_lat[0] = x_lat[0] + dlat
    d_lat = computa_distancias(origem=x_lat,
                               destinos=pontos_de_chegada,
                               sistema_cartesiano=sistema_cartesiano)

    # Distances under longitude perturbation
    x_lon = solucao_deg.copy()
    x_lon[1] = x_lon[1] + dlon
    d_lon = computa_distancias(origem=x_lon,
                               destinos=pontos_de_chegada,
                               sistema_cartesiano=sistema_cartesiano)

    n = pontos_de_chegada.shape[0]

    # Build J (N x 3): [dmu/dlat, dmu/dlon, dmu/dt0]
    # Units: seconds/degree, seconds/degree, dimensionless
    J = np.empty((n, 3), dtype=np.float64)
    inv_c = 1.0 / AVG_LIGHT_SPEED
    inv_dlat = 1.0 / dlat
    inv_dlon = 1.0 / dlon

    for i in range(n):
        dd_dlat = (d_lat[i] - d0[i]) * inv_dlat
        dd_dlon = (d_lon[i] - d0[i]) * inv_dlon
        J[i, 0] = dd_dlat * inv_c
        J[i, 1] = dd_dlon * inv_c
        J[i, 2] = 1.0

    # FIM = (1/sigma_t^2) * J^T J
    fim = np.zeros((3, 3), dtype=np.float64)
    inv_sigma2 = 1.0 / (sigma_t * sigma_t)

    for i in range(n):
        j0 = J[i, 0]
        j1 = J[i, 1]
        j2 = J[i, 2]
        fim[0, 0] += j0 * j0
        fim[0, 1] += j0 * j1
        fim[0, 2] += j0 * j2
        fim[1, 0] += j1 * j0
        fim[1, 1] += j1 * j1
        fim[1, 2] += j1 * j2
        fim[2, 0] += j2 * j0
        fim[2, 1] += j2 * j1
        fim[2, 2] += j2 * j2

    for r in range(3):
        for c in range(3):
            fim[r, c] *= inv_sigma2

    # CRLB = inv(FIM)
    crlb, _ = _inv_3x3(fim)

    # Convert (lat, lon) sub-block from degrees to radians
    deg2rad = np.pi / 180.0
    t00 = crlb[0, 0] * (deg2rad * deg2rad)
    t01 = crlb[0, 1] * (deg2rad * deg2rad)
    t11 = crlb[1, 1] * (deg2rad * deg2rad)

    # Local tangent-plane transform: [dN, dE]^T = A [dphi, dlambda]^T
    phi_rad = solucao_deg[0] * deg2rad
    R = AVG_EARTH_RADIUS
    cphi = np.cos(phi_rad)

    # cov_NE = A * cov_phiLambda * A^T, with A = diag(R, R*cos(phi))
    cov_ne = np.empty((2, 2), dtype=np.float64)
    cov_ne[0, 0] = (R * R) * t00
    cov_ne[0, 1] = (R * R * cphi) * t01
    cov_ne[1, 0] = cov_ne[0, 1]
    cov_ne[1, 1] = (R * R * cphi * cphi) * t11

    return fim, crlb, cov_ne


@jit(nopython=True, cache=True, fastmath=True)
def crlb_ne_summary(cov_ne: np.ndarray) -> Tuple[np.float64,
                                                 np.float64,
                                                 np.float64,
                                                 np.float64,
                                                 np.float64,
                                                 np.float64]:
    """
    Summarize a 2x2 CRLB covariance matrix in the local tangent plane (N, E).

    Given a covariance matrix in North-East coordinates (meters^2), this function
    returns marginal standard deviations, a scalar summary, and the parameters of
    the 1-sigma uncertainty ellipse.

    Parameters
    ----------
    cov_ne : np.ndarray
        Covariance matrix of shape (2, 2) in the local tangent plane (North-East),
        expressed in meters^2. The expected layout is:

            [[var_N, cov_NE],
             [cov_NE, var_E]].

    Returns
    -------
    sigma_n : np.float64
        Marginal standard deviation in the North direction (meters),
        ``sigma_n = sqrt(var_N)``.

    sigma_e : np.float64
        Marginal standard deviation in the East direction (meters),
        ``sigma_e = sqrt(var_E)``.

    sigma_ne : np.float64
        Scalar summary (meters) defined as ``sqrt(sigma_n^2 + sigma_e^2)``.
        This is a convenient single-number indicator of horizontal uncertainty.

    sigma_major : np.float64
        1-sigma semi-axis length (meters) of the major axis of the uncertainty ellipse,
        equal to ``sqrt(lambda_max)``, where ``lambda_max`` is the largest eigenvalue.

    sigma_minor : np.float64
        1-sigma semi-axis length (meters) of the minor axis of the uncertainty ellipse,
        equal to ``sqrt(lambda_min)``, where ``lambda_min`` is the smallest eigenvalue.

    angle_rad : np.float64
        Orientation angle (radians) of the major axis measured counterclockwise from
        the North axis. Computed as:

            angle = 0.5 * atan2(2*cov_NE, var_N - var_E).

    Notes
    -----
    - Small negative variances/eigenvalues due to floating-point roundoff are clamped
      to zero before taking square roots.
    - If the matrix contains ``np.inf`` entries (e.g., from a near-singular FIM),
      the returned values will propagate accordingly.
    """
    var_n = cov_ne[0, 0]
    var_e = cov_ne[1, 1]
    cov  = cov_ne[0, 1]

    # Clamp marginal variances for numerical safety
    if var_n < 0.0:
        var_n = 0.0
    if var_e < 0.0:
        var_e = 0.0

    sigma_n = np.sqrt(var_n)
    sigma_e = np.sqrt(var_e)
    sigma_ne = np.sqrt(var_n + var_e)

    # Eigenvalues of a symmetric 2x2 matrix (closed form)
    tr = var_n + var_e
    det = var_n * var_e - cov * cov

    disc = tr * tr - 4.0 * det
    if disc < 0.0:
        disc = 0.0
    sdisc = np.sqrt(disc)

    lam1 = 0.5 * (tr + sdisc)  # lambda_max
    lam2 = 0.5 * (tr - sdisc)  # lambda_min

    if lam1 < 0.0:
        lam1 = 0.0
    if lam2 < 0.0:
        lam2 = 0.0

    sigma_major = np.sqrt(lam1)
    sigma_minor = np.sqrt(lam2)

    # Orientation of the major axis (North-reference)
    angle_rad = 0.5 * np.arctan2(2.0 * cov, (var_n - var_e))

    return sigma_n, sigma_e, sigma_ne, sigma_major, sigma_minor, angle_rad

