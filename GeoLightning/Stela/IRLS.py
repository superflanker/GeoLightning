import numpy as np
from numba import jit
from typing import Tuple
from GeoLightning.Utils.Constants import SIGMA_T, AVG_EARTH_RADIUS, AVG_LIGHT_SPEED
from GeoLightning.Utils.Utils import computa_distancias
from GeoLightning.Stela.Common import \
    residuos_espaciais, \
    computa_pesos, \
    solver_2x2_simples


@jit(nopython=True, cache=True, fastmath=True)
def irls(solucao_inicial: np.ndarray,
         tempos_de_chegada: np.ndarray,
         pontos_de_chegada: np.ndarray,
         sigma_t: np.float64 = SIGMA_T,
         max_iter: np.int32 = 15,
         pesos_alg: str = "rbf",
         sistema_cartesiano: bool = False,
         k: np.float64 = 3.0,  # huber, tukey somente
         lm_mu0: np.float64 = 1e-3,
         use_central_diff: bool = True) -> Tuple[np.ndarray,
                                                 np.float64,
                                                 np.ndarray,
                                                 np.ndarray]:
    """
    Refine a TOA multilateration solution using IRLS with a weighted LM step.

    This routine performs a robust local refinement of an initial event location
    estimate using Iteratively Reweighted Least Squares (IRLS) combined with a
    Levenberg–Marquardt (LM) damped Gauss–Newton update. Residuals are expressed
    in spatial units (meters) by converting temporal errors through the assumed
    propagation speed ``AVG_LIGHT_SPEED``.

    At each iteration, the method:
    1) computes propagation distances from the current location hypothesis to all
       sensor positions;
    2) estimates the event origin time ``t0`` via a weighted back-projection;
    3) forms spatial residuals ``r_i = c (t_i - t0) - d_i``;
    4) updates per-observation weights from residuals using the selected weighting
       rule (RBF, Huber, or Tukey);
    5) computes a finite-difference Jacobian with respect to latitude and longitude
       (in degrees) and solves a 2x2 LM-normal system for the update step.

    The origin time ``t0`` is recomputed at each evaluation via a weighted average
    of individual origin-time estimates ``t_i - d_i / c``, yielding a variable
    projection approach where ``t0`` is not optimized by LM directly but updated
    in closed form.

    Parameters
    ----------
    solucao_inicial : np.ndarray
        Initial event location hypothesis. In spherical mode (``sistema_cartesiano=False``),
        it must be a vector ``[latitude, longitude, altitude]`` with angles in degrees
        and altitude in meters. In Cartesian mode (``sistema_cartesiano=True``), it must be
        ``[x, y, z]`` in meters. Only the first two components (latitude/longitude or x/y,
        depending on the chosen distance model) are updated by this implementation.

    tempos_de_chegada : np.ndarray
        One-dimensional array of absolute arrival times (seconds), one per sensor detection.

    pontos_de_chegada : np.ndarray
        Array of sensor coordinates aligned with ``tempos_de_chegada``. Each row must be a
        sensor position in the same coordinate system expected by the distance routine
        (geodetic degrees/meters or Cartesian meters).

    sigma_t : np.float64, optional
        Standard deviation of arrival-time jitter (seconds). This is converted into a
        spatial scale ``sigma_r = AVG_LIGHT_SPEED * sigma_t`` for weight computation.
        Default is ``SIGMA_T``.

    max_iter : np.int32, optional
        Maximum number of IRLS iterations. Default is 15.

    pesos_alg : str, optional
        Weighting rule identifier used by the IRLS outer loop. Supported values are
        ``"rbf"``, ``"huber"``, and ``"tukey"``. Unknown values fall back to ``"rbf"``.
        Default is ``"rbf"``.

    sistema_cartesiano : bool, optional
        If True, Cartesian distances are used; otherwise spherical distances are used.
        Default is False.

    k : np.float64, optional
        Dimensionless tuning constant for robust weighting rules (Huber/Tukey). Ignored by
        the RBF rule. Default is 3.0.

    lm_mu0 : np.float64, optional
        Initial LM damping parameter. The damping is decreased upon step acceptance and
        increased upon rejection. Default is 1e-3.

    Returns
    -------
    x : np.ndarray
        Refined event location estimate with the same shape as ``solucao_inicial``.
        The returned vector is updated in its first two components according to the LM step.

    t0 : np.float64
        Estimated event origin time (seconds) corresponding to the refined location.

    pesos : np.ndarray
        Final per-observation weights produced by the selected weighting rule, aligned with
        ``tempos_de_chegada``.

    r : np.ndarray
        Final spatial residual vector (meters), aligned with ``tempos_de_chegada``.

    Notes
    -----
    - The Jacobian is computed via forward finite differences by perturbing latitude and
      longitude by a small step in degrees derived from an approximately 1-meter arc length:
      ``step_deg = (1 / AVG_EARTH_RADIUS) * (180 / pi)``.
    - The acceptance rule compares weighted squared residual costs:
      ``sum_i w_i r_i^2`` for the current and candidate solutions.
    - The stopping criterion checks for sufficiently small parameter updates in degrees.
    - Callers should ensure that the input arrays are consistent in length and coordinate
      conventions, and that ``sigma_t`` is strictly positive.

    See Also
    --------
    residuos_espaciais
        Computes spatial residuals and a weighted origin-time estimate for a candidate location.
    computa_pesos
        Maps residuals to IRLS weights using the selected weighting rule.
    solver_2x2_simples
        Solves the 2x2 symmetric normal-equation system used by the LM update.
    """

    # Spatial scale corresponding to sigma_t
    sigma_r = AVG_LIGHT_SPEED * sigma_t

    step_m = 0.1

    x = solucao_inicial.copy()
    pesos = np.ones(len(tempos_de_chegada), dtype=np.float64)
    mu = lm_mu0

    # Convert "step_m" to degrees latitude (~ arc-length on sphere)
    step_deg_lat = (step_m / AVG_EARTH_RADIUS) * (180.0 / np.pi)

    # Initial residuals and t0 (weights initially uniform)
    r, t0 = residuos_espaciais(solucao_inicial=x,
                               tempos_de_chegada=tempos_de_chegada,
                               pontos_de_chegada=pontos_de_chegada,
                               pesos=pesos,
                               sistema_cartesiano=sistema_cartesiano)

    for _ in range(max_iter):

        # Update weights from current residuals (or keep them constant if "none")
        pesos = computa_pesos(residuos=r,
                              sigma=sigma_r,
                              k=k,
                              pesos_alg=pesos_alg)

        # Recompute residuals with the updated weights (t0 depends on weights)
        r, t0 = residuos_espaciais(solucao_inicial=x,
                                   tempos_de_chegada=tempos_de_chegada,
                                   pontos_de_chegada=pontos_de_chegada,
                                   pesos=pesos,
                                   sistema_cartesiano=sistema_cartesiano)

        # Finite-difference steps (degrees)
        dlat = step_deg_lat

        # Longitude step must be scaled by cos(latitude) to represent ~step_m meters eastward
        lat_rad = x[0] * (np.pi / 180.0)
        cphi = np.cos(lat_rad)
        cphi_abs = np.abs(cphi)
        if cphi_abs < 1e-6:
            cphi_abs = 1e-6
        dlon = step_deg_lat / cphi_abs

        # Jacobian samples
        if use_central_diff:
            # LAT: x +/- dlat
            x_lat_p = x.copy()
            x_lat_m = x.copy()
            x_lat_p[0] = x_lat_p[0] + dlat
            x_lat_m[0] = x_lat_m[0] - dlat

            r_lat_p, _ = residuos_espaciais(solucao_inicial=x_lat_p,
                                            tempos_de_chegada=tempos_de_chegada,
                                            pontos_de_chegada=pontos_de_chegada,
                                            pesos=pesos,
                                            sistema_cartesiano=sistema_cartesiano)

            r_lat_m, _ = residuos_espaciais(solucao_inicial=x_lat_m,
                                            tempos_de_chegada=tempos_de_chegada,
                                            pontos_de_chegada=pontos_de_chegada,
                                            pesos=pesos,
                                            sistema_cartesiano=sistema_cartesiano)

            # LON: x +/- dlon
            x_lon_p = x.copy()
            x_lon_m = x.copy()
            x_lon_p[1] = x_lon_p[1] + dlon
            x_lon_m[1] = x_lon_m[1] - dlon

            r_lon_p, _ = residuos_espaciais(solucao_inicial=x_lon_p,
                                            tempos_de_chegada=tempos_de_chegada,
                                            pontos_de_chegada=pontos_de_chegada,
                                            pesos=pesos,
                                            sistema_cartesiano=sistema_cartesiano)

            r_lon_m, _ = residuos_espaciais(solucao_inicial=x_lon_m,
                                            tempos_de_chegada=tempos_de_chegada,
                                            pontos_de_chegada=pontos_de_chegada,
                                            pesos=pesos,
                                            sistema_cartesiano=sistema_cartesiano)

            inv2dlat = 1.0 / (2.0 * dlat)
            inv2dlon = 1.0 / (2.0 * dlon)

        else:
            # Forward differences
            x_lat = x.copy()
            x_lon = x.copy()
            x_lat[0] = x_lat[0] + dlat
            x_lon[1] = x_lon[1] + dlon

            r_lat, _ = residuos_espaciais(solucao_inicial=x_lat,
                                          tempos_de_chegada=tempos_de_chegada,
                                          pontos_de_chegada=pontos_de_chegada,
                                          pesos=pesos,
                                          sistema_cartesiano=sistema_cartesiano)

            r_lon, _ = residuos_espaciais(solucao_inicial=x_lon,
                                          tempos_de_chegada=tempos_de_chegada,
                                          pontos_de_chegada=pontos_de_chegada,
                                          pesos=pesos,
                                          sistema_cartesiano=sistema_cartesiano)

            invdlat = 1.0 / dlat
            invdlon = 1.0 / dlon

        # Assemble weighted normal equations (2x2) for [dphi, dlmb]
        a11 = 0.0
        a12 = 0.0
        a22 = 0.0
        b1 = 0.0
        b2 = 0.0

        for i in range(len(tempos_de_chegada)):

            if use_central_diff:
                j1 = (r_lat_p[i] - r_lat_m[i]) * inv2dlat
                j2 = (r_lon_p[i] - r_lon_m[i]) * inv2dlon
            else:
                j1 = (r_lat[i] - r[i]) * invdlat
                j2 = (r_lon[i] - r[i]) * invdlon

            wi = pesos[i]

            a11 += wi * j1 * j1
            a12 += wi * j1 * j2
            a22 += wi * j2 * j2

            b1 += wi * j1 * (-r[i])
            b2 += wi * j2 * (-r[i])

        # LM damping
        a11_mu = a11 + mu
        a22_mu = a22 + mu

        dphi, dlmb = solver_2x2_simples(a11=a11_mu,
                                        a12=a12,
                                        a22=a22_mu,
                                        b1=b1,
                                        b2=b2)

        # Candidate update
        x_new = x.copy()
        x_new[0] = x_new[0] + dphi
        x_new[1] = x_new[1] + dlmb

        r_new, t0_new = residuos_espaciais(solucao_inicial=x_new,
                                           tempos_de_chegada=tempos_de_chegada,
                                           pontos_de_chegada=pontos_de_chegada,
                                           pesos=pesos,
                                           sistema_cartesiano=sistema_cartesiano)

        # Accept/reject with weighted SSE
        cost = 0.0
        cost_new = 0.0
        for i in range(len(tempos_de_chegada)):
            cost += pesos[i] * r[i] * r[i]
            cost_new += pesos[i] * r_new[i] * r_new[i]

        if cost_new < cost:
            x = x_new
            r = r_new
            t0 = t0_new

            mu = mu * 0.3
            if mu < 1e-12:
                mu = 1e-12

            # stopping rule (very small step in degrees)
            if np.abs(dphi) < 1e-10 and np.abs(dlmb) < 1e-10:
                break
        else:
            mu = mu * 10.0

    return (x,
            t0,
            pesos,
            r)
