import numpy as np
import pandas as pd
import scipy.signal as sg
import cv2 as cv


def process_clustering_cepstro(time_axis: np.ndarray, quef_axis: np.ndarray, cepstro: np.ndarray, thr: float = 12, kernel_size: int = 21) -> pd.DataFrame:
    """
    Process a clustering on the pre-processed time/quefrency representation.

    Parameters
    ----------
    time_axis : np.ndarray
        Vector containing the time values.
    quef_axis : np.ndarray
        Vector containing the quefrency values.
    cepstro : np.ndarray
        Cepstrogram.
    thr : float, optional
        Threshold for cepstrogram normalization.
    kernel_size : int, optional
        Kernel size for median filtering.

    Returns
    -------
    df_sources : pd.DataFrame
        DataFrame containing the processed clustering information.
    """
    # Normalize the cepstrogram
    cepstro_matrix = 100 * (np.abs(cepstro) / 0.2)

    # Apply median filtering
    filtered_bkg_noise = sg.medfilt2d(cepstro_matrix, (kernel_size, kernel_size))

    # Thresholding
    cepstro_over_bkg_noise = cepstro_matrix.copy()
    cepstro_over_bkg_noise[cepstro_over_bkg_noise <= filtered_bkg_noise + thr] = np.nan

    cepstro_matrix = cepstro_over_bkg_noise
    id_time = np.arange(len(time_axis))

    # Round and create matrices
    cepstro_matrix = np.round(cepstro_matrix * 2) // 2
    matrix_time = np.tile(time_axis, (len(quef_axis), 1))
    matrix_id_time = np.tile(id_time, (len(quef_axis), 1))
    matrix_quef = np.tile(quef_axis[:, np.newaxis], (1, len(time_axis)))
    ix = np.tile(np.arange(len(time_axis)), (len(quef_axis), 1))
    iy = np.tile(np.arange(len(quef_axis))[:, np.newaxis], (1, len(time_axis)))

    # Flatten matrices
    tt = matrix_time.ravel()
    id = matrix_id_time.ravel()
    qq = matrix_quef.ravel()
    ll = cepstro_matrix.ravel()

    # Filter out NaN values
    valid_indices = ~np.isnan(ll)
    id = id[valid_indices]
    tt = tt[valid_indices]
    qq = qq[valid_indices]
    ll = ll[valid_indices]

    # Create DataFrame
    df_sources = pd.DataFrame({'datetime': tt, 'id_time': id, 'quef': qq, 'level': ll})
    df_sources['id_time'] = df_sources['id_time'].astype(int)
    df_sources['datetime'] = pd.to_datetime(df_sources['datetime'])

    return df_sources


def get_ray_length(water_depth: float, sensor_depth: float, source_distance: float, reflections: list) -> float:
    """
    Returns the ray length for a given distance, knowing the sensor depth, water depth, and number of reflections.

    Parameters
    ----------
    water_depth : float
        Depth of the water.
    sensor_depth : float
        Depth of the sensor.
    source_distance : float
        Distance between the source and the sensor.
    reflections : list of int
        List containing the number of bottom reflections and surface reflections [bottom_reflections, surface_reflections].

    Returns
    -------
    ray_length : float
        The calculated ray length.
    """
    bottom_reflections = reflections[0]
    surface_reflections = reflections[1]
    k = 2 * (surface_reflections - bottom_reflections) + 1
    ray_length = np.sqrt((2 * bottom_reflections * water_depth + sensor_depth * k) ** 2 + source_distance ** 2)
    return ray_length


def find_the_best_distance(dmin: float, dmax: float, dd: float, measured_tdoa: float, s1: int, b1: int, s2: int, b2: int, water_depth: float, sensor_depth: float, sound_speed: float) -> float:
    """
    Find the best distance that matches the measured TDOA (Time Difference of Arrival).

    Parameters
    ----------
    dmin : float
        Minimum distance to consider.
    dmax : float
        Maximum distance to consider.
    dd : float
        Distance step size.
    measured_tdoa : float
        Measured Time Difference of Arrival.
    s1 : int
        Number of surface reflections for the first path.
    b1 : int
        Number of bottom reflections for the first path.
    s2 : int
        Number of surface reflections for the second path.
    b2 : int
        Number of bottom reflections for the second path.
    water_depth : float
        Depth of the water.
    sensor_depth : float
        Depth of the sensor.
    sound_speed : float
        Speed of sound in water.

    Returns
    -------
    best_distance : float
        The distance that best matches the measured TDOA.
    """
    dist_range = np.arange(dmin, dmax, dd)

    l1 = get_ray_length(water_depth, sensor_depth, dist_range, [b1, s1])
    l2 = get_ray_length(water_depth, sensor_depth, dist_range, [b2, s2])

    computed_tdoa = np.abs((l1 - l2) / sound_speed)

    idx = np.argmin(np.abs(computed_tdoa - measured_tdoa))
    return dist_range[idx]


def get_source_possible_distances(df_sources: pd.DataFrame,
                                  max_rebounds: int,
                                  dist_min: float, dist_max: float,
                                  cepstro_time_scale: np.ndarray,
                                  dist_resolution: float,
                                  quef_resolution: float,
                                  water_depth: float, sensor_depth: float, sound_speed: float) -> tuple:
    """
    Compute possible source distances based on cepstrogram data.

    Parameters
    ----------
    df_sources : pd.DataFrame
        DataFrame containing source information.
    max_rebounds : int
        Maximum number of rebounds to consider.
    dist_min : float
        Minimum distance to consider.
    dist_max : float
        Maximum distance to consider.
    cepstro_time_scale : np.ndarray
        Time scale of the cepstrogram.
    dist_resolution : float
        Distance resolution.
    quef_resolution : float
        Quefrency resolution.
    water_depth : float
        Depth of the water.
    sensor_depth : float
        Depth of the sensor.
    sound_speed : float
        Speed of sound in water.

    Returns
    -------
    df_dist : pd.DataFrame
        DataFrame containing possible source distances.
    lst_ref : np.ndarray
        Array of unique references.
    """
    res_dist = []
    for quef in df_sources.quef.unique():
        for second in range(max_rebounds - 1):
            for first in range(second, max_rebounds):
                if first != second:
                    best_dist = find_the_best_distance(dist_min, dist_max, dist_resolution, quef, first, first, second,
                                                       second, water_depth, sensor_depth, sound_speed)
                    best_dist = int(np.round(best_dist / dist_resolution) * dist_resolution)
                    res_dist.append([quef, best_dist, f'{second}-{first}'])

    res_dist = pd.DataFrame(res_dist, columns=['quef', 'dist', 'ref'])
    lst_ref = res_dist.ref.unique()
    res_dist.set_index('ref', inplace=True)
    res_dist = res_dist.reset_index().set_index(['ref', 'quef'])

    df_dist = pd.DataFrame([])
    for ref in lst_ref:
        dftmp = df_sources.copy()
        dftmp['ref'] = ref
        dftmp.set_index(['ref', 'quef'], inplace=True)
        dftmp['dist'] = res_dist['dist']
        df_dist = pd.concat([df_dist, dftmp])

    df_dist = df_dist[df_dist.dist > 0]

    df_dist = df_dist.reset_index()
    df_dist['idQuef'] = (df_dist.quef / quef_resolution).astype(int)
    df_dist['idDist'] = (df_dist.dist / dist_resolution).astype(int)
    df_dist.set_index('datetime', inplace=True)
    df_dist = df_dist.reset_index().set_index('ref')

    return df_dist, lst_ref


def get_distance_matrix_per_ref(df_dist: pd.DataFrame, dist_scale: np.ndarray, time_scale_cepstro: np.ndarray, kernel_size: tuple = (12, 21)) -> np.ndarray:
    """
    Compute the distance matrix per reference.

    Parameters
    ----------
    df_dist : pd.DataFrame
        DataFrame containing distance information.
    dist_scale : np.ndarray
        Array of distance scale values.
    time_scale_cepstro : np.ndarray
        Array of time scale values for the cepstrogram.
    kernel_size : tuple, optional
        Size of the smoothing kernel.

    Returns
    -------
    matrix_layer : np.ndarray
        3D array containing the distance matrix per reference.
    """
    kernel = np.ones(kernel_size, np.float32)

    df_dist = df_dist.reset_index().set_index('ref')

    unique_refs = df_dist.index.unique()
    nb_layers = len(unique_refs)

    matrix_layer = np.zeros((nb_layers, len(dist_scale), len(time_scale_cepstro)))

    for iLayer, layer in enumerate(unique_refs):
        dftmp = df_dist.loc[layer]
        matrix_layer[iLayer, dftmp.idDist.to_numpy().astype(int), dftmp.id_time.to_numpy().astype(int)] = 1

    # Apply smoothing filter to each layer
    for iLayer in range(nb_layers):
        matrix_layer[iLayer] = cv.filter2D(matrix_layer[iLayer], -1, kernel)
        matrix_layer[iLayer] /= matrix_layer[iLayer].max()

    return matrix_layer
