import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks


def process_clustering_tf(time_axis, freq_axis, intensity_matrix, bearing_matrix, elevation_matrix):
    # Process a clustering on the pre-processed time/angle representation
    #     - time_axis: vector containing the time values
    #     - freq_axis: vector containing the frequency values
    #     - bearing_matrix: the bearing matrix

    # returns:
    #     - clustering.labels_: the labels of the clusters
    #     - ix, iy: the original coordinates corresponding to make the link between the labels and the
    #       input data in bearing_matrix
    #     - lst_sources: the extracted sources

    time_axis = pd.to_datetime(time_axis)

    res_time = (time_axis[1] - time_axis[0]).total_seconds()
    res_level = 0.5
    intensity_matrix = np.round(intensity_matrix * 2) // 2
    res_freq = freq_axis[1] - freq_axis[0]
    res_bearing = 0.5

    bearing_matrix = np.round(bearing_matrix * 2) // 2

    matrix_time = np.tile((time_axis - time_axis[0]).total_seconds(), (bearing_matrix.shape[0], 1))
    matrix_freq = np.tile(freq_axis, (bearing_matrix.shape[1], 1)).T

    ix = np.tile(np.arange(len(time_axis)), (bearing_matrix.shape[0], 1))
    iy = np.tile(np.arange(len(freq_axis)), (bearing_matrix.shape[1], 1)).T

    zz = np.round((bearing_matrix.ravel() * 2) // 2)
    # ee = elevation_matrix.ravel()  # added in new version
    tt = matrix_time.ravel()
    ff = matrix_freq.ravel()
    ll = intensity_matrix.ravel()

    ix = ix.ravel()
    iy = iy.ravel()

    ii = ~np.isnan(ll)
    zz = zz[ii]
    # ee = ee[ii]
    tt = tt[ii]
    ff = ff[ii]
    ll = ll[ii]
    ix = ix[ii]
    iy = iy[ii]

    # Conditionnement des differentes grandeurs pour le clustering
    tt_c = (tt / res_time) / 25
    ll_c = (ll / res_level) / 50
    zz_c = (zz / res_bearing) / 3.5
    ff_c = (ff / res_freq) / 32  # must be big enough to be sure to separate the freq channels

    X = np.transpose([tt_c, ff_c, zz_c, ll_c])

    # min_samples=4 => will work if the input data contains ~3.5 spectrums/second (this condition is supposed to be solved when computing the spectrograms)
    clustering = DBSCAN(eps=1, min_samples=10).fit(X)

    unique_labels = set(clustering.labels_)

    lst_sources = []
    lst_points = []

    for l in unique_labels:
        if l != -1:
            ic = clustering.labels_ == l
            nb_point = len(np.where(ic)[0])

            lst_points.append(nb_point)
            lst_sources.append([l, nb_point])

    lst_points = np.array(lst_points)
    i_sort = np.argsort(-lst_points)

    if len(lst_sources) > 0:
        lst_sources = np.array(lst_sources)
        lst_sources = lst_sources[i_sort, :]

    return clustering.labels_, ix, iy, lst_sources


def get_source_details(labels: np.ndarray, ix: np.ndarray, iy: np.ndarray, src: np.ndarray, bearing_matrix: np.ndarray, elevation_matrix: np.ndarray = None):
    """
    Extracts the source details for an identified source in a frequency channel.

    Parameters:
    - labels: Array of cluster labels.
    - ix: Array of x-coordinates.
    - iy: Array of y-coordinates.
    - src: Source identifier.
    - bearing_matrix: Matrix containing bearing values.
    - elevation_matrix: Matrix containing elevation values (optional).

    Returns:
    - source_bearing: Matrix with source bearing values.
    - az_along_time_axis: Mean bearing values along the time axis.
    - ele_along_time_axis: Mean elevation values along the time axis (if elevation_matrix is provided).
    """

    source_bearing = np.full(bearing_matrix.shape, np.nan)

    source_indices = labels == src[0]
    freq_indices = iy[source_indices].astype(int)
    time_indices = ix[source_indices].astype(int)

    source_bearing[freq_indices, time_indices] = bearing_matrix[freq_indices, time_indices]
    az_along_time_axis = np.nanmean(source_bearing, axis=0)

    if elevation_matrix is not None:
        source_elevation = np.full(elevation_matrix.shape, np.nan)
        source_elevation[freq_indices, time_indices] = elevation_matrix[freq_indices, time_indices]
        ele_along_time_axis = np.nanmean(source_elevation, axis=0)
        return source_bearing, az_along_time_axis, ele_along_time_axis

    return source_bearing, az_along_time_axis


def get_sources_azimuths(time_axis, freq_axis, intensity_matrix, bearing_matrix, elevation_matrix):

    labels, ix, iy, lst_sources = process_clustering_tf(time_axis, freq_axis, intensity_matrix, bearing_matrix, elevation_matrix)

    id_source = 0
    df_sources = pd.DataFrame([])

    for cpt, source in enumerate(lst_sources):
        source_matrix, az = get_source_details(labels, ix, iy, source, bearing_matrix)
        i_t, i_f = np.argwhere(~np.isnan(source_matrix))[:, 1], np.argwhere(~np.isnan(source_matrix))[:, 0]
        t_source = time_axis[i_t]
        f_source = freq_axis[i_f]
        l_source = intensity_matrix[i_f, i_t]
        az_source = az[i_t]

        if (t_source.max() - t_source.min()).total_seconds() > 3600 * 0.5:
            df_tmp = pd.DataFrame(np.transpose([t_source, f_source, l_source, az_source]),
                                 columns=['time', 'freq', 'level', 'az'])
            df_tmp['id_source'] = id_source
            df_sources = pd.concat([df_sources, df_tmp])
            id_source += 1

    if df_sources.shape[0] > 0:
        df_sources.set_index('id_source', inplace=True)
        df_sources['datetime'] = pd.to_datetime(df_sources.time)

    return df_sources

