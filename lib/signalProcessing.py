import numpy as np
import scipy.signal as sp
import pandas as pd
import quaternion as quat
from scipy.signal import decimate, resample
from obspy import Stream
from obspy.core import Trace
from numba import jit, prange
# from numba.fft import rfft, rfftfreq
# import scipy.fft as fft

def get_slog(spectro_cpx_pressure, f, fftsize):
    fs=(f.max()-f.min())*2
    return 120+10*np.log10(np.abs(spectro_cpx_pressure))-10*np.log10(fs/fftsize)

    
def get_cepstro_from_trace(tr, fftsize: int, noverlap: int, integration: int = None, demBounds: list = None) -> tuple:
    """
    For convenience, computes both the spectrogram and cepstrogram with the given parameters.

    Parameters
    ----------
    tr : Trace
        The ObsPy trace containing the data to process.
    fftsize : int
        FFT window size (must be a power of 2).
    noverlap : int
        Number of points to overlap between windows.
    integration : int, optional
        Number of spectra to average to obtain the resulting spectrum.
    demBounds : list of float, optional
        Frequency boundaries [fmin, fmax] for demodulation.

    Returns
    -------
    tuple
        Tuple containing:
        - t: Time scale of the cepstrogram.
        - q: Quefrency scale of the cepstrogram.
        - c: The resulting cepstrogram.
    """
    f, t, S = get_spectrogram(tr, fftsize, noverlap, integration, demBounds)
    t, q, C = get_filtered_cepstro(t, f, S)
    return t, q, C


def integrate_tf_representation(t: np.ndarray, R: np.ndarray, i: int) -> tuple:
    """
    Compute the integration of the time-dependent representation (Spectrogram or Cepstrogram).

    Parameters
    ----------
    t : np.ndarray
        Array of time values.
    R : np.ndarray
        Spectrogram or cepstrogram.
    i : int
        Number of spectra to average to obtain the resulting representation.

    Returns
    -------
    tuple
        Tuple containing:
        - tres: New time scale.
        - Rres: New integrated representation.
    """
    tres = t[0::i]
    Rres = pd.DataFrame(R).T.rolling(i, min_periods=1).mean().T.to_numpy()[:, 0::i]
    return tres, Rres



def get_spectrogram(tr: Trace, fftsize: int, noverlap: int, integration: int = None, demBounds: list = None) -> tuple:
    samples = tr.data
    additional_freq = 0
    sampling_rate = tr.stats.sampling_rate

    if demBounds:
        try:
            samples, sampling_rate = get_demodulated_samples(samples, sampling_rate, demBounds)
            additional_freq = demBounds[0]
        except Exception as e:
            print(f"Error in demodulation: {e}")

    # frequencies, times, spectrogram = compute_stft_cpu(samples, sampling_rate, fftsize, noverlap)
    frequencies, times, spectrogram = sp.stft(samples, fs=sampling_rate, nperseg=int(fftsize), noverlap=noverlap)   
    frequencies += additional_freq

    times = pd.date_range(start=tr.stats.starttime.datetime,
                          end=tr.stats.endtime.datetime,
                          periods=times.shape[0])

    if integration:
        times, spectrogram = integrate_tf_representation(times, np.abs(spectrogram), integration)

    return frequencies, times, spectrogram



# @jit(nopython=False, parallel=True)
# def compute_intensity_vector(dict_spectro, no_hydro):
#     intensity_vector = {}
#     if no_hydro:
#         intensity_vector['x'] = np.abs(dict_spectro['2'])
#         intensity_vector['y'] = np.abs(dict_spectro['1'])
#         try:
#             intensity_vector['z'] = np.abs(dict_spectro['Z'])
#             intensity_vector['iv'] = np.sqrt(intensity_vector['x'] ** 2 + intensity_vector['y'] ** 2 + intensity_vector['z'] ** 2)
#         except KeyError:
#             intensity_vector['iv'] = np.sqrt(intensity_vector['x'] ** 2 + intensity_vector['y'] ** 2)
#     else:
#         intensity_vector['x'] = 0.5 * np.real(dict_spectro['H'] * dict_spectro['2'].conj())
#         intensity_vector['y'] = 0.5 * np.real(dict_spectro['H'] * dict_spectro['1'].conj())
#         intensity_vector['z'] = 0.5 * np.real(dict_spectro['H'] * dict_spectro['Z'].conj())
#     return intensity_vector

def get_intensity_vector(st: Stream, fftsize: int, noverlap: int, integration: int = None, demBounds: list = None, inv = None, orientation_angle: float = None, no_hydro: bool = False) -> tuple:
    """
    Compute the intensity vector and spectrograms for the given stream. 

    Parameters
    ----------
    st : Stream
        The ObsPy Stream object containing the data.
    fftsize : int
        FFT window size (must be a power of 2).
    noverlap : int
        Number of points to overlap between windows.
    integration : int, optional
        Number of spectra to average to obtain the resulting spectrum.
    demBounds : list of float, optional
        Frequency boundaries [fmin, fmax] for demodulation.
    inv : Inventory, optional
        Inventory object for response removal.
    orientation_angle : float, optional
        Orientation angle for rotation.
    no_hydro : bool, optional
        Flag to indicate if hydrophone data is not available.

    Returns
    -------
    tuple
        Tuple containing:
        - f: Frequency scale of the computed time/frequency representations.
        - t: Time scale of the computed time/frequency representations.
        - intensity_vector: A dict of intensity vector representations {'iv', 'x', 'y', 'z'}.
        - dict_spectro: A dict of the channel spectrograms {'iv', 'x', 'y', 'z'}.
    """
    channel_map = {'1': '1', '2': '2', '3': 'Z', 'Z': 'Z', 'H': 'H', 'N': '1', 'E': '2'}
    dict_spectro = {}
    if inv:
        st.remove_response(inventory=inv)

    for tr in st:
        f, t, S = get_spectrogram(tr, fftsize, noverlap, demBounds=demBounds)
        channel_code = channel_map[tr.stats.channel[-1]]
        dict_spectro[channel_code] = S

    intensity_vector = {}

    if no_hydro:
        intensity_vector['x'] = np.abs(dict_spectro['2'])
        intensity_vector['y'] = np.abs(dict_spectro['1'])
        try:
            intensity_vector['z'] = np.abs(dict_spectro['Z'])
            intensity_vector['iv'] = np.sqrt(intensity_vector['x'] ** 2 + intensity_vector['y'] ** 2 + intensity_vector['z'] ** 2)
        except KeyError:
            intensity_vector['iv'] = np.sqrt(intensity_vector['x'] ** 2 + intensity_vector['y'] ** 2)
            print('Error: no elevation')
    else:
        intensity_vector['x'] = 0.5 * np.real(dict_spectro['H'] * dict_spectro['2'].conj())
        intensity_vector['y'] = 0.5 * np.real(dict_spectro['H'] * dict_spectro['1'].conj())
        intensity_vector['z'] = 0.5 * np.real(dict_spectro['H'] * dict_spectro['Z'].conj())
        intensity_vector['iv'] = np.sqrt(intensity_vector['x']**2 +
                                         intensity_vector['y']**2 +
                                         intensity_vector['z']**2)


    # intensity_vector = compute_intensity_vector(dict_spectro, no_hydro)

    t_res = t
    if integration:
        for key in intensity_vector.keys():
            t_res, intensity_vector[key] = integrate_tf_representation(t, intensity_vector[key], integration)
        for key in dict_spectro.keys():
            t_res, dict_spectro[key] = integrate_tf_representation(t, np.abs(dict_spectro[key]), integration)

    if orientation_angle:
        intensity_vector = apply_rotation(orientation_angle, intensity_vector)

    return f, t_res, intensity_vector, dict_spectro


def apply_rotation(orientation_angle: float, iv: dict) -> dict:
    """
    Apply a 3D rotation matrix to correct pitch, roll, and heading.

    Parameters
    ----------
    orientation_angle : float
        The orientation angle in degrees.
    iv : dict
        Dictionary containing the intensity vector components 'x', 'y', and 'z'.

    Returns
    -------
    dict
        Dictionary containing the rotated intensity vector components 'x', 'y', and 'z'.
    """
    # Convert the intensity vector to quaternion form
    iv_quat = spectro_to_quaternion(iv['x'], iv['y'], iv['z'])

    iv_rot = np.copy(iv_quat)
    # Find the rotation matrices
    for i in range(iv['x'].shape[1]):
        qrotx = qrot_axis_angle(np.array([1, 0, 0]), 0)
        qroty = qrot_axis_angle(np.array([0, 1, 0]), 0)
        qrotz = qrot_axis_angle(np.array([0, 0, 1]), np.deg2rad(-orientation_angle))  # validated with preset heading

    for i in range(iv['x'].shape[1]):
        tmp = rot_ptary(iv_quat[:, i], qrotx)
        tmp = rot_ptary(tmp, qroty)
        iv_rot[:, i] = rot_ptary(tmp, qrotz)

    sx, sy, sz = quatarray_to_xyzarray(iv_rot)
    iv['x'], iv['y'], iv['z'] = sx, sy, sz
    return iv

def get_vector_angle(intensity_vector: dict) -> tuple:
    """
    Compute the bearing and elevation angles from the intensity vector.

    Parameters
    ----------
    intensity_vector : dict
        Dictionary containing the intensity vector components 'x', 'y', and 'z'.

    Returns
    -------
    tuple
        Tuple containing the bearing and elevation angles in degrees.
    """
    bearing = np.arctan2(intensity_vector['x'], intensity_vector['y'])
    bearing = np.rad2deg(bearing)
    bearing = np.mod(bearing + 360, 360)
    bearing_amplitude = np.sqrt(intensity_vector['x'] ** 2 + intensity_vector['y'] ** 2)

    if 'z' in intensity_vector:
        elevation = np.arctan2(intensity_vector['z'], bearing_amplitude)
        elevation = np.rad2deg(elevation)
    else:
        elevation = bearing * np.nan
    return bearing, elevation


def spectro_to_quaternion(sx: np.ndarray, sy: np.ndarray, sz: np.ndarray) -> np.ndarray:
    """
    Convert spectrogram components to quaternions.

    Parameters
    ----------
    sx : np.ndarray
        X component of the spectrogram.
    sy : np.ndarray
        Y component of the spectrogram.
    sz : np.ndarray
        Z component of the spectrogram.

    Returns
    -------
    np.ndarray
        Array of quaternions.
    """
    initial_shape = np.shape(sx)
    x = sx.ravel()
    y = sy.ravel()
    z = sz.ravel()

    npary = np.column_stack((x, y, z))

    tmpary = np.zeros((npary.shape[0], 4))
    tmpary[:, 1:] = npary
    qnpary = quat.as_quat_array(tmpary)

    qnpary = qnpary.reshape(initial_shape)

    return qnpary

def npXYZary_to_quaternion(npary: np.ndarray) -> np.ndarray:
    """
    Convert an array of XYZ components to quaternions.

    Parameters
    ----------
    npary : np.ndarray
        Array of XYZ components.

    Returns
    -------
    np.ndarray
        Array of quaternions.
    """
    tmpary = np.zeros((npary.shape[0], 4))
    tmpary[:, 1:] = npary
    qnpary = quat.as_quat_array(tmpary)
    return qnpary

def quatarray_to_xyzarray(quatarray: np.ndarray) -> tuple:
    """
    Convert an array of quaternions to separate X, Y, and Z component arrays.

    Parameters
    ----------
    quatarray : np.ndarray
        Array of quaternions.

    Returns
    -------
    tuple
        Tuple containing arrays of X, Y, and Z components.
    """
    initial_shape = np.shape(quatarray)
    arr = quatarray.ravel()

    float_array = quat.as_float_array(arr)
    x = float_array[:, 1].reshape(initial_shape)
    y = float_array[:, 2].reshape(initial_shape)
    z = float_array[:, 3].reshape(initial_shape)

    return x, y, z

def qrot_axis_angle(rotation_axis: np.ndarray, rotation_angle: float) -> quat.quaternion:
    """
    Create a quaternion representing a rotation around a given axis by a given angle.

    Parameters
    ----------
    rotation_axis : np.ndarray
        The axis of rotation (3D vector).
    rotation_angle : float
        The angle of rotation in radians.

    Returns
    -------
    quat.quaternion
        The resulting quaternion representing the rotation.
    """
    half_angle = rotation_angle / 2.0
    # Normalize the rotation axis
    normalized_axis = rotation_axis / np.linalg.norm(rotation_axis)
    q_rot = quat.quaternion(0, *normalized_axis)
    qrot = np.cos(half_angle) + np.sin(half_angle) * q_rot
    return qrot

def rot_ptary(quat_arr: np.ndarray, rotquat: quat.quaternion) -> np.ndarray:
    """
    Rotate an array of quaternions by a given rotation quaternion.

    Parameters
    ----------
    quat_arr : np.ndarray
        Array of quaternions to be rotated.
    rotquat : quat.quaternion
        Rotation quaternion.

    Returns
    -------
    np.ndarray
        Array of rotated quaternions.
    """
    rotated_quat_arr = rotquat * quat_arr * rotquat.conjugate()
    return rotated_quat_arr


def get_demodulated_samples(samples: np.ndarray, fs: float, demodulation_boundaries: list) -> tuple:
    """
    Compute the demodulation of the given samples at sample rate fs.

    Parameters
    ----------
    samples : np.ndarray
        The samples (time series) to filter and demodulate.
    fs : float
        The original sample rate.
    demodulation_boundaries : list of float
        Array [fmin, fmax] containing the demodulation boundaries.

    Returns
    -------
    demodulated_samples : np.ndarray
        A time series containing the demodulated samples.
    new_fs : float
        The new sample rate (fmax - fmin) * 2.
    Compute the demodulation of the given samples at sample rate fs.

    Parameters
    ----------
    samples : np.ndarray
        The samples (time series) to filter and demodulate.
    fs : float
        The original sample rate.
    demodulation_boundaries : list of float
        Array [fmin, fmax] containing the demodulation boundaries.

    Returns
    -------
    demodulated_samples : np.ndarray
        A time series containing the demodulated samples.
    new_fs : float
        The new sample rate (fmax - fmin) * 2.
    """

    fmin, fmax = demodulation_boundaries
    band_width = fmax - fmin
    new_fs = band_width * 2

    current_fs = fs
    filtered = np.copy(samples)
    order = 4
    while (current_fs / 2) > fmax*4:

        filtered = decimate(filtered, 4, ftype='fir')
        current_fs /= 4

    if demodulation_boundaries[0] > 0:
        # Bandpass filtering
        b, a = sp.butter(order, demodulation_boundaries, 'bandpass', fs=current_fs)
        filtered = sp.filtfilt(b, a, filtered, padlen=150)

        # Demodulation step
        time_band = np.arange(len(filtered)) / current_fs
        filtered = np.real(filtered) * np.cos(2 * np.pi * demodulation_boundaries[0] * time_band)

        # Lowpass filter
        b, a = sp.butter(order, band_width, 'lowpass', fs=current_fs)
        filtered = sp.filtfilt(b, a, filtered)
    else:
        # Lowpass filter
        b, a = sp.butter(order, band_width, 'lowpass', fs=current_fs)
        filtered = sp.filtfilt(b, a, filtered)

    # Resample
    demodulated_samples = resample(filtered, int(len(filtered) / (current_fs / new_fs)))

    return demodulated_samples, new_fs



def get_cepstro(t: np.ndarray, f: np.ndarray, s: np.ndarray) -> tuple:
    """
    Compute the cepstrum of the given spectrogram.

    Parameters
    ----------
    t : np.ndarray
        Array of time values.
    f : np.ndarray
        Array of frequency values.
    s : np.ndarray
        Spectrogram (complex values).

    Returns
    -------
    t : np.ndarray
        Array of time values.
    q : np.ndarray
        Array of quefrency values.
    c : np.ndarray
        Cepstrum of the spectrogram.
    """

    c = np.zeros(np.shape(s))
#     q = 1/f
#     q[0] = 0
    df = f[1] - f[0]
    q = np.fft.rfftfreq(2*(len(f) - 1), df)
    c = np.fft.irfft(np.log(np.abs(s)), axis=-2)
    c = c[..., :len(q),:]
    return t, q, c

def find_knees(s):
    yn = s / np.max(s, axis=-1)[..., np.newaxis]
    xn = np.linspace(0, 1, yn.shape[-1])
    dn = 1 - yn - xn
    knee = np.argmax(dn, axis=-1)
    return knee


def get_filtered_cepstro(t: np.ndarray, f: np.ndarray, spectro: np.ndarray) -> tuple:
    """
    Compute the filtered cepstrum of the given spectrogram using SVD.

    Parameters
    ----------
    t : np.ndarray
        Array of time values.
    f : np.ndarray
        Array of frequency values.
    spectro : np.ndarray
        Spectrogram (complex values).

    Returns
    -------
    t : np.ndarray
        Array of time values.
    q : np.ndarray
        Array of quefrency values.
    cepstro_filtered : np.ndarray
        Filtered cepstrum of the spectrogram.
    """
    t, q, c = get_cepstro(t, f, spectro)

    try:
        u, s, vh = np.linalg.svd(c - np.mean(c), full_matrices=False)
        knee = find_knees(s)
        mask = np.arange(s.shape[0]) < knee
        s[mask] = 0
        cepstro_filtered = u @ np.diag(s) @ vh
        cepstro_filtered = u @ np.diag(s) @ vh
        return t, q, cepstro_filtered

    except np.linalg.LinAlgError as e:
        print(f'SVD filtering failed: {e}')
    except np.linalg.LinAlgError as e:
        print(f'SVD filtering failed: {e}')
        return t, q, c


def dirstat(theta: np.ndarray, nhist: int, bins: int, t: np.ndarray) -> tuple:
    """
    Compute directional statistics for the given angles.

    Parameters
    ----------
    theta : np.ndarray
        Array of angles.
    nhist : int
        Number of histogram bins.
    bins : int
        Number of bins for the histogram.
    t : np.ndarray
        Array of time values.

    Returns
    -------
    theta_centers : np.ndarray
        Array of bin centers for the histogram.
    t : pd.DatetimeIndex
        Array of time values.
    h : np.ndarray
        Histogram of the angles.
    """
    theta_edges = np.linspace(-np.pi, np.pi, bins + 1)
    theta_centers = (theta_edges[1:] + theta_edges[:-1]) / 2
    hist = lambda a: np.histogram(a, bins=theta_edges, density=True)[0]

    # Reshape theta to ensure it is divisible by nhist

    # Reshape theta to ensure it is divisible by nhist
    theta = theta[:, :theta.shape[1] - (theta.shape[1] % nhist)]
    theta = theta.T.reshape((theta.shape[1] // nhist, theta.shape[0] * nhist))
    h = np.apply_along_axis(hist, 1, theta).T

    t = pd.date_range(t[0], t[-1], periods=h.shape[1])
    t = pd.date_range(t[0], t[-1], periods=h.shape[1])
    return theta_centers, t, h