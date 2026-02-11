from obspy.core import inventory
import os
import logging
import pandas as pd
import glob
from datetime import timedelta
from obspy import read, UTCDateTime, Stream
from datetime import timedelta
import gc
import numpy as np
from pydub import AudioSegment
from mutagen.flac import FLAC
from scipy.signal import butter, filtfilt

import os
import glob
import pandas as pd
import logging
from obspy.core.trace import Trace




def get_network_details(net_code: str, inventory_path: str) -> pd.DataFrame:
    """
    Extracts the network details from the inventory XML file.

    Parameters
    ----------
    net_path : str
        The network code.
    inventory_path : str
        Path to the inventory folder.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the station details.
    """
    logging.info(f'Loading inventory for network: {net_code} from {inventory_path}')
    inv = inventory.read_inventory(os.path.join(inventory_path, f'{net_code}*.xml'))

    df_stations = []
    for net in inv:
        for station in net:
            for channel in station:
                sensitivity = channel.response.instrument_sensitivity.value if channel.response.instrument_sensitivity else None
                net_start = net.start_date.datetime if net.start_date else None
                net_end = net.end_date.datetime if net.end_date else None
                sta_start = station.start_date.datetime if station.start_date else None
                sta_end = station.end_date.datetime if station.end_date else None

                try:
                    if sensitivity is not None:
                        df_stations.append([
                            net.code,
                            net_start,
                            net_end,
                            station.code,
                            station.longitude,
                            station.latitude,
                            station.elevation,
                            channel.code,
                            channel.sample_rate,
                            sensitivity,
                            sta_start,
                            sta_end
                        ])
                except Exception as e:
                    logging.error(f'Error in function "get_network_details": {net.code} {e}')
                    df_stations.append([
                        net.code,
                        station.code,
                        station.longitude,
                        station.latitude,
                        station.elevation,
                        channel.code,
                        channel.sample_rate,
                        None,
                        None,
                        None
                    ])

    df_stations = pd.DataFrame(df_stations, columns=[
        'net', 'net_start', 'net_end', 'sta', 'lon', 'lat', 'ele', 'cha', 'sample_rate', 'sensitivity', 'starttime', 'endtime'
    ])
    return df_stations.drop_duplicates()


def get_network_file_list(net: str, sta: str, sds_path: str) -> pd.DataFrame:
    """
    Create the list of mseed files available for the network and station.

    Parameters
    ----------
    net : str
        The network code ("*" for all networks").
    sta : str
        The station name ("*" for all stations").
    sds_path : str
        Path to the SDS files.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with all the corresponding files and some details.
    """

    # Use os.path.join to create a platform-independent file pattern
    logging.info(f'Loading file list for network: {net}, station: {sta} from {sds_path}')
    file_pattern = os.path.join(sds_path, '*', net, sta, '*', '*.*')
    logging.info(f'Looking for files with pattern: {file_pattern}')
    file_list = sorted(glob.glob(file_pattern))
    logging.info(f'Found {len(file_list)} files.')

    # Separate valid and invalid files
    valid_files = []
    invalid_files = []

    for file in file_list:
        # Check if the file path has the expected structure
        path_parts = file.split(os.sep)
        if len(path_parts) >= 6 and len(path_parts[-1].split('.')) >= 7:
            valid_files.append(file)
        else:
            invalid_files.append(file)

    # Log invalid files
    if invalid_files:
        logging.warning(f'{len(invalid_files)} invalid files found:')
        for invalid_file in invalid_files:
            logging.warning(f'Invalid file: {invalid_file}')

    # Process valid files
    df_files = pd.DataFrame(valid_files, columns=['filename'])
    logging.info(f'Processing {len(df_files)} valid files.')

    # Normalize file paths to use the correct separator for the platform
    df_files['filename'] = df_files['filename'].apply(os.path.normpath)

    # Extract components from the filename
    df_files['year'] = df_files['filename'].apply(lambda x: x.split(os.sep)[-5])
    df_files['net'] = df_files['filename'].apply(lambda x: x.split(os.sep)[-4])
    df_files['sta'] = df_files['filename'].apply(lambda x: x.split(os.sep)[-3])
    df_files['cha'] = df_files['filename'].apply(lambda x: x.split(os.sep)[-2])
    df_files['julian'] = df_files['filename'].apply(lambda x: x.split(os.sep)[-1].split('.')[6])

    df_files['starttime'] = df_files['filename'].apply(
        lambda x: x.split(os.sep)[-1].split('.')[7] if len(x.split(os.sep)[-1].split('.')) > 7 else '000000'
    )

    # Convert to datetime
    df_files['starttime'] = pd.to_datetime(
        df_files['year'] + df_files['julian'] + df_files['starttime'], format='%Y%j%H%M%S', errors='coerce'
    )
    df_files['datetime'] = pd.to_datetime(
        df_files['year'] + df_files['julian'], format='%Y%j', errors='coerce'
    )

    return df_files
# def get_network_file_list(net: str, sta: str, sds_path: str) -> pd.DataFrame:
#     """
#     Create the list of mseed files available for the network and station.

#     Parameters
#     ----------
#     net : str
#         The network code ("*" for all networks").
#     sta : str
#         The station name ("*" for all stations").
#     sds_path : str
#         Path to the SDS files.

#     Returns
#     -------
#     pd.DataFrame
#         A pandas DataFrame with all the corresponding files and some details.
#     """
#     # Use os.path.join to create a platform-independent file pattern

#     logging.info(f'Loading file list for network: {net}, station: {sta} from {sds_path}')
#     file_pattern = os.path.join(sds_path, '*', net, sta, '*', '*.*')
#     logging.info(f'Looking for files with pattern: {file_pattern}')
#     file_list = sorted(glob.glob(file_pattern))
#     df_files = pd.DataFrame(file_list, columns=['filename'])
#     logging.info(f'Found {len(df_files)} files.')

#     # Normalize file paths to use the correct separator for the platform
#     df_files['filename'] = df_files['filename'].apply(os.path.normpath)
#     logging.info(df_files.head())
#     logging.info(df_files.iloc[0].filename)

#     # Split the file path into components using os.sep
#     df_files['year'] = df_files['filename'].apply(lambda x: x.split(os.sep)[-5])
#     df_files['net'] = df_files['filename'].apply(lambda x: x.split(os.sep)[-4])
#     df_files['sta'] = df_files['filename'].apply(lambda x: x.split(os.sep)[-3])
#     df_files['cha'] = df_files['filename'].apply(lambda x: x.split(os.sep)[-2])
#     df_files['julian'] = df_files['filename'].apply(lambda x: x.split(os.sep)[-1].split('.')[6])

#     df_files['starttime'] = df_files['filename'].apply(
#         lambda x: x.split(os.sep)[-1].split('.')[7] if len(x.split(os.sep)[-1].split('.')) > 8 else '00'
#     )

#     df_files['starttime'] = pd.to_datetime(df_files['year'] + df_files['julian'] + df_files['starttime'], format='%Y%j%H')
#     df_files['datetime'] = pd.to_datetime(df_files['year'] + df_files['julian'], format='%Y%j')

#     return df_files


def get_stream_for_selected_period(df_files: pd.DataFrame, starttime: str, endtime: str, channel: str = None) -> Stream:
    """
    Create a stream containing the data of the considered time window. For computation matters, the time window should
    not exceed a few hours in the case of a low sample rate (i.e., 48h at fs=50Hz).

    Parameters
    ----------
    df_files : pd.DataFrame
        A pandas DataFrame containing the existing files for the current station.
    starttime : str
        Starting point of the time window (datetime).
    endtime : str
        End point of the time window (datetime).
    channel : str, optional
        The channel to select.

    Returns
    -------
    Stream
        An ObsPy Stream containing the data.
    """
    df_selection = df_files.set_index('datetime').loc[
        pd.date_range(pd.to_datetime(starttime).floor('D'), pd.to_datetime(endtime).floor('D')).date]

    stream = Stream()

    for filename in df_selection.filename.to_numpy():
        stream += read(filename)

    if channel:
        stream = stream.select(channel=channel)

    try:
        stream = stream.merge(method=1)
    except Exception as e:
        logging.error(f"Error merging stream: {e}")

    stream.trim(UTCDateTime(starttime), UTCDateTime(endtime))
    return stream


def read_flac_file(flac_file):
    """
    Read a FLAC file and access its audio data and metadata, returning an ObsPy Stream.

    Parameters:
    - flac_file: Path to the FLAC file.

    Returns:
    - stream: ObsPy Stream object containing the audio data as a Trace.
    """

    # Read audio data
    audio = AudioSegment.from_file(flac_file, format="flac")

    # Read metadata
    flac = FLAC(flac_file)
    metadata = {key: flac[key] for key in flac.keys()}

    # Convert audio segment to numpy array
    samples = np.array(audio.get_array_of_samples())

    # Create an ObsPy Trace
    trace = Trace(data=samples)
    trace.stats.starttime = UTCDateTime(metadata["starttime"][0])
    trace.stats.sampling_rate = audio.frame_rate 
    trace.stats.channel = metadata["channel"][0]
    trace.stats.station = metadata["sta"][0]
    trace.stats.network = metadata["net"][0]

    # Create an ObsPy Stream and add the trace
    stream = Stream(traces=[trace])

    # Clean up to free memory
    del audio
    gc.collect()

    return stream


def get_stream_for_selected_file(filename: str, channel: str = None, day: str = None) -> Stream:
    """
    Create a stream containing the data of the considered file.

    Parameters
    ----------
    filename : str
        The mseed file to read.
    channel : str, optional
        The channel to select.
    day : str, optional
        The day to trim the stream to (in 'YYYY-MM-DD' format).

    Returns
    -------
    Stream
        An ObsPy Stream containing the data.
    """
    def highpass_filter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)
    
    if '.flac' in filename:
        stream = read_flac_file(filename)
    else:
        stream = read(filename)
        
        if day:
            dt1 = UTCDateTime(day)
            dt2 = dt1 + timedelta(hours=24)
            stream.trim(dt1, dt2)

    if channel:
        stream = stream.select(channel=channel)

    try:
        for tr in stream:
            tr.data = tr.data.astype(float)
            tr.data = highpass_filter(tr.data, cutoff=1.0, fs=tr.stats.sampling_rate)
        # stream = stream.merge()
        # stream.merge(method=1, fill_value=0)
    except Exception as e:
        logging.error(f"Impossible to merge stream: {e}")

    return stream


def get_calibrated_stream(stream: Stream, df_stations: pd.DataFrame) -> Stream:
    """
    Calibrate the given stream using the sensitivity values from the station details DataFrame.

    Parameters
    ----------
    stream : Stream
        The ObsPy Stream object containing the data to be calibrated.
    df_stations : pd.DataFrame
        A pandas DataFrame containing the station details, including sensitivity values.

    Returns
    -------
    Stream
        The calibrated ObsPy Stream.
    """
    df_stations = df_stations.reset_index().set_index(['sta', 'cha'])
    
    for trace in stream:
        try:
            station = trace.stats.station
            channel = trace.stats.channel
            sensitivity = float(df_stations.loc[(station, channel)].sensitivity)
            trace.data = trace.data / sensitivity
        except KeyError:
            logging.error(f"Sensitivity not found for station {station} and channel {channel}.")
            if 'MAHY' in station:
                logging.info("Applying MAHY calibration.")
                # sensitivity=-163.5
                # TO_VOLT = (5/(2**24))

                # trace.data = (trace.data * TO_VOLT) / (10 ** (sensitivity / 20))
                trace.data = trace.data / (4.5*(10**7))
        except Exception as e:
            logging.error(f"Error calibrating trace for station {station} and channel {channel}: {e}")

    return stream
