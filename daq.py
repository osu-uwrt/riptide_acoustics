import numpy as np
import uldaq
import time

from ctypes import Array

class SampleReader:
    # DAQ Configuration
    DAQ_INPUT_RANGE = uldaq.Range.BIP1VOLTS         # The range in volts the daq should use
    DAQ_SCAN_OPTIONS = uldaq.ScanOption.DEFAULTIO   # The scan method to use, and how the io should be transferred
    DAQ_FLAGS = uldaq.AInScanFlag.DEFAULT           # Determines processing on analog data
    DAQ_READ_TIMEOUT = -1                           # The timeout for daq scans for BOTH search and ping scan reads, -1 is no timeout
    CHANNEL_SAMPLING_RATE = 100000                  # The sampling rate of the daq. This is enforced so it must be supported by the daq
    PING_PERIOD = 2                                 # How often a ping occurs
    PING_SAMPLE_READ_DURATION_MS = 10               # Duration of the sample to read when processing pings

    CENTER_MICROPHONE_CHANNEL = 0   # The channel on the daq for the center microphone (must be sequential)
    X_MICROPHONE_CHANNEL = 1        # The channel on the daq for the x microphone (must be sequential)
    Y_MICROPHONE_CHANNEL = 2        # The channel on the daq for the y microphone (must be sequential)

    # DAQ Runtime Variables
    daq_device = None               # The DaqDevice for the daq
    ai_device = None                # The analog input subsystem for the daq
    status = uldaq.ScanStatus.IDLE  # Whether or not the daq is in the middle of a scan
    ping_data = None                # The preallocated buffer for ping data
    search_data = None              # The preallocated buffer for search data
    input_mode = None               # The input mode for the daq (set to single ended in init_daq)

    # Search Runtime Variables
    ping_found = False              # Determines if following variables are valid
    ping_start_time = None          # The start time of each ping in an interval of PING_PERIOD (ping_time mod period)
    last_ping_time = None           # The last ping found. Used to determine if pings have been missed
    last_ping_search = False        # If last ping time was from a search. This silences an error that it missed a ping since this is expected during searches

    def __init__(self):
        """Creates new instance of sample reader and initializes daq device

        """
        # Calculated class variables
        self.high_channel = max(self.CENTER_MICROPHONE_CHANNEL, self.X_MICROPHONE_CHANNEL, self.Y_MICROPHONE_CHANNEL)
        self.low_channel = min(self.CENTER_MICROPHONE_CHANNEL, self.X_MICROPHONE_CHANNEL, self.Y_MICROPHONE_CHANNEL)
        self.channel_count = self.high_channel - self.low_channel + 1
        self.ping_channel_sample_read_size = (self.PING_SAMPLE_READ_DURATION_MS / 1000.0) * self.CHANNEL_SAMPLING_RATE
        self.search_channel_sample_read_size = self.PING_PERIOD * self.CHANNEL_SAMPLING_RATE
        
        # Verification of class variables
        if self.channel_count != 3:
            raise RuntimeError("Microphone channels are not sequential")

        if self.low_channel < 0:
            raise RuntimeError("Lowest channel is less than 0")
        
        self.init_daq()

    def init_daq(self, debug: bool = False) -> None:
        """Initializes the DAQ so it is ready to read samples
        Will also do startup checks to make sure that the parameters above are valid for this DAQ

        Returns: None

        Code from uldaq example library: https://pypi.org/project/uldaq/
        """
        # Get descriptors for all of the available DAQ devices.
        devices = uldaq.get_daq_device_inventory(uldaq.InterfaceType.USB)
        number_of_devices = len(devices)
        if number_of_devices == 0:
            raise RuntimeError('Error: No DAQ devices found')

        if debug:
            print('Found', number_of_devices, 'DAQ device(s):')
            for i in range(number_of_devices):
                print('  [', i, '] ', devices[i].product_name, ' (',
                    devices[i].unique_id, ')', sep='')

        if number_of_devices > 1:
            print("WARNING: More than 1 daq present. Choosing first device:", end='')
            print(devices[0].product_name, ' (', devices[0].unique_id, ')', sep='')

        # Create the DAQ device from the descriptor at the specified index.
        self.daq_device = uldaq.DaqDevice(devices[0])

        # Get the AiDevice object and verify that it is valid.
        self.ai_device = self.daq_device.get_ai_device()
        if self.ai_device is None:
            raise RuntimeError('Error: The DAQ device does not support analog '
                               'input')

        # Verify the specified device supports hardware pacing for analog input.
        ai_info = self.ai_device.get_info()
        if not ai_info.has_pacer():
            raise RuntimeError('\nError: The specified DAQ device does not '
                               'support hardware paced analog input')

        # Establish a connection to the DAQ device.
        descriptor = self.daq_device.get_descriptor()
        if debug:
            print('\nConnecting to', descriptor.dev_string, '- please wait...')
        # For Ethernet devices using a connection_code other than the default
        # value of zero, change the line below to enter the desired code.
        self.daq_device.connect(connection_code=0)

        # The default input mode is SINGLE_ENDED.
        self.input_mode = uldaq.AiInputMode.SINGLE_ENDED
        if ai_info.get_num_chans_by_mode(uldaq.AiInputMode.SINGLE_ENDED) <= 0:
            raise RuntimeError("The daq does not supported signle ended analog inputs")

        # Get the number of channels and validate the high channel number.
        number_of_channels = ai_info.get_num_chans_by_mode(self.input_mode)

        if self.high_channel >= number_of_channels:
            raise RuntimeError("Highest channel greater than number of supported channels ({0})".format(number_of_channels))

        # Get a list of supported ranges and validate the range index.
        ranges = ai_info.get_ranges(self.input_mode)
        if self.DAQ_INPUT_RANGE not in ranges:
            raise RuntimeError("Requested input range of {0} is not valid for this daq".format(self.DAQ_INPUT_RANGE.name))

        # Allocate a buffer to receive the data.
        self.ping_data = self.create_float_buffer(self.channel_count, self.ping_channel_sample_read_size)
        self.search_data = self.create_float_buffer(1, self.search_channel_sample_read_size)

        if debug:
            print('\n', descriptor.dev_string, ' ready', sep='')
            print('    Function demonstrated: ai_device.a_in_scan()')
            print('    Channels: ', self.low_channel, '-', self.high_channel)
            print('    Input mode: ', self.input_mode.name)
            print('    Range: ', self.DAQ_INPUT_RANGE.name)
            print('    Samples per channel: ', self.ping_channel_sample_read_size)
            print('    Rate: ', self.CHANNEL_SAMPLING_RATE, 'Hz')
            print('    Scan options:', uldaq.display_scan_options(self.DAQ_SCAN_OPTIONS))

    def close_daq(self) -> None:
        """Closes DAQ so it can be used elsewhere

        Returns: None
        """

        if self.daq_device:
            # Stop the acquisition if it is still running.
            if self.status == uldaq.ScanStatus.RUNNING:
                self.ai_device.scan_stop()
            if self.daq_device.is_connected():
                self.daq_device.disconnect()
            self.daq_device.release()

            self.ai_device = None
            self.daq_device = None

    def getSearchSampleData(self) -> tuple[float, Array[float]]:
        """Obtains a search sample from the DAQ

        Returns: A tuple with the start time of the sample and a numpy array containing the sample
        """
        true_rate = self.ai_device.a_in_scan(self.CENTER_MICROPHONE_CHANNEL, self.CENTER_MICROPHONE_CHANNEL, self.input_mode,
                                   self.DAQ_INPUT_RANGE, self.search_channel_sample_read_size,
                                   self.CHANNEL_SAMPLING_RATE, self.DAQ_SCAN_OPTIONS, self.DAQ_FLAGS, self.search_data)
        search_start_time = time.time()
        
        if true_rate != self.CHANNEL_SAMPLING_RATE:
            raise RuntimeError("The DAQ failed to read the data at the requested sampling rate")

        self.ai_device.scan_wait(uldaq.WaitType.WAIT_UNTIL_DONE, self.DAQ_READ_TIMEOUT)
        self.status, transfer_status = self.ai_device.get_scan_status()

        if transfer_status.current_scan_count != self.search_channel_sample_read_size:
            raise RuntimeError("Transfer finished with partially filled buffer")

        if transfer_status.current_scan_count != transfer_status.current_total_count:
            raise RuntimeError("Not all channels were fully read")

        return search_start_time, self.search_data

    def setStartTime(self, found_start_time: float) -> None:
        """Sets the start time to start reading pings from

        Arguments:
            found_start_time: The start time of the found ping in seconds
        
        Returns: None
        """
        self.ping_start_time = found_start_time % self.PING_PERIOD
        self.last_ping_time = found_start_time
        self.ping_found = True
        self.last_ping_search = True

    def getPingSampleData(self) -> tuple[Array[float], Array[float], Array[float]]:
        """Obtains a ping sample from the DAQ
        Requires that the start time be set previously

        Returns: A tuple containing the arrays for the x, y, and center microphone data
        """
        if not self.ping_found:
            raise RuntimeError("Attempting to find ping without a start time specified")

        if not self.last_ping_search and self.last_ping_time + self.PING_PERIOD < time.time():
            print("WARNING: A ping has been missed, running {0} pings behind", (time.time() - self.last_ping_time) // self.PING_PERIOD)
        self.last_ping_search = False

        current_time = time.time()
        next_ping = current_time - (time.time() % self.PING_PERIOD) + self.PING_PERIOD + self.ping_start_time
        sleep_time = next_ping - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

        true_rate = self.ai_device.a_in_scan(self.low_channel, self.high_channel, self.input_mode,
                                   self.DAQ_INPUT_RANGE, self.ping_channel_sample_read_size,
                                   self.CHANNEL_SAMPLING_RATE, self.DAQ_SCAN_OPTIONS, self.DAQ_FLAGS, self.ping_data)
        
        if true_rate != self.CHANNEL_SAMPLING_RATE:
            raise RuntimeError("The DAQ failed to read the data at the requested sampling rate")

        self.ai_device.scan_wait(uldaq.WaitType.WAIT_UNTIL_DONE, self.DAQ_READ_TIMEOUT)
        self.status, transfer_status = self.ai_device.get_scan_status()

        if transfer_status.current_scan_count != self.ping_channel_sample_read_size:
            raise RuntimeError("Transfer finished with partially filled buffer")

        if transfer_status.current_scan_count * 4 != transfer_status.current_total_count:
            raise RuntimeError("Not all channels were fully read")

        center_microphone_data = np.array(self.ping_data[self.CENTER_MICROPHONE_CHANNEL - self.low_channel::3])
        x_microphone_data = np.array(self.ping_data[self.X_MICROPHONE_CHANNEL - self.low_channel::3])
        y_microphone_data = np.array(self.ping_data[self.Y_MICROPHONE_CHANNEL - self.low_channel::3])

        return (center_microphone_data, x_microphone_data, y_microphone_data)