import numpy as np
import torch

from ml import DeepModel

# Prepare our model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

acoustics_model = DeepModel(4)
acoustics_model.load()
acoustics_model = acoustics_model.to(device)


class InvalidInputException(Exception):
    pass

class NoSampleFoundException(Exception):
    pass

class PingerLocator:
    # Given Constants
    SAMPLING_FREQUENCY = 100000  # Sampling Frequency in Hz
    PINGER_DURATION = 0.004      # Duration of the ping in seconds
    VALID_FREQUENCIES = range(25000, 41000, 1000)
    SAMPLE_SIZE = 2              # Size of the expected sample in seconds
    SPEED_OF_SOUND = 1531        # The speed of sound in whatever medium this code is being run for, in meters/second
    MICROPHONE_DISTANCE = 0.012  # The distance between the microphones in meters

    # Customizable Thresholds
    SEGMENT_SEARCH_SIZE = 0.005  # Size in seconds to do the first search with
    SEARCH_THRESHOLD = 1.5       # The minimum percent difference (expressed as a decimal) for a maximum value to contain a signal

    # Use the model loaded at the beginning of this script for ML
    model = acoustics_model

    def __init__(self, target_frequency: int):
        """Creates new instance of PingerLocator

        Args:
            target_frequency: The frequency in Hz of the ping
        """
        if target_frequency not in self.VALID_FREQUENCIES:
            raise InvalidInputException("Invalid Input Frequency")

        # Setup the class variables
        self.target_frequency = target_frequency

        # Do checks on the presets to make sure that the code will work with those presets
        if not (self.SAMPLE_SIZE / self.SEGMENT_SEARCH_SIZE).is_integer():
            # This check is used for dividing the sample into segments for searching. It must be a whole number to
            # evenly divide the sample up for searching
            raise InvalidInputException("Invalid Preset: Segment size is not evenly divisible by the segment search size")

        if not (self.SEGMENT_SEARCH_SIZE * self.SAMPLING_FREQUENCY).is_integer():
            # This check makes sure that the sample can be evenly divided into the the size of the segment search
            raise InvalidInputException("Invalid Preset: Segment Search Size is not a whole number of samples")

        if self.PINGER_DURATION > self.SEGMENT_SEARCH_SIZE:
            # This check is required since the search can have only up to two segments with a pinger in it. If the
            # search segments are smaller than the ping duration, there can more than three segments with a ping,
            # breaking the logic
            raise InvalidInputException("Invalid Preset: Pinger Duration longer than segment search size")

    def searchSegment(self, microphone_data: np.ndarray) -> float:
        """Searches input microphone data for a pulse and returns its location
        The input microphone data must conform to the constants declared in this class

        Args:
            microphone_data: A numpy array with the microphone data

        Returns:
            The time in seconds of the start of the pulse
        """

        if len(microphone_data) != self.SAMPLE_SIZE * self.SAMPLING_FREQUENCY:
            raise InvalidInputException("The size of the microphone data does not match the expected size")

        # Generate all of the constants for the rough search
        # NOTE: We can force into integer because we checked that it was an integer during intialization
        search_count = int(self.SAMPLE_SIZE / self.SEGMENT_SEARCH_SIZE)
        search_sample_size = int(self.SEGMENT_SEARCH_SIZE * self.SAMPLING_FREQUENCY)
        assert search_count * search_sample_size == self.SAMPLE_SIZE * self.SAMPLING_FREQUENCY

        # Perform the rough search to find the segment with the highest amplitude
        segment_fft_data = np.zeros(search_count)
        for segment_num in range(search_count):
            segment_offset = segment_num * search_sample_size
            search_fft = np.fft.fft(microphone_data[segment_offset:segment_offset+search_sample_size])

            # Search the fft for approximate data
            frequency_offset = round(self.target_frequency * search_sample_size / self.SAMPLING_FREQUENCY)
            segment_fft_data[segment_num] = np.absolute(search_fft[frequency_offset])

        # Get rough location of maximum value
        max_segment = np.argmax(segment_fft_data)

        # TODO: Find out if this is needed
        # Do check to make sure that the maximum value stands above the average noise of the sample
        segment_mean = np.mean(segment_fft_data)
        percent_difference = (segment_fft_data[max_segment] - segment_mean) / ((segment_fft_data[max_segment]+segment_mean)/2)
        if percent_difference < self.SEARCH_THRESHOLD:
            raise NoSampleFoundException("Threshold not met for maximum value")

        # This is the size of the pinger pulse in samples. This is what is going to be the size of the fine ffts
        pinger_sample_size = round(self.PINGER_DURATION * self.SAMPLING_FREQUENCY)

        # The first sample of the first fft in the fine search. This needs to start in the previous segment so it can
        # detect pulses that overlap two segments
        start_sample = (max_segment * search_sample_size) - (search_sample_size * 2)

        # Check to make sure the start sample isn't negative, which is possible if the pulse occurs at the beginning of
        # the microphone data
        if start_sample < 0:
            start_sample = 0


        # The first sample of the last fft to be run during the fine search
        # This value is calculated by getting the last sample of the next segment (since it is possible for the pinger
        # to be in neighboring segments), then subtracting the size of the pinger since that is what is being searched
        end_sample = (max_segment * search_sample_size) + (search_sample_size * 2) - pinger_sample_size

        # Check to make sure that an fft won't be calculated with data that doesn't exist in the microphone data
        if (end_sample + pinger_sample_size) > len(microphone_data):
            end_sample = len(microphone_data) - pinger_sample_size

        maximum_magnitude = 0
        maximum_sample_index = -1
        for sample_offset in range(start_sample, end_sample+1):
            search_fft = np.fft.fft(microphone_data[sample_offset:sample_offset + pinger_sample_size])

            frequency_offset = round(self.target_frequency * pinger_sample_size / self.SAMPLING_FREQUENCY)

            if np.absolute(search_fft[frequency_offset]) > maximum_magnitude:
                maximum_sample_index = sample_offset
                maximum_magnitude = np.absolute(search_fft[frequency_offset])

        # If no fft had a magnitude greater than 0, then something went terribly wrong
        assert maximum_sample_index != -1

        return (maximum_sample_index / self.SAMPLING_FREQUENCY)

    def getPhaseAngle(self, microphone_data: np.ndarray) -> float:
        """Gets the phase angle of the pulse in the given microphone data
        Note: This assumes that microphone_data is trimmed to contain only the pulse

        Args:
            microphone_data: A numpy array with the microphone data

        Returns:
            The phase angle of the inputted data as an angle in radians
        """

        microphone_fft = np.fft.fft(microphone_data)
        frequency_offset = round(self.target_frequency * len(microphone_data) / self.SAMPLING_FREQUENCY)
        return np.angle(microphone_fft[frequency_offset])

    def getPingIndexes(self, microphone_1: np.ndarray, microphone_2: np.ndarray, microphone_3: np.ndarray) -> tuple[int, int]:
        """Locates the best location in the sample to find the phase shift
        This should not be used on large samples, since it does a fine search of the signals without doing a larger search first

        Args:
            microphone_1, microphone_2, microphone_3: The three microphones to find the times to search

        Returns:
            A tuple with the start and end samples that cna be used for a fourier transform and contains the signal
        """

        pinger_sample_size = round(self.PINGER_DURATION * self.SAMPLING_FREQUENCY)

        # Search for pulse in microphone 1
        microphone_1_maximum_magnitude = 0
        microphone_1_maximum_sample_index = -1
        for sample_offset in range(len(microphone_1)-pinger_sample_size):
            search_fft = np.fft.fft(microphone_1[sample_offset:sample_offset + pinger_sample_size])

            frequency_offset = round(self.target_frequency * pinger_sample_size / self.SAMPLING_FREQUENCY)

            if np.absolute(search_fft[frequency_offset]) > microphone_1_maximum_magnitude:
                microphone_1_maximum_magnitude = np.absolute(search_fft[frequency_offset])
                microphone_1_maximum_sample_index = sample_offset

        assert microphone_1_maximum_sample_index != -1

        # Find a start and end index that must contain the ping in all three signals
        # This can be done since we know that all micrphones are within 1/2 a wavelength of each other
        period_length = (1 / self.target_frequency) * self.SAMPLING_FREQUENCY
        start_index = microphone_1_maximum_sample_index + int(period_length)
        end_index = microphone_1_maximum_sample_index + pinger_sample_size - int(period_length)

        # Return the valid indexes which contain the entire signal
        return (start_index, end_index)


    def getPhaseDifference(self, phase_angle_1: float, phase_angle_2: float) -> float:
        """Gets the phase difference of the incoming signal between the two microphones

        Args:
            phase_angle_1: The phase shift of the first microphone's signal in radians
            phase_angle_2: The phase shift of the second microphone's signal in radians

        Returns:
            The phase difference of the incoming signal in radians
        """

        phase_difference = phase_angle_2 - phase_angle_1

        # Wrap around subtraction into valid range for angle calculations
        if phase_difference < -1*np.pi:
            phase_difference = phase_difference + 2*np.pi
        elif phase_difference > np.pi:
            phase_difference = phase_difference - 2*np.pi

        # Return the angle of the oncoming signal
        return phase_difference

    def calcHeading(self, center_microphone_data: np.ndarray, x_microphone_data: np.ndarray, y_microphone_data: np.ndarray) -> np.ndarray:
        """Calculates the heading of an incoming signal from 3 microphones in a right triangle

        Args:
            center_microphone_data: A numpy array with the samples from the center microphone
            x_microphone_data: A numpy array with the samples from the microphone at an x offset of the center
            y_microphone_data: A numpy array with the samples from the microphone at an y offset of the center

        Returns:
            A numpy array with three elements for the normalized x, y, and z heading
        """

        # Get the location of the pulse in the incoming signal
        search_indexes = self.getPingIndexes(center_microphone_data, x_microphone_data, y_microphone_data)

        # Calculate the phase angle of the pulse for each of the three signals 
        center_phase_angle = self.getPhaseAngle(center_microphone_data[search_indexes[0]:search_indexes[1]])
        x_phase_angle = self.getPhaseAngle(x_microphone_data[search_indexes[0]:search_indexes[1]])
        y_phase_angle = self.getPhaseAngle(y_microphone_data[search_indexes[0]:search_indexes[1]])

        # Calculate the x and y phase differences
        x_phase_difference = self.getPhaseDifference(center_phase_angle, x_phase_angle)
        y_phase_difference = self.getPhaseDifference(center_phase_angle, y_phase_angle)

        # Use the ML model to find the approximate signal heading
        model_input = np.array([self.MICROPHONE_DISTANCE * 100, self.target_frequency/10000, x_phase_difference, y_phase_difference])
        predicted = self.model.evaluate(model_input)

        # Normalize the predicted signal heading
        signal_heading = predicted.copy()
        signal_heading /= np.linalg.norm(signal_heading)

        return signal_heading

