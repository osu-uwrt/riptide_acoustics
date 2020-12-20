import random
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal
import time
from datetime import datetime

# All units are metric. Distance is in meters
class SampleGenerator:
    # Sample generation parameters
    sampleRate = 100000
    sampleLength = 1024

    # Pulse generation parameters
    micSpacing = 0.012
    speedOfSound = 1531
    maxDistance = 20
    numOfMics = 3
    pingDuration = 0.004

    # Noise generation parameters
    noiseAmplitude = 2

    # Band pass filter parameters
    lowestFrequency = 20000
    highestFrequency = 49000

    # Digitization parameters
    adc_resolution_bits = 16
    signal_saturation = 0.5

    # Position of each microphone in XYZ order
    micPositions = np.array([
        [0, 0, 0],
        [0, micSpacing, 0],
        [micSpacing, 0, 0]
    ])

    def __init__(self, pingFrequency, batchSize=0):
        self.pingFrequency = pingFrequency
        self.radiansPerSample = self.pingFrequency / self.sampleRate * 2 * math.pi
        self.batchSize = batchSize

    def generateSample(self, *args):
        # Generate origin of the sound
        origin = np.array([
            random.uniform(-self.maxDistance, self.maxDistance),
            random.uniform(-self.maxDistance, self.maxDistance),
            random.uniform(-self.maxDistance, 0),
        ])

        # Compute distance to each microphone
        distances = np.zeros(self.numOfMics)
        for i in range(self.numOfMics):
            distances[i] = np.linalg.norm(self.micPositions[i] - origin)

        # Compute time offsets in samples caused by distance differences
        timeOffsets = (distances - np.min(distances)) / self.speedOfSound * self.sampleRate

        # Apply the ping to each waveform
        startTime = random.uniform(self.sampleLength * .1, self.sampleLength * .9 - self.pingDuration * self.sampleRate)
        waveforms = np.zeros((self.numOfMics, self.sampleLength))
        for micIndex in range(self.numOfMics):
            micStart = startTime + timeOffsets[micIndex]
            #print("Mic",micIndex,"start index:",int(micStart))
            for i in range(int(self.sampleRate * self.pingDuration)):
                currentIndex = int(micStart) + i
                waveforms[micIndex][currentIndex] = math.sin((currentIndex - micStart) * self.radiansPerSample)

        # Add noise
        waveforms += np.random.uniform(-self.noiseAmplitude, self.noiseAmplitude, (self.numOfMics,self.sampleLength))
        waveforms /= np.std(waveforms)

        noise = np.zeros((self.numOfMics, self.sampleLength))
        noise += np.random.uniform(-self.noiseAmplitude, self.noiseAmplitude, (self.numOfMics,self.sampleLength))
        noise /= np.std(noise)

        sos = scipy.signal.butter(1, (self.lowestFrequency, self.highestFrequency), btype='bandpass', analog=False, output='sos', fs=self.sampleRate)
        filtered = scipy.signal.sosfilt(sos, waveforms)

        # TODO: Add in noise after amplification
        digitized = filtered * self.signal_saturation
        resolution = 2 ** (self.adc_resolution_bits - 1)
        digitized *= resolution
        digitized = np.floor(digitized)
        digitized[digitized > (resolution - 1)] = resolution - 1
        digitized[digitized < (-resolution)] = -resolution

        # Generate label by normalizing vector
        label = origin / np.linalg.norm(origin)

        return digitized, label, (int(startTime + np.max(timeOffsets)), int(startTime) + int(self.sampleRate * self.pingDuration)), waveforms

    def generateSamples(self, size=None):
        if size is None:
            size = self.batchSize

        # Process each sample simultaneously
        results = []
        for _ in range(size):
            results.append(self.generateSample())

        # Output
        return [result[0] for result in results], [result[1] for result in results], [result[2] for result in results], [result[3] for result in results]


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

    def __init__(self, target_frequency):
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

    def search_segment(self, microphone_data):
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

    def get_phase_angle(self, microphone_data):
        """Gets the phase shift of the given microphone data
        Note: This assumes that all of the input data contains the pulse

        Args:
            microphone_data: A numpy array with the microphone data

        Returns:
            The phase angle of the inputted data as an angle in radians
        """

        microphone_fft = np.fft.fft(microphone_data)
        frequency_offset = round(self.target_frequency * len(microphone_data) / self.SAMPLING_FREQUENCY)
        return np.angle(microphone_fft[frequency_offset])

    def get_ping_indexes(self, microphone_1, microphone_2, microphone_3):
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
        period_length = (1 / self.target_frequency) * self.SAMPLING_FREQUENCY
        start_index = microphone_1_maximum_sample_index + int(period_length)
        end_index = microphone_1_maximum_sample_index + pinger_sample_size - int(period_length)

        # Return the valid indexes which contain the entire signal
        return (start_index, end_index)


    def get_signal_angle(self, phase_angle_1, phase_angle_2):
        """Gets the angle of the incoming signal from the first microphone

        Args:
            phase_angle_1: The phase shift of the first microphone's signal in radians
            phase_angle_2: The phase shift of the second microphone's signal in radians

        Returns:
            The angle of the incoming signal in radians
        """

        phase_difference = phase_angle_2 - phase_angle_1
        wavelength = self.SPEED_OF_SOUND / self.target_frequency

        # Wrap around subtraction into valid range for angle calculations
        if phase_difference < -1*np.pi:
            phase_difference = phase_difference + 2*np.pi
        elif phase_difference > np.pi:
            phase_difference = phase_difference - 2*np.pi

        # Calculate the angle of the incoming wave based on p hase shift
        signal_angle_cos = (wavelength * phase_difference) / (2 * np.pi * self.MICROPHONE_DISTANCE)
        
        # Fix the cos of the angle if it is outside of the domain of arccos
        # TODO: Find the error causing the phase difference to be too large for the input
        broken = 0
        if signal_angle_cos > 1:
            signal_angle_cos = 1
            broken = 1
        elif signal_angle_cos < -1:
            signal_angle_cos = -1
            broken = 1

        # Return the angle of the oncoming signal
        return np.arccos(signal_angle_cos), broken

    def calc_heading(self, center_microphone_data, x_microphone_data, y_microphone_data, dummy_data):
        """Calculates the heading of an incoming signal from 3 microphones in a right triangle

        Args:
            center_microphone_data: A numpy array with the samples from the center microphone
            x_microphone_data: A numpy array with the samples from the microphone at an x offset of the center
            y_microphone_data: A numpy array with the samples from the microphone at an y offset of the center

        Returns:
            A tuple with three elements for the normalized x, y, and z heading
        """

        # Get the location in the incoming signal of the pulse
        search_indexes = self.get_ping_indexes(center_microphone_data, x_microphone_data, y_microphone_data)
        # print("Calculated Time Offset:", search_indexes)

        # search_indexes = dummy_data

        # print("Start:", search_indexes[0] - dummy_data[0], "- End:", search_indexes[1] - dummy_data[1])

        """
        plt.subplot(3, 1, 1)
        plt.axvline(search_indexes[0], color='red')
        plt.axvline(search_indexes[1], color='red')
        plt.plot(range(SampleGenerator.sampleLength), center_microphone_data)
        plt.subplot(3, 1, 2)
        plt.plot(range(SampleGenerator.sampleLength), x_microphone_data)
        plt.subplot(3, 1, 3)
        plt.plot(range(SampleGenerator.sampleLength), y_microphone_data)
        plt.suptitle(label[0])
        plt.show()
        """


        # Calculate the phase angle of each of the three signals of the pulse
        center_phase_angle = self.get_phase_angle(center_microphone_data[search_indexes[0]:search_indexes[1]])
        x_phase_angle = self.get_phase_angle(x_microphone_data[search_indexes[0]:search_indexes[1]])
        y_phase_angle = self.get_phase_angle(y_microphone_data[search_indexes[0]:search_indexes[1]])

        # Calculate the x and y angles of the heading
        x_signal_angle, broken1 = self.get_signal_angle(center_phase_angle, x_phase_angle)
        y_signal_angle, broken2 = self.get_signal_angle(center_phase_angle, y_phase_angle)

        broken = broken1 + broken2

        # Calculate the three components of the heading as a unit vector
        signal_x_component = np.cos(x_signal_angle)
        signal_y_component = np.cos(y_signal_angle)
        signal_z_component = -np.sqrt(1 - np.power(signal_x_component, 2) - np.power(signal_y_component, 2))

        # Fix for the signal going outside of the expected bounds
        # TODO: Find the error causing the x and y to have a magnitude > 1
        if np.isnan(signal_z_component):
            broken += 4
            signal_z_component = 0

            # Fix the x and y components so that they have a magnitude equal to 1
            magnitude = np.linalg.norm(np.array([signal_x_component, signal_y_component]))
            signal_x_component = signal_x_component / magnitude
            signal_y_component = signal_y_component / magnitude

        # Create a heading vector with the individual components
        signal_heading = np.array([signal_x_component, signal_y_component, signal_z_component])

        assert np.linalg.norm(signal_heading) > 0.999 and np.linalg.norm(signal_heading) < 1.001

        return (signal_heading, broken)

import sys

if __name__ == "__main__":
    num_samples = 128
    ping_frequency = 25000
    generator = SampleGenerator(ping_frequency, num_samples)
    startTime = datetime.now()
    inputs, label, timeOffsets, cleanSignal = generator.generateSamples()
    elapsedTime = datetime.now() - startTime

    angle_difference = np.zeros(num_samples)
    calculated_heading = [-1] * num_samples
    sample_broken = [-1] * num_samples

    angle_difference_clean = np.zeros(num_samples)

    for sample in range(num_samples):
        locator = PingerLocator(ping_frequency)
        print("Time Offset:", timeOffsets[sample])
        calculated_heading[sample], sample_broken[sample] = locator.calc_heading(inputs[sample][0], inputs[sample][2], inputs[sample][1], timeOffsets[sample])
        print("Generated Heading:", label[sample])
        print("Calculated Heading:", calculated_heading[sample])
        print("Angle Difference (deg):",np.arccos(np.dot(calculated_heading[sample], label[sample])) * 180/np.pi)
        angle_difference[sample] = np.arccos(np.dot(calculated_heading[sample], label[sample])) * 180/np.pi
        print()

    for sample in range(num_samples):
        locator = PingerLocator(ping_frequency)
        calculated_heading_test,_ = locator.calc_heading(cleanSignal[sample][0], cleanSignal[sample][2], cleanSignal[sample][1], timeOffsets[sample])
        angle_difference_clean[sample] = np.arccos(np.dot(calculated_heading_test, label[sample])) * 180/np.pi

    
    average_difference = np.average(angle_difference)
    print("Average Signal Difference: ", average_difference)

    # Find if any of the outliers were caused by broken samples
    Q3 = np.percentile(angle_difference, 75, interpolation = 'midpoint')
    Q1 = np.percentile(angle_difference, 25, interpolation = 'midpoint')

    no_noise_Q3 = np.percentile(angle_difference_clean, 75, interpolation = 'midpoint')
    no_noise_Q1 = np.percentile(angle_difference_clean, 25, interpolation = 'midpoint')

    for sample in range(num_samples):
        if angle_difference[sample] > ((Q3 - Q1) * 1.5 + Q3):
            print("Sample:", sample, "- Outlier:", angle_difference[sample], "- Broken:", sample_broken[sample], "- Heading:", label[sample], "- Calculated:", calculated_heading[sample])
        if angle_difference_clean[sample] > ((no_noise_Q3 - no_noise_Q1) * 3 + no_noise_Q3):
            print("Clean Sample:", sample, "- Outlier:", angle_difference_clean[sample])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.boxplot(angle_difference)
    ax2.boxplot(angle_difference_clean)
    ax3.hist(angle_difference)
    ax4.hist(angle_difference_clean)
    plt.show()

