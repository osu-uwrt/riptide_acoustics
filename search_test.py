import numpy as np
import math
import matplotlib.pyplot as plt
import time

def generate_sample():
    sampling_freq = 512000
    length = 2              # Length of sample
    frequency = 40000       # Ping frequency
    ping_length = 0.004     # Duration of ping in seconds
    ping_amplitude = 10

    num_data_points = sampling_freq * length
    start_time = np.random.randint(num_data_points // 10, num_data_points * 9 // 10)

    #print(start_time/sampling_freq)

    data = np.zeros(num_data_points)

    # Fill in the ping in the randomly selected spot
    for i in range(int(ping_length * sampling_freq)):
        val = ping_amplitude * math.cos((2 * math.pi * frequency / sampling_freq * i) + 1.0 )
        data[start_time + i] = val

    # Get fft data of clean signal to do testing
    clean_fft = np.fft.fft(data[start_time:start_time + int(ping_length * sampling_freq)])
    frequency_offset = round(frequency * int(ping_length * sampling_freq) / sampling_freq)
    frequency_data = clean_fft[frequency_offset]

    # Add noise
    data += 1.0 * (2 * np.random.normal(0, 1, num_data_points) - 1)

    # Plot
    #plt.plot(data)
    #plt.show()
    return (data, (start_time/sampling_freq), frequency_data)

class InvalidInputException(Exception):
    pass

class NoSampleFoundException(Exception):
    pass

class PingerLocator:

    # Given Constants
    SAMPLING_FREQUENCY = 512000  # Sampling Frequency in Hz
    PINGER_DURATION = 0.004      # Duration of the ping in seconds
    VALID_FREQUENCIES = range(25000, 41000, 1000)
    SAMPLE_SIZE = 2              # Size of the expected sample in seconds

    # Customizable Thresholds
    SEGMENT_SEARCH_SIZE = 0.005  # Size in seconds to do the first search with
    SEARCH_THRESHOLD = 1.5       # The minimum percent difference (expressed as a decimal) for a maximum value to contain a signal
    NEIGHBOR_THRESHOLD = 0.1     # The maximum percent difference between magnitudes neighboring the maximum signal

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
        #print("Searching For {} khz pinger".format(self.target_frequency/1000))

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

        # Do check to make sure that the maximum value stands above the average noise of the sample
        segment_mean = np.mean(segment_fft_data)
        percent_difference = (segment_fft_data[max_segment] - segment_mean) / ((segment_fft_data[max_segment]+segment_mean)/2)
        if percent_difference < self.SEARCH_THRESHOLD:
            raise NoSampleFoundException("Threshold not met for maximum value")

        #print("Rough Time of Found Signal:", max_segment * self.SEGMENT_SEARCH_SIZE)

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

        maximum_value = 0
        maximum_magnitude = 0
        maximum_sample_index = -1
        for sample_offset in range(start_sample, end_sample+1):
            search_fft = np.fft.fft(microphone_data[sample_offset:sample_offset + pinger_sample_size])

            frequency_offset = round(self.target_frequency * pinger_sample_size / self.SAMPLING_FREQUENCY)

            if np.absolute(search_fft[frequency_offset]) > maximum_magnitude:
                maximum_sample_index = sample_offset
                maximum_magnitude = np.absolute(search_fft[frequency_offset])
                maximum_value = search_fft[frequency_offset]

        # If no fft had a magnitude greater than 0, then something went terribly wrong
        assert maximum_sample_index != -1

        #print("Exact Time of Found Signal: ", maximum_sample_index / self.SAMPLING_FREQUENCY)
        return ((maximum_sample_index / self.SAMPLING_FREQUENCY), maximum_value)


for i in range(1, 101):
    locator = PingerLocator(40000)
    my_sample, generated_time, clean_frequency_data = generate_sample()
    search_time, recovered_frequency_data = locator.search_segment(my_sample)

    if search_time != generated_time:
        print("Time Test Failed! Difference:", search_time - generated_time,"- Samples:", round((search_time - generated_time) * locator.SAMPLING_FREQUENCY))
    if np.angle(clean_frequency_data) != np.angle(recovered_frequency_data):
        percent_difference = abs(np.angle(clean_frequency_data) - np.angle(recovered_frequency_data)) / ((np.angle(clean_frequency_data) + np.angle(recovered_frequency_data)) / 2)
        print("Phase Test Failed! Difference:", np.angle(clean_frequency_data) - np.angle(recovered_frequency_data),"- Percent Difference:", percent_difference * 100)

    if (i % 10) == 0:
        print("Tested", i, "Samples")
