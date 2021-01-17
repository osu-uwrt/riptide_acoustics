import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.signal
from datetime import datetime

from algorithm import PingerLocator

model_name = "ML"
num_samples = 1024
ping_frequency = 25000

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
    noiseAmplitude = 1

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

        # Add noise (This is noise in the environment the microphone picks up)
        waveforms += np.random.uniform(-self.noiseAmplitude, self.noiseAmplitude, (self.numOfMics,self.sampleLength))
        waveforms /= np.std(waveforms)

        sos = scipy.signal.butter(1, (self.lowestFrequency, self.highestFrequency), btype='bandpass', analog=False, output='sos', fs=self.sampleRate)
        filtered = scipy.signal.sosfilt(sos, waveforms)

        # Create a digitized sample that is a small percentage of the total incoming signal
        digitized = filtered * self.signal_saturation
        resolution = 2 ** self.adc_resolution_bits

        # Multiply the signal to the range of the resolution and shift it so zero is at resolution/2
        digitized *= (resolution/2)
        digitized += (resolution/2)

        # Force the signal into the given resolution
        digitized = np.floor(digitized)

        # Clip the signal if required
        digitized[digitized > (resolution - 1)] = resolution - 1
        digitized[digitized < 0] = 0

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

if __name__ == "__main__":
    generator = SampleGenerator(ping_frequency, num_samples)
    startTime = datetime.now()
    inputs, label, timeOffsets, cleanSignal = generator.generateSamples()
    elapsedTime = datetime.now() - startTime

    angle_difference = np.zeros(num_samples)
    calculated_heading = [-1] * num_samples

    for sample in range(num_samples):
        locator = PingerLocator(ping_frequency)
        calculated_heading[sample] = locator.calcHeading(inputs[sample][0], inputs[sample][2], inputs[sample][1])
        print("Time Offset:", timeOffsets[sample])
        print("Generated Heading:", label[sample])
        print("Calculated Heading:", calculated_heading[sample])
        
        # Project the vectors onto the x-y plane
        """label[sample][2] = 0
        calculated_heading[sample][2] = 0
        label[sample] /= np.linalg.norm(label[sample])
        calculated_heading[sample] /= np.linalg.norm(calculated_heading[sample])
        assert np.linalg.norm(label[sample]) > 0.999 and np.linalg.norm(label[sample]) < 1.001
        assert np.linalg.norm(calculated_heading[sample]) > 0.999 and np.linalg.norm(calculated_heading[sample]) < 1.001"""

        angle_difference[sample] = np.arccos(np.dot(calculated_heading[sample], label[sample])) * 180/np.pi
        print("Angle Difference (deg):", angle_difference[sample])
        print()

    average_difference = np.average(angle_difference)
    print("Average Signal Difference: ", average_difference)

    print("Median Angle Difference:", np.median(angle_difference))

    max_difference = np.max(angle_difference)
    print("Max Angle Difference:", max_difference)

    truevectorfig = plt.figure()
    truevectorfig.canvas.set_window_title(model_name + ' True Vector Plot')
    truevectorax = truevectorfig.gca(projection='3d', title="True Ping Angle with Length as Angle Difference")

    estvectorfig = plt.figure()
    estvectorfig.canvas.set_window_title(model_name + 'Estimated Vector Plot')
    estvectorax = estvectorfig.gca(projection='3d', title="Est. Ping Angle with Length as Angle Difference")

    x, y, z, u1, v1, w1, u2, v2, w2 = np.zeros((9, num_samples))
    colors = np.zeros((num_samples*3, 3))
    for sample in range(num_samples):
        u1[sample], v1[sample], w1[sample] = label[sample] * angle_difference[sample]
        u2[sample], v2[sample], w2[sample] = calculated_heading[sample] * angle_difference[sample]
        my_color = mpl.colors.hsv_to_rgb(((max_difference - angle_difference[sample])/max_difference * 0.6667, 1, 1))
        colors[sample] = my_color
        colors[2*sample+num_samples] = my_color
        colors[1+2*sample+num_samples] = my_color
    w1 *= -1  # Invert the z axis since the simulation runs with the assumption that it is pointing downwards
    w2 *= -1

    truevectorax.quiver(x, y, z, u1, v1, w1, arrow_length_ratio=0.1, colors=colors)
    estvectorax.quiver(x, y, z, u2, v2, w2, arrow_length_ratio=0.1, colors=colors)

    truemax_axis = np.max((np.max(np.abs(u1)), np.max(np.abs(v1)), np.max(np.abs(w1))))
    truevectorax.set_xlim(-truemax_axis, truemax_axis)
    truevectorax.set_ylim(-truemax_axis, truemax_axis)
    truevectorax.set_zlim(0, truemax_axis)
    truevectorax.set_xlabel('X Axis')
    truevectorax.set_ylabel('Y Axis')
    truevectorax.set_zlabel('Z Axis (Inverted)')

    estmax_axis = np.max((np.max(np.abs(u2)), np.max(np.abs(v2)), np.max(np.abs(w2))))
    estvectorax.set_xlim(-estmax_axis, estmax_axis)
    estvectorax.set_ylim(-estmax_axis, estmax_axis)
    estvectorax.set_zlim(0, estmax_axis)
    estvectorax.set_xlabel('X Axis')
    estvectorax.set_ylabel('Y Axis')
    estvectorax.set_zlabel('Z Axis (Inverted)')


    scatterfig, (elevationax, azimuthax) = plt.subplots(2)
    scatterfig.canvas.set_window_title(model_name + ' Scatter Plots')
    elevationax.set_title("Elevation Angle Difference")
    elevationax.set_xlabel("Angle of Elevation of Oncoming Ping (deg)")
    elevationax.set_ylabel("Angle Difference (deg)")
    azimuthax.set_title("Aximuth Angle Difference")
    azimuthax.set_xlabel("Angle of Azimuth of Oncoming Ping (deg)")
    azimuthax.set_ylabel("Angle Difference (deg)")
    
    elevation_angles = np.zeros(num_samples)
    azimuth_angles = np.zeros(num_samples)
    for sample in range(num_samples):
        vector_height = label[sample][2] * -1  # Invert since it simulates downward
        vector_xy_length = np.sqrt(label[sample][0]**2 + label[sample][1]**2)
        elevation_angles[sample] = (180.0/np.pi) * np.arctan(vector_height/vector_xy_length)
        azimuth_angles[sample] = (180.0/np.pi) * np.arctan2(label[sample][1], label[sample][0])
        if azimuth_angles[sample] < 0:
            azimuth_angles[sample] += 360

    elevationax.scatter(elevation_angles, angle_difference, c=colors[:num_samples])
    azimuthax.scatter(azimuth_angles, angle_difference, c=colors[:num_samples])

    distfig, (ax1, ax2) = plt.subplots(1, 2)
    distfig.canvas.set_window_title(model_name + ' Distribution Plots')
    ax1.boxplot(angle_difference)
    ax1.set_ylabel("Angle Difference (deg)")
    ax2.hist(angle_difference)
    ax2.set_xlabel("Angle Difference (deg)")
    ax2.set_ylabel("Num Samples")
    plt.show()

