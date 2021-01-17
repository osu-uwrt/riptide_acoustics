#!/usr/bin/env python3
import numpy as np

from algorithm import PingerLocator, NoSampleFoundException
from daq import SampleReader


def broadcastHeading(heading: np.ndarray) -> None:
    # TODO: Implement this method
    print("Ping Heading:", heading)

def shouldRun() -> bool:
    # TODO: Implement this method
    return True

def mainRunLoop(ping_frequency: int) -> None:
    """The main run loop which would search for pings and then calculate the heading of the oncoming ping

    Arguments:
        ping_frequency: The frequency in Hz of the ping to look for
    
    Returns: None
    """
    locator = PingerLocator(ping_frequency)
    reader = SampleReader()
    while shouldRun():
        if reader.ping_found:
            center_microphone_data, x_microphone_data, y_microphone_data = reader.getPingSampleData()
            try:
                heading = locator.calcHeading(np.array(center_microphone_data), np.array(x_microphone_data), np.array(y_microphone_data))
                broadcastHeading(heading)
            except NoSampleFoundException:
                pass
        else:
            search_start_time, search_microphone_data = reader.getSearchSampleData()
            try:
                ping_offset = locator.search_segment(np.array(search_microphone_data))
            except NoSampleFoundException:
                continue
            ping_time = search_start_time + ping_offset
            print("Ping found at {0} seconds".format(ping_time)) 
            reader.setStartTime(ping_time)

if __name__ == "__main__":
    mainRunLoop(25000)