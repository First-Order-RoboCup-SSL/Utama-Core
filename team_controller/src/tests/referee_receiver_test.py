import os
import sys
import time

""" TODO: Traceback (most recent call last):
  File "/home/fredh/robocup_ssl/Robocup/src/tests/referee_receiver_test.py", line 25, in <module>
    if receiver.check_new_message():
  File "/home/fredh/robocup_ssl/Robocup/src/data/referee_receiver.py", line 34, in check_new_message
    serialized_data = self._serialize_relevant_fields(data)
  File "/home/fredh/robocup_ssl/Robocup/src/data/referee_receiver.py", line 23, in _serialize_relevant_fields
    message_copy.ParseFromString(data)
TypeError: a bytes-like object is required, not 'NoneType' """

from team_controller.src.data.referee_receiver import RefereeMessageReceiver

# Example usage:
if __name__ == "__main__":
    receiver = RefereeMessageReceiver()
    try:
        while True:
            start_time = time.time()
            if receiver.check_new_message():
                command, des_pos = receiver.get_latest_command()
                stage_time_left = receiver.get_stage_time_left()
                message = receiver.get_latest_message()
                time_stamp = receiver.get_packet_timestamp()
                next_command = receiver.get_next_command()
                # print(next_command)
                # if next_command != None:
                #     time.sleep(2)
                # print(f"Command: {command}, Designated position: {des_pos}\n")
            time.sleep(max(0, 0.03334 - (time.time() - start_time)))
    except KeyboardInterrupt:
        print("\nExiting...")
