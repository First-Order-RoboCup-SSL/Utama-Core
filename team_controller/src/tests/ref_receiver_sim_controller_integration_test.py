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

# TODO: Imcomplete implementation

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(project_root)
sys.path.insert(0, project_root)

from controllers.sim_controller import SimulatorController
from data.referee_receiver import RefereeMessageReceiver
from generated_code.ssl_gc_referee_message_pb2 import Referee

# Example usage:
if __name__ == "__main__":
    receiver = RefereeMessageReceiver()
    sim_control = SimulatorController()
    # TODO: implement place -> stop -> place -> stop sequence
    desired_sequenceA = [Referee.BALL_PLACEMENT_YELLOW, Referee.BALL_PLACEMENT_BLUE, Referee.STOP]
    desired_sequenceB = [Referee.BALL_PLACEMENT_BLUE, Referee.BALL_PLACEMENT_YELLOW, Referee.STOP]

    try:
        while True:
            start_time = time.time()
            if receiver.check_new_message():
                command, des_pos = receiver.get_latest_command()
                # print(command, des_pos, "\n")
                message = receiver.get_latest_message()
                # Yellow card stuff 
                yellow_team_yellow_card_times = list(message.yellow.yellow_card_times)
                blue_team_yellow_card_times = list(message.blue.yellow_card_times)
                yellow_team_robots_removed = []
                blue_team_robots_removed = []
                id = 0
                
                print("Yellow team yellow card times:", yellow_team_yellow_card_times, "\n")
                print("Blue team yellow card times:", blue_team_yellow_card_times, "\n")
                if len(yellow_team_yellow_card_times) == 0:
                    print("!!!!!!!!!")
                                
                if len(yellow_team_yellow_card_times) != 0 and len(yellow_team_yellow_card_times) > len(yellow_team_robots_removed):
                    yellow_team_robots_removed.append(id)
                    for yellow_team_yellow_card_time in yellow_team_yellow_card_times[len(yellow_team_robots_removed) - 1:]: 
                        print("Yellow card detected! (yellow)")
                        # TODO: Implement method to chose which robot to remove
                        sim_control.set_robot_presence(id, team_colour_is_blue=False, should_robot_be_present=False)
                elif (robots_to_add := len(yellow_team_robots_removed) - len(yellow_team_yellow_card_times)) > 0:
                    for _ in range(robots_to_add):
                        sim_control.set_robot_presence(yellow_team_robots_removed[0], team_colour_is_blue=False, should_robot_be_present=True)   
                        yellow_team_robots_removed.pop(0)
            
                if len(blue_team_yellow_card_times) != 0:
                    blue_team_robots_removed.append(id)
                    for blue_team_yellow_card_time in blue_team_yellow_card_times[len(blue_team_robots_removed) - 1:]:
                        print("Yellow card detected! (blue)")
                        sim_control.set_robot_presence(id, team_colour_is_blue=True, should_robot_be_present=False)
                elif (robots_to_add := len(blue_team_robots_removed) - len(blue_team_yellow_card_times)) > 0:
                    for _ in range(robots_to_add):
                        sim_control.set_robot_presence(blue_team_robots_removed[0], team_colour_is_blue=True, should_robot_be_present=True)
                        blue_team_robots_removed.pop(0)
                
                # "Manual" automatic ball placement
                if receiver.check_new_command():
                    if receiver.check_command_sequence(desired_sequenceA or desired_sequenceB):
                        print("Desired sequence detected!")
                        if des_pos != (0.0, 0.0):
                            print("Teleporting ball to", des_pos) 
                            sim_control.teleport_ball(des_pos[0]/1000, des_pos[1]/1000)
            time.sleep(max(0, 0.03334 - (time.time() - start_time)))
    except KeyboardInterrupt:
        print("\nExiting...")