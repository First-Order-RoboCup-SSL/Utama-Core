# from controllers.decision_maker import DecisionMaker
from controllers.sim_controller import SimulatorController
# from data.vision_receiver import VisionDataReceiver
# from data.referee_receiver import RefereeMessageReceiver
# import threading

#TODO: untested use the tests for now

# def main():
#     # Initialize components
#     vision_receiver = VisionDataReceiver()
#     referee_receiver = RefereeMessageReceiver()
#     decision_maker = DecisionMaker(port=10302, address='localhost', vision_receiver=vision_receiver)
#     sim_controller = SimulatorController()

#     # Start threads for data reception and decision-making
#     vision_thread = threading.Thread(target=vision_receiver.receive_data)
#     referee_thread = threading.Thread(target=referee_receiver.check_new_message)
#     decision_thread = threading.Thread(target=decision_maker.startup)

#     vision_thread.start()
#     referee_thread.start()
#     decision_thread.start()

#     try:
#         # Wait for threads to complete
#         vision_thread.join()
#         referee_thread.join()
#         decision_thread.join()
#     except KeyboardInterrupt:
#         print("Shutting down...")
#         # Perform any cleanup if necessary

# if __name__ == "__main__":
#     main()
def main():
    controller = SimulatorController()
    controller.teleport_ball(0, 0)
    controller.set_robot_presence(0, team_colour_is_blue=False, should_robot_be_present=True)

if __name__ == "__main__":
    main()