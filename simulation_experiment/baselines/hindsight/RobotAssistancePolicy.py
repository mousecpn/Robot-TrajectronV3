#coding=utf-8
from typing import List
from .GoalAssistance import GoalAssistance as Goal
import numpy as np
from . import GoalPredictorAssistance
from .RobotState import RobotState, Action
from . import AssistancePolicy


class RobotAssistancePolicy:
    """
    Robot Assistance Policy Class \n
    Args:
        goals: list of GoalAssistance
        robot_state: RobotState
        print_on_file: True if print on file info, False otherwise
        file: PrintOnFile
    """
    def __init__(self, goals: List[Goal], robot_state: RobotState, print_on_file: bool, file):
        self._assist_policy = AssistancePolicy.AssistancePolicy(robot_state, goals)
        self._goal_predictor = GoalPredictorAssistance.GoalPredictorAssistance(goals) 
        self._robot_state = robot_state
        self._goals = goals

        self._ind_max = 0
        self._weight_sc = 1.0 # TODO: tune this parameter

        # if need to print data to file
        self._print_on_file = print_on_file
        self._file = file


    def update(self, robot_pose, user_action=Action()):
        """
        Update robot assistance policy \n
        Args:
            user_action: Action
        """
        self._assist_policy.robot_state.updateState(robot_pose)
        self._assist_policy.update(user_action)
        v_values,q_values = self._assist_policy.get_values()
        if self._print_on_file:
            self._file.write_with_title(v_values, "v_values")
            self._file.write_with_title(q_values, "q_values")

        self._goal_predictor.update_distribution(v_values, q_values, self._weight_sc)
        first_max, second_max = self._goal_predictor.get_ind_maxes()
        self._ind_max = first_max
        

    def get_action(self, goal_distribution=np.array([]), ig_action=None, **kwargs):
        """
        Get Action \n
        Args:
            goal_distribution
        Return: assisted action Action
        """
        if goal_distribution.size == 0:
            goal_distribution = self._goal_predictor.get_distribution()

        self._policy_twist = self._assist_policy.get_assisted_action(goal_distribution, ig_action, **kwargs)
        
        assisted_action = Action(twist=self._policy_twist,
                                 finger_vel=self._assist_policy.user_action.finger_vel,
                                 switch_mode_to=self._assist_policy.user_action.switch_mode_to)

        return assisted_action


    def visualize_prob(self):
        """
        Visualize goal distribution probability
        """
        goal_distribution = self._goal_predictor.get_distribution()
        print("=====-- Goals --=====")
        for idx, g_prob in enumerate(goal_distribution):
            print("= Goal: " + str(self._goals[idx].getID()) + " with prob = " + str(g_prob))


    def getDistribution(self):
        """
        Get goal distribution \n
        Return: goal distribution
        """
        goal_distribution = self._goal_predictor.get_distribution()
        return goal_distribution


    def getIndexMax(self):
        """
        Get index of max probability \n
        Return: index of max probability
        """
        return self._ind_max


    def get_selected_goal(self):
        """
        Get goal with max probability \n
        Return: goal
        """
        return self._goals[self._ind_max]

