import numpy as np
import random
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from typing import NamedTuple
from collections import deque
from enum import IntEnum


# descending signals with an "importance" metric. This was meant to be used to allow certain pathways to overrule other ones.
# It is not used for that anymore, but is used to weight turning actions as they get modified by other downstream pathways
class CommandWithImportance(NamedTuple):
    left_descending_signal: float
    right_descending_signal: float
    importance: float  # 0->1

    # get an array that can be passed into the step_cpg
    def get_drive(self):
        return np.array([self.left_descending_signal, self.right_descending_signal])


# Encodes a direction to turn when escaping
class EscapeDirection(IntEnum):
    LEFT = -1
    RIGHT = 1


# Overall controller state
class ControllerState(IntEnum):
    SEEKING_ODOR = 0
    TURNING = 1
    RETURNING_HOME = 2


class Controller(BaseController):
    def __init__(self, timestep=1e-4, seed=0):
        from .preprogrammed_steps import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.timestep = timestep
        self.time = 0

        # odor history
        self.ODOR_HIST_LEN = 500
        self.odor_history = deque([], self.ODOR_HIST_LEN)
        self.odor_turn_timer = 0


        # Escape state
        self.escape_timer = 0
        self.turn_timer = 0
        self.escape_direction = EscapeDirection.LEFT
        self.ESCAPE_DURATION = 1000
        self.TURN_DURATION = 400

        # Path integration variables
        self.POS_HIST_LEN = 5000

        self.heading = 0.0
        self.integrated_position = np.zeros(2)
        self.integrated_position_history = deque([], self.POS_HIST_LEN)
        self.pos_inhibit_cooldown = 0

        self.controller_state = ControllerState.SEEKING_ODOR

        self.turning_steps = 0
        self.u_turning_steps = 100  # Number of steps to turn 180°

        self.last_drive = np.zeros(2)

    def get_integrated_position(self) -> np.ndarray:
        return self.integrated_position.copy()

    def get_last_drive(self) -> np.ndarray:
        return self.last_drive.copy()

    def get_odor_taxis(self, obs: Observation) -> CommandWithImportance:
        ODOR_GAIN = -600
        DELTA_MIN = 0.2
        DELTA_MAX = 0.8
        BASE_IMPORTANCE = 0.4
        MAX_IMPORTANCE = 0.9

        if self.odor_turn_timer > 0:
            diff = obs["heading"] - self.odor_target_heading
            if np.abs(diff) > 0.02:
                if self.odor_turn_timer == 500:
                    print("turning...")
                    self.odor_turn_timer -= 1
                magnitude = np.clip(diff, 0.2, 1)
                if diff < 0: 
                    return CommandWithImportance(-0.2, magnitude, 2)
                else:
                    return CommandWithImportance(magnitude, -0.2, 2)
            else: 
                if self.odor_turn_timer == 499:
                    print("straight...")
                self.odor_turn_timer -= 1
                return CommandWithImportance(0.5, 0.5, 2)


        velocity = obs["velocity"]

        I_right = (obs["odor_intensity"][0][1] + obs["odor_intensity"][0][3]) / 2
        I_left = (obs["odor_intensity"][0][0] + obs["odor_intensity"][0][2]) / 2
        I_total = I_left + I_right
        I_norm = min(I_total / 0.0025, 1.0)
        importance = BASE_IMPORTANCE + (MAX_IMPORTANCE - BASE_IMPORTANCE) * I_norm

        self.odor_history.appendleft(I_total)

        asymmetry = (I_left - I_right) / ((I_left + I_right + 1e-6) / 2)
        s = asymmetry * ODOR_GAIN
        turning_bias = np.abs(np.tanh(s))

        if s > 0:
            right_descending_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * turning_bias
            left_descending_signal = DELTA_MAX
        else:
            left_descending_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * turning_bias
            right_descending_signal = DELTA_MAX

        # Damping based on velocity, only apply when odor is strong
        velocity_mag = np.linalg.norm(velocity[:2])
        VELOCITY_THRESHOLD = 0.4
        MAX_DAMPING = 0.5

        if I_total > 0.0005:  # only damp near odor
            damping_factor = 1.0 - min(velocity_mag / VELOCITY_THRESHOLD, 1.0) * MAX_DAMPING
        else:
            damping_factor = 1.0
            # we are too far away from the odor, trigger a turn if it's been decreasing
            if self.odor_turn_timer == 0: 
                differences = np.diff(self.odor_history)
                percent_negative = np.count_nonzero(differences < 1e-7)/len(self.odor_history)
                if percent_negative > 0.6 or np.any(np.abs(self.integrated_position) > 35):
                    print("Going too far, triggering random turn")
                    self.odor_turn_timer = 1500
                    self.odor_target_heading = obs["heading"] - np.pi
                    if self.odor_target_heading < -np.pi:
                        self.odor_target_heading = 2*np.pi + self.odor_target_heading


        left_descending_signal *= damping_factor
        right_descending_signal *= damping_factor

        # Clamp minimum signal to avoid total shutdown
        left_descending_signal = max(left_descending_signal, 0.1)
        right_descending_signal = max(right_descending_signal, 0.1)

        return CommandWithImportance(left_descending_signal, right_descending_signal, importance)


    def pillar_avoidance(
        self, obs: Observation, odor_taxis_command: CommandWithImportance
    ) -> CommandWithImportance:
        MAX_DELTA = 0.8
        MIN_DELTA = 0.2
        IMPORTANCE = 0.9
        GAIN = 40000

        vision = obs["vision"]
        brightness = np.mean(vision, axis=2)
        left_weighted = np.sum(brightness[0])
        right_weighted = np.sum(brightness[1])
        left_front = brightness[0, 620:]
        right_front = brightness[1, :100]
        front_overlap_brightness = np.mean(np.concatenate([left_front, right_front]))
        left_side_brightness = np.mean(brightness[0, 500:620])
        right_side_brightness = np.mean(brightness[1, 100:220])

        velocity_mag = np.linalg.norm(obs["velocity"][:2])
        contact = obs["contact_forces"]

        # Contact forces for front and middle legs
        front_forces = contact[0][:2] + contact[3][:2]
        middle_forces = contact[1][:2] + contact[4][:2]
        total_force = front_forces + middle_forces
        force_mag = np.linalg.norm(total_force)

        # -------- ESCAPE MODE --------
        if self.escape_timer > 0:
            self.escape_timer -= 1
            # Turn slightly away from the contact force vector
            if self.escape_timer == 0:
                # start turn
                self.turn_timer = self.TURN_DURATION

            drive_strength = -np.clip((self.ESCAPE_DURATION - self.escape_timer)*(0.5/20) + 0.1, 0, 0.5)
            return CommandWithImportance(drive_strength, drive_strength, 1.0)

        if self.turn_timer > 0:
            self.turn_timer -= 1
            turn_bias = np.clip(self.escape_direction, -1, 1)
            left = 0.1 + 0.3 * (-turn_bias)
            right = 0.1 + 0.3 * (turn_bias)
            return CommandWithImportance(left, right, 1.0)

        # Trigger escape if strong force and not moving
        if force_mag > 0.3 and velocity_mag < 0.4:
            print("touch recoil")
            self.escape_timer = self.ESCAPE_DURATION
            # Direction to escape: +1 = left leg more contact → escape right
            #                     -1 = right leg more contact → escape left
            escape_vector = total_force
            self.escape_direction = EscapeDirection(
                np.sign(escape_vector[0])
            )  # x-axis as directional hint
            return CommandWithImportance(-0.6, -0.6, 1.0)

        if self.pos_inhibit_cooldown > 0:
            self.pos_inhibit_cooldown -= 1

        # Final fallback: if we've stayed in the same place for a long time then trigger an escape behaviour.
        # Has a cooldown to ensure the position history buffer has enough time to fully turnover.
        if (
            self.pos_inhibit_cooldown == 0
            and self.escape_timer == 0
            and len(self.integrated_position_history) == self.POS_HIST_LEN
        ):
            distance = np.linalg.norm(
                self.integrated_position_history[-1] - self.integrated_position
            )
            if distance < 2:
                print(
                    f"Stayed in place for too long: distance {distance}, escaping... "
                )
                self.escape_timer = self.ESCAPE_DURATION
                self.pos_inhibit_cooldown = self.POS_HIST_LEN + 1

        # Visual emergency:
        if (
            front_overlap_brightness < 10
            or left_side_brightness < 3
            or right_side_brightness < 3
        ) and velocity_mag < 0.2:
            print("visual emergency")
            if left_side_brightness < right_side_brightness:
                left_signal = 0.1
                right_signal = 1.0
            elif right_side_brightness < left_side_brightness:
                left_signal = 1.0
                right_signal = 0.1
            else:
                left_signal, right_signal = (
                    (1.0, 0.1) if random.random() < 0.5 else (0.1, 1.0)
                )

            return CommandWithImportance(
                IMPORTANCE * left_signal
                + (1 - IMPORTANCE) * odor_taxis_command.left_descending_signal,
                IMPORTANCE * right_signal
                + (1 - IMPORTANCE) * odor_taxis_command.right_descending_signal,
                IMPORTANCE,
            )
        

        if odor_taxis_command.importance > 1:
            # we have a really important odor taxis command, so really reduce the importance of visual avoidance signals
            IMPORTANCE = 0.1
        
        # Regular vision-based turning
        diff = left_weighted - right_weighted
        turn_signal = np.tanh(GAIN * diff) + np.random.uniform(-0.05, 0.05)

        left_signal = MAX_DELTA
        right_signal = MAX_DELTA
        if turn_signal > 0:
            left_signal -= (MAX_DELTA - MIN_DELTA) * abs(turn_signal)
        else:
            right_signal -= (MAX_DELTA - MIN_DELTA) * abs(turn_signal)

        left_descending_signal = (
            IMPORTANCE * left_signal
            + odor_taxis_command.importance * odor_taxis_command.left_descending_signal
        )
        right_descending_signal = (
            IMPORTANCE * right_signal
            + odor_taxis_command.importance * odor_taxis_command.right_descending_signal
        )

        return CommandWithImportance(
            left_descending_signal, right_descending_signal, IMPORTANCE
        )

    def path_integration(self, obs: Observation):
        heading = self.heading
        vel = obs["velocity"].copy()

        heading = -heading

        self.integrated_position_history.appendleft(self.integrated_position.copy())

        rot_matrix = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )

        world_frame_vel = vel.ravel() @ rot_matrix
        self.integrated_position += world_frame_vel * self.timestep

    def return_to_home(self) -> np.ndarray:
        match self.controller_state:
            case ControllerState.SEEKING_ODOR:
                # if we're in this, we need to change state to turning as soon as we enter.
                print("Unexpected state in return to home, should not be here")
                self.controller_state = ControllerState.TURNING
            case ControllerState.TURNING:
                # Stop and turn 180° at odor source
                if self.turning_steps < self.u_turning_steps:
                    # Turn in place
                    self.turning_steps += 1
                else:
                    self.controller_state = ControllerState.RETURNING_HOME

                return np.array([0.0, 1.0])
            case ControllerState.RETURNING_HOME:
                # Homing: go back to initial position
                # Compute vector to home
                to_home = -self.integrated_position
                dist_to_home = np.linalg.norm(to_home)

                if dist_to_home < 4:  # Close enough, stop
                    # End the simulation when the fly returns to the drop position after reaching the odor source
                    self.quit = True
                    print("Homing done, quitting simulation.")

                # Compute desired heading
                desired_heading = np.arctan2(to_home[1], to_home[0])
                heading_error = desired_heading - self.heading
                heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
                # Simple proportional controller for turning
                if abs(heading_error) > 0.1:
                    # Turn towards home
                    action = (
                        np.array([0.0, 1.0])
                        if heading_error > 0
                        else np.array([1.0, 0.0])
                    )
                else:
                    # Move forward
                    action = np.array([1.0, 1.0])

                return action
            case _:
                raise RuntimeError(f"Invalid ControllerState {self.controller_state}")

    def get_actions(self, obs: Observation) -> Action:
        self.time += self.timestep

        # Update heading and position from observation
        self.heading = obs["heading"].copy()
        self.path_integration(obs)

        odor_intensity = np.mean(obs["odor_intensity"])

        if odor_intensity > 0.2:
            self.controller_state = ControllerState.TURNING
            
        if self.controller_state == ControllerState.SEEKING_ODOR:
            odor_taxis_command = self.get_odor_taxis(obs)
            combined_command = self.pillar_avoidance(obs, odor_taxis_command)
            drive = combined_command.get_drive()
        else:
            drive = self.return_to_home()

        self.last_drive = drive

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=drive,
        )
        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def done_level(self, obs: Observation):
        return self.quit

    def reset(self, **kwargs):
        """Clean reset of all state"""
        self.cpg_network.reset()
        self.escape_timer = 0
        self.turn_timer = 0
        self.escape_direction = 0.0
        self.heading = 0.0
        self.initial_position = None
        self.reached_target = False
        self.go_home = False
        self.turning_steps = 0
        self.homing_done = False
