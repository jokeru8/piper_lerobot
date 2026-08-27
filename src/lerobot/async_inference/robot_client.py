# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    koch_follower,
    make_robot_from_config,
    piper_follower,
    so100_follower,
    so101_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()
        buffer_factory = getattr(self.robot, "make_async_action_buffer", None)
        self.deployment_action_buffer = buffer_factory() if callable(buffer_factory) else None
        self._deployment_buffer_initialized = False
        self._bootstrap_observation_sent = False

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()
        disconnect_error: Exception | None = None
        try:
            self.robot.disconnect()
            self.logger.debug("Robot disconnected")
        except Exception as exc:
            disconnect_error = exc
            self.logger.exception("Robot disconnect failed")
        finally:
            self.channel.close()
            self.logger.debug("Client channel closed")
        if disconnect_error is not None:
            raise disconnect_error

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        if self.deployment_action_buffer is not None:
            queue_size = self.deployment_action_buffer.remaining
            with self.latest_action_lock:
                latest_action = self.latest_action
            timestamps = list(range(latest_action + 1, latest_action + queue_size + 1))
            self.logger.debug(f"Deployment buffer size: {queue_size}")
            return queue_size, timestamps
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Merge a received chunk into the standard queue or Marvin deployment buffer."""
        if self.deployment_action_buffer is not None:
            if not incoming_actions:
                return
            values = torch.stack([action.get_action() for action in incoming_actions]).detach().cpu().numpy()
            if not self._deployment_buffer_initialized:
                self.deployment_action_buffer.integrate_first_chunk(values)
                self._deployment_buffer_initialized = True
            else:
                self.deployment_action_buffer.integrate_new_chunk(values)
            return

        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def _prepare_incoming_action_chunk(self, timed_actions: list[TimedAction]) -> list[TimedAction]:
        """Allow a robot deployment layer to retime/interpolate a complete chunk."""

        prepare = getattr(self.robot, "prepare_policy_action_chunk", None)
        if not callable(prepare) or not timed_actions:
            return timed_actions
        action_tensor = torch.stack([action.get_action() for action in timed_actions])
        processed = prepare(action_tensor)
        if not isinstance(processed, torch.Tensor):
            processed = torch.as_tensor(processed, dtype=action_tensor.dtype)
        if processed.ndim != 2 or processed.shape[0] < 1:
            raise ValueError(f"prepared action chunk must have shape (T, A), got {processed.shape}")
        first = timed_actions[0]
        return [
            TimedAction(
                timestamp=first.get_timestamp() + index * self.config.environment_dt,
                timestep=first.get_timestep() + index,
                action=action,
            )
            for index, action in enumerate(processed)
        ]

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                timed_actions = self._prepare_incoming_action_chunk(timed_actions)
                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose and timed_actions:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    new_range = f"{new_timesteps[0]}:{new_timesteps[-1]}" if new_timesteps else "empty"
                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_range}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        if self.deployment_action_buffer is not None:
            return self.deployment_action_buffer.remaining > 0
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        mapper = getattr(self.robot, "policy_action_to_dict", None)
        if callable(mapper):
            return mapper(action_tensor)
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        get_start = time.perf_counter()
        if self.deployment_action_buffer is not None:
            self.action_queue_size.append(self.deployment_action_buffer.remaining)
            action_array = self.deployment_action_buffer.get_next_action()
            if action_array is None:
                raise RuntimeError("deployment action buffer became empty while popping")
            action_tensor = torch.as_tensor(action_array)
            with self.latest_action_lock:
                action_timestep = self.latest_action + 1
        else:
            # Lock only for queue operations
            with self.action_queue_lock:
                self.action_queue_size.append(self.action_queue.qsize())
                timed_action = self.action_queue.get_nowait()
            action_tensor = timed_action.get_action()
            action_timestep = timed_action.get_timestep()
        get_end = time.perf_counter() - get_start

        action_dict = self._action_tensor_to_action_dict(action_tensor)
        send_policy_action = getattr(self.robot, "send_policy_action", None)
        if callable(send_policy_action):
            _performed_action = send_policy_action(action_dict)
        else:
            _performed_action = self.robot.send_action(action_dict)
        with self.latest_action_lock:
            self.latest_action = action_timestep

        if verbose:
            if self.deployment_action_buffer is not None:
                current_queue_size = self.deployment_action_buffer.remaining
            else:
                with self.action_queue_lock:
                    current_queue_size = self.action_queue.qsize()

            self.logger.debug(f"Action #{action_timestep} performed | Queue size: {current_queue_size}")

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        if self.deployment_action_buffer is not None:
            if not self._deployment_buffer_initialized:
                return not self._bootstrap_observation_sent
            return self.deployment_action_buffer.mark_launch_if_ready()
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            if self.deployment_action_buffer is not None:
                current_queue_size = self.deployment_action_buffer.remaining
                observation.must_go = self.must_go.is_set() and current_queue_size == 0
            else:
                with self.action_queue_lock:
                    observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                    current_queue_size = self.action_queue.qsize()

            sent = self.send_observation(observation)
            if sent and self.deployment_action_buffer is not None and not self._deployment_buffer_initialized:
                self._bootstrap_observation_sent = True

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)
    stop_watcher = None

    def request_stop(mode: str) -> None:
        set_stop_at = getattr(client.robot, "set_stop_at", None)
        halt = getattr(client.robot, "halt_policy_actions", None)
        if callable(set_stop_at):
            set_stop_at(mode)
        if callable(halt):
            halt()
        client.shutdown_event.set()

    if not client.start():
        set_stop_at = getattr(client.robot, "set_stop_at", None)
        if callable(set_stop_at):
            set_stop_at("hold")
        client.stop()
        return

    client.logger.info("Starting action receiver thread...")
    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()

    # Piper exposes Marvin-style s=hold / h=smooth-home hotkeys. Importing
    # this lazily keeps the generic async client independent of Piper.
    if callable(getattr(client.robot, "set_stop_at", None)):
        from lerobot.robots.piper_follower.keyboard_stop import KeyboardStopWatcher

        stop_watcher = KeyboardStopWatcher(request_stop)
        stop_watcher.start()

    try:
        client.control_loop(task=cfg.task)
    except KeyboardInterrupt:
        request_stop("hold")
    finally:
        if stop_watcher is not None:
            stop_watcher.stop()
        stop_error: Exception | None = None
        try:
            client.stop()
        except Exception as exc:
            stop_error = exc
        action_receiver_thread.join(timeout=5.0)
        if action_receiver_thread.is_alive():
            client.logger.warning("Action receiver thread did not stop within 5 seconds")
        if cfg.debug_visualize_queue_size and client.action_queue_size:
            visualize_action_queue_size(client.action_queue_size)
        client.logger.info("Client stopped")
        if stop_error is not None:
            raise stop_error


if __name__ == "__main__":
    async_client()  # run the client
