from datetime import datetime
import multiprocessing
import os
from pathlib import Path
import queue as py_queue
import time

from aerosandbox.numpy import np
import hydra
from omegaconf import DictConfig
import pandas as pd
import psutil
from tqdm import tqdm

from generative_inverse_wing_design.DataGenerator.airfoilrepository import (
    AirfoilRepository,
)
from generative_inverse_wing_design.DataGenerator.appconfig import AppConfig
from generative_inverse_wing_design.DataGenerator.sampler import WingSampler
from generative_inverse_wing_design.DataGenerator.utils.combinationiterator import (
    CombinationIterator,
)

GEOMETRY_PARAM_NAMES: tuple[str, ...] = (
    "chord_ctrl_pos",
    "chord_weight_ctrl",
    "chord_tip",
    "twist_root",
    "twist_tip",
    "twist_mean_pos",
    "spar_sweep",
)
GEOMETRY_DIST_ATTRS: tuple[str, ...] = tuple(f"{name}_dist" for name in GEOMETRY_PARAM_NAMES)


def get_stats() -> str:
    try:
        pid = os.getpid()
        root = psutil.Process(pid)
        child_procs = []
        child_procs += root.children(recursive=True)

        # initialize cpu_percent baseline
        # so subsequent cpu_percent(None) returns meaningful values
        for p in child_procs:
            try:
                p.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        time.sleep(0.5)  # brief sleep to allow cpu_percent to accumulate

        avg_cpu_prc = 0.0
        avg_mem_pct = 0.0
        rss = 0
        vms = 0
        num_procs = len(child_procs) + 1
        num_z_procs = 0
        for p in child_procs:
            try:
                with p.oneshot():
                    # read since paseline
                    avg_cpu_prc += p.cpu_percent(None)
                    avg_mem_pct += p.memory_percent()
                    mem_info = p.memory_info()
                    rss += mem_info.rss / (1024 * 1024)  # Convert to MB
                    vms += mem_info.vms / (1024 * 1024)  # Convert to MB
            except psutil.ZombieProcess:
                num_z_procs += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        root_cpu_prc = 0.0
        root_mem_pct = 0.0
        try:
            root_cpu_prc = root.cpu_percent(None)
            root_mem_pct = root.memory_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        avg_cpu_prc /= max(1, num_procs - 1)
        avg_mem_pct /= max(1, num_procs - 1)
        stats = (
            f"Processes: {num_procs} (Zombies: {num_z_procs}), "
            f"Wrapper CPU%: {root_cpu_prc:.2f}, Wrapper Mem%: {root_mem_pct:.2f},\n"
            f"Child Processes average stats:, "
            f"CPU%: {avg_cpu_prc:.2f}, Mem%: {avg_mem_pct:.2f}, "
            f"RSS: {rss:.2f} MB, VMS: {vms:.2f} MB\n"
        )

    except Exception as e:
        stats = f"Error getting stats: {e}"
    return stats


def write_to_csv(
    file_path: Path, data: dict, error_queue: multiprocessing.Queue, lock: multiprocessing.Lock
) -> None:  # type: ignore
    try:
        wing_df = pd.DataFrame([data])
        with lock:  # type: ignore[attr-defined]
            wing_df.to_csv(file_path, mode="a", header=not os.path.exists(file_path), index=False)
    except Exception as e:
        error_msg = f"Error writing result to file: {e}"
        error_queue.put(error_msg)


def process_wrapper(
    cfg: dict,
    mode: str,
    output_path: Path,
    param_queue: multiprocessing.JoinableQueue,
    error_queue: multiprocessing.Queue,
    lock: multiprocessing.Lock,  # type: ignore
    ctx: multiprocessing.context,  # type: ignore
    process_num: int,  # type: ignore
) -> None:
    """Wrapper function to call the wing processing worker. Times the porocessing time,
    and restarts if the worker exceedas 3 min for solveing the wing
    Args:
        cfg (dict): Configuration for the data generation.
        mode: Data-generation execution mode.
        output_path (Path): Path to the output folder with run ID.
        param_queue (mp.JoinableQueue): Queue to receive airfoil family indices or wing params.
        error_queue (mp.Queue): Queue to send error information
        lock (mp.Lock): Lock for synchronized file writing.
        ctx (mp.Context): Multiprocessing context.
        process_num (int): The process number for randomization.
    """
    from collections import deque
    import gc
    import signal
    import time

    second_param_queue = ctx.JoinableQueue()  # type: ignore
    solved_event = ctx.Event()  # type: ignore
    max_retrys_per_family: int = 3
    process_times: deque[float] = deque(maxlen=30)

    try:
        process = ctx.Process(  # type: ignore
            target=wing_processing_worker,
            args=(
                cfg,
                mode,
                output_path,
                second_param_queue,
                error_queue,
                lock,
                solved_event,
                process_num,
            ),
        )
        process.start()
        wrapper_start_time = time.time()
        while True:
            params, params2 = param_queue.get()
            # stopping signal
            if params is None:
                param_queue.task_done()
                break

            start_time = time.time()
            solved_event.clear()
            second_param_queue.put((params, params2))

            # Wait for the item to be processed
            while not solved_event.is_set():
                time.sleep(0.5)

                # Every 10 minutes, report average processing time
                if time.time() - wrapper_start_time > 600 and len(process_times) > 0:
                    wrapper_start_time = time.time()
                    avg_time = sum(process_times) / len(process_times)
                    error_msg = f"Worker {process_num} average processing time per wing: {avg_time:.2f} seconds.\n\
                                {get_stats()}"
                    error_queue.put(error_msg)

                elapsed_time = time.time() - start_time
                # Check if the processing time has exceeded the limit
                if elapsed_time > 180:  # 3 minutes
                    # Restart the worker process except if max retrys exceeded

                    error_msg = f"Worker {process_num} taking too long ({elapsed_time:.2f} seconds). Restarting worker."
                    error_queue.put(error_msg)
                    process.terminate()
                    process.join(timeout=5)

                    if process.is_alive():
                        try:
                            # multiprocessing.Process.kill() exists on recent Python
                            process.kill()
                        except AttributeError:
                            os.kill(process.pid, signal.SIGKILL)  # type: ignore
                            process.join(timeout=5)
                        except Exception as e:
                            error_queue.put(f"Error terminating worker {process_num}: {e}")
                        finally:
                            # record exit code and free memory
                            exitcode = getattr(process, "exitcode", None)
                            error_queue.put(f"Worker {process_num} exited with code: {exitcode}")
                    try:
                        second_param_queue.cancel_join_thread()
                        second_param_queue.close()
                    except Exception:
                        pass
                    try:
                        gc.collect()
                    except Exception:
                        pass

                    second_param_queue = ctx.JoinableQueue()  # type: ignore
                    solved_event = ctx.Event()  # type: ignore

                    # Restart the worker process
                    process = ctx.Process(  # type: ignore
                        target=wing_processing_worker,
                        args=(
                            cfg,
                            mode,
                            output_path,
                            second_param_queue,
                            error_queue,
                            lock,
                            solved_event,
                            process_num,
                        ),
                    )
                    process.start()
                    start_time = time.time()  # Reset start time after restart

                    max_retrys_per_family -= 1
                    if max_retrys_per_family <= 0:
                        max_retrys_per_family = 3
                        error_msg = f"Worker {process_num} reached max retries\
                              for {params}. Skipping."
                        error_queue.put(error_msg)
                        break

                    # try solve the same airfoil family again
                    solved_event.clear()
                    second_param_queue.put((params, params2))
            # Reset the event for the next item
            param_queue.task_done()
            end_time = time.time()
            process_times.append(end_time - start_time)

    finally:
        second_param_queue.put_nowait((None, None))

        t_end = time.time() + 5
        while time.time() < t_end and process.is_alive():
            time.sleep(0.2)

        # force-terminate process if still alive
        process.terminate()
        process.join(timeout=5)

        if process.is_alive():
            try:
                # multiprocessing.Process.kill() exists on recent Python
                process.kill()
            except AttributeError:
                os.kill(process.pid, signal.SIGKILL)  # type: ignore
                process.join(timeout=5)
            except Exception as e:
                error_queue.put(f"Error terminating worker: {e}")
            finally:
                # record exit code and free memory
                exitcode = getattr(process, "exitcode", None)
                error_queue.put(f"Worker exited with code: {exitcode}")
        try:
            gc.collect()
        except Exception:
            pass


def wing_processing_worker(
    cfg: dict,
    mode: str,
    output_path: Path,
    param_queue: multiprocessing.JoinableQueue,
    error_queue: multiprocessing.Queue,
    lock: multiprocessing.Lock,  # type: ignore
    solved_event: multiprocessing.Event,  # type: ignore
    process_num: int,  # type: ignore
) -> None:
    """Worker function to process a single wing given airfoil family indices.
    Args:
        cfg (dict): Configuration for the data generation.
        mode: Data-generation execution mode.
        output_path (Path): Path to the output folder with run ID.
        param_queue (mp.JoinableQueue): Queue to receive airfoil family indices.
        error_queue (mp.Queue): Queue to send error information
        solved_event (mp.Event): Event to signal wing has been solved.
        lock (mp.Lock): Lock for synchronized file writing.
    """
    import aerosandbox.numpy as np

    from generative_inverse_wing_design.DataGenerator.airfoilrepository import (
        AirfoilRepository,
    )
    from generative_inverse_wing_design.DataGenerator.appconfig import AppConfig

    try:
        app_cfg: AppConfig = AppConfig.from_dict(cfg)
        airfoil_repo: AirfoilRepository = AirfoilRepository(csv_path=app_cfg.run.airfoils_csv_path)
        wing_sampler: WingSampler = WingSampler(app_cfg, airfoil_repo)

        if app_cfg.run.seed is not None:
            np.random.seed(app_cfg.run.seed + process_num)

        if mode != "datagen":
            raise AssertionError(f"Unknown mode {mode} for wing processing worker.")
        output_file_path = output_path / "wings_data.csv"

    except Exception as e:
        error = f"Error initializing worker {process_num}: {e}"
        error_queue.put(error)
    error_queue.put(f"Worker {process_num} initialized successfully.")

    airfoil_familys = None
    while True:
        airfoil_familys, _ = param_queue.get()

        # stopping signal
        if airfoil_familys is None:
            param_queue.task_done()
            return
        try:
            # sample wing
            try:
                wing = None
                wing = wing_sampler.sample_wing(airfoil_familys)
            except Exception as e:
                error = f"Error in worker {process_num} constructing wing for airfoil familys \
                        {airfoil_familys}: {e}"
                error_queue.put(error)
                solved_event.set()  # type: ignore
                continue
            # solve mechanical problem
            try:
                wing.mechanical_wing.solve_for_properties()
            except Exception as e:
                error = f"Error in worker {process_num} solving mechanical wing for airfoil familys \
                        {airfoil_familys}: {e}"
                error_queue.put(error)
                solved_event.set()  # type: ignore
                continue
            # solve aerodynamic and aeroelastic problems
            try:
                wing.solve_all_my_problems()
            except Exception as e:
                error = f"Error in worker {process_num} processing airfoil familys \
                        {airfoil_familys}: {e}"
                error_queue.put(error)
                solved_event.set()  # type: ignore
                continue

            result = wing.to_dict()
            result["airfoil_familys"] = airfoil_familys.tolist()

            # Successfully processed wing
            write_to_csv(output_file_path, result, error_queue, lock)
            solved_event.set()  # type: ignore
        except Exception as e:
            error = f"Unexpected error in worker {process_num} for airfoil familys \
                    {airfoil_familys}: {e}"
            error_queue.put(error)
            solved_event.set()  # type: ignore
            continue
        finally:
            param_queue.task_done()


class Datarunner:
    """A class to run data generation for training the generative model."""

    def __init__(self, cfg: DictConfig):
        self.cfg: AppConfig = AppConfig(cfg)

        if self.cfg.run.seed is not None:
            np.random.seed(self.cfg.run.seed)

        self.num_workers: int = self.cfg.run.num_workers
        self.output_folder_path: Path = self.cfg.run.output_folder_path
        self.airfoils_csv_path: Path = self.cfg.run.airfoils_csv_path

        self.airfoil_repo: AirfoilRepository = AirfoilRepository(csv_path=self.airfoils_csv_path)

        self.combination_iterator: CombinationIterator = CombinationIterator(
            n_samples=len(self.cfg.fixed.ys_over_half_span),
            n_classes=self.airfoil_repo.n_clusters,
            n_combinations=self.cfg.run.num_samples,
            tip_family=self.cfg.run.tip_airfoil_family,
        )

    def _drain_error_queue(
        self, error_queue: multiprocessing.Queue, errors_log_file_path: Path
    ) -> None:
        """Drains the error queue and prints/logs any error messages.
        Args:
            error_queue (mp.Queue): Queue to receive error messages.
        """
        # DON'T use .empty(); use get_nowait() with queue.Empty
        while True:
            try:
                error_msg = error_queue.get_nowait()
            except py_queue.Empty:
                break
            except Exception as e:
                # unexpected queue error — log and break
                with open(errors_log_file_path, "a") as f:
                    f.write(f"{datetime.now()}: Error draining error queue: {e}\n")
                break
            else:
                print(f"{datetime.now()}: {error_msg}")
                with open(errors_log_file_path, "a") as f:
                    f.write(f"{datetime.now()}: {error_msg}\n")

    def run(self) -> None:
        """Runs the data generation process for a specified number of samples."""
        if self.cfg.run.tip_airfoil_family is not None:
            print(f"Using tip airfoil family {self.cfg.run.tip_airfoil_family} for all samples.")

        ctx = multiprocessing.get_context("fork")
        file_lock = ctx.Lock()

        param_queue: multiprocessing.JoinableQueue = ctx.JoinableQueue(
            maxsize=self.num_workers + 1
        )
        error_queue: multiprocessing.Queue = ctx.Queue()

        run_id = int(time.time())
        output_folder_path = self.output_folder_path / f"run_{run_id}"
        os.makedirs(output_folder_path, exist_ok=True)
        errors_folder_path = output_folder_path / "errors"
        os.makedirs(errors_folder_path, exist_ok=True)
        errors_log_file_path = errors_folder_path / "errors.log"

        # --- start worker processes (explicit Processes) ---
        workers = []
        for i in range(self.num_workers):
            p = ctx.Process(  # type: ignore
                target=process_wrapper,
                args=(
                    self.cfg.to_dict(),
                    "datagen",
                    output_folder_path,
                    param_queue,
                    error_queue,
                    file_lock,
                    ctx,
                    i,
                ),
                daemon=False,
            )
            p.start()
            workers.append(p)

        # Right after starting processes drain the error queue once so init messages appear
        time.sleep(0.1)
        self._drain_error_queue(error_queue, errors_log_file_path)

        start_time = time.time()

        try:
            for airfoil_familys in tqdm(
                self.combination_iterator,
                total=len(self.combination_iterator),
                desc="Running data generation",
            ):

                # block if queue is full, but drain errors while waiting
                while param_queue.full():
                    self._drain_error_queue(error_queue, errors_log_file_path)
                    time.sleep(0.1)

                if time.time() - start_time > 600:
                    start_time = time.time()
                    error_msg = f"Main Process Stats:\n\
                                {get_stats()}"
                    error_queue.put(error_msg)

                param_queue.put((airfoil_familys, None))

                # drain new worker-side messages (initialization/errors)
                self._drain_error_queue(error_queue, errors_log_file_path)

                # small backoff to avoid tight loop
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("KeyboardInterrupt received, shutting down...")
            raise
        finally:
            # send stop signals
            for _ in range(self.num_workers):
                param_queue.put((None, None))

            # wait for all queued items to be processed
            param_queue.join()

            # give workers a short grace period to exit normally
            t_end = time.time() + 10
            for p in workers:
                while time.time() < t_end and p.is_alive():
                    time.sleep(0.2)

            # force-terminate any still-alive processes
            for p in workers:
                if p.is_alive():
                    try:
                        p.terminate()
                    except Exception:
                        pass

            # final drain of any remaining error messages
            self._drain_error_queue(error_queue, errors_log_file_path)

    def complete_dataset(self, path_to_current: Path | str) -> None:
        """Completes a partially run data generation process.
        Args:
            path_to_current (Path | str): Path to the current run folder.
        """
        csv_path = Path(path_to_current)
        existing_data = pd.read_csv(csv_path)

        families = existing_data["airfoil_familys"].apply(eval).tolist()
        done_set = set(tuple(family) for family in families)

        # overwrite combination iterator with done set
        self.combination_iterator: CombinationIterator = CombinationIterator(  # type: ignore
            n_samples=len(self.cfg.fixed.ys_over_half_span),
            n_classes=self.airfoil_repo.n_clusters,
            n_combinations=self.cfg.run.num_samples,
            tip_family=self.cfg.run.tip_airfoil_family,
            done_set=done_set,
        )

        self.run()

@hydra.main(version_base=None, config_path="../../../configs/", config_name="dataset_config")
def main(cfg: DictConfig):
    datarunner = Datarunner(cfg)

    datarunner.run()

    # datarunner.complete("data/processed/wings/combined_wings_data.csv")



if __name__ == "__main__":
    main()
