#!/usr/bin/env python
import argparse
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path


CURR_DIR = Path(__file__).resolve().parent
CWAH_DIR = CURR_DIR.parent
sys.path.append(str(CWAH_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a saved CoELA case log and record a VirtualHome video."
    )
    parser.add_argument("--log", required=True, help="Path to logs_agent_*.pik.")
    parser.add_argument(
        "--dataset-path",
        default="./dataset/test_env_set_help.pik",
        help="Dataset used by the original test.",
    )
    parser.add_argument(
        "--executable-file",
        default="../executable/linux_exec.v2.3.0.x86_64",
        help="VirtualHome Unity executable path, relative to cwah/ by default.",
    )
    parser.add_argument("--base-port", type=int, default=6514)
    parser.add_argument("--output-folder", default="../test_results/videos")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--frame-rate", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--camera-mode",
        nargs="+",
        default=["AUTO"],
        help='VirtualHome camera mode, e.g. AUTO, PERSON_FROM_BACK, or a camera id like "0".',
    )
    parser.add_argument(
        "--modality",
        nargs="+",
        default=["normal"],
        help="Image synthesis modalities to save, e.g. normal depth seg_inst.",
    )
    parser.add_argument(
        "--x-display",
        default="98",
        help="X display id used by the Unity executable.",
    )
    parser.add_argument(
        "--start-xvfb",
        action="store_true",
        help="Start Xvfb on --x-display before launching Unity.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="VirtualHome animation speed multiplier.",
    )
    parser.add_argument(
        "--replay-mode",
        choices=["step", "full"],
        default="step",
        help="Replay step-by-step or submit the whole script at once.",
    )
    parser.add_argument(
        "--no-concat",
        action="store_true",
        help="Do not concatenate step videos with ffmpeg.",
    )
    parser.add_argument(
        "--concat-camera",
        default="0",
        help="Camera subfolder to use when concatenating PNG frames.",
    )
    parser.add_argument(
        "--concat-existing",
        action="store_true",
        help="Only concatenate existing step PNG frames; do not launch Unity.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the replay script without launching Unity.",
    )
    return parser.parse_args()


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_episode_index(env_task_set, saved_info):
    matches = []
    for idx, task in enumerate(env_task_set):
        if task.get("task_id") != saved_info["task_id"]:
            continue
        if task.get("env_id") != saved_info["env_id"]:
            continue
        if task.get("task_name") != saved_info["task_name"]:
            continue
        matches.append(idx)
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one matching dataset episode for "
            f"task_id={saved_info['task_id']}, env_id={saved_info['env_id']}, "
            f"task_name={saved_info['task_name']}; found {matches}."
        )
    return matches[0]


def action_to_script(agent_id, action):
    if action is None:
        return None
    action = action.strip()
    if not action or action.startswith("[send_message]"):
        return None
    return f"<char{agent_id}> {action}"


def build_replay_script(actions):
    script = []
    num_steps = max(len(actions.get(0, [])), len(actions.get(1, [])))
    for step in range(num_steps):
        pieces = []
        for agent_id in sorted(actions):
            agent_actions = actions[agent_id]
            if step >= len(agent_actions):
                continue
            item = action_to_script(agent_id, agent_actions[step])
            if item is not None:
                pieces.append(item)
        if pieces:
            script.append("|".join(pieces))
    return script


def maybe_start_xvfb(display):
    if shutil.which("Xvfb") is None:
        raise RuntimeError(
            "Xvfb is not installed. Install it or run without --start-xvfb "
            "if you already have a working X display."
        )
    proc = subprocess.Popen(
        ["Xvfb", f":{display}", "-screen", "0", "1024x768x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return proc


def display_is_available(display):
    if shutil.which("xdpyinfo") is None:
        return False
    env = os.environ.copy()
    env["DISPLAY"] = f":{display}"
    result = subprocess.run(
        ["xdpyinfo"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        check=False,
    )
    return result.returncode == 0


def render_one(env, script, args, output_folder, prefix):
    return env.comm.render_script(
        script,
        recording=True,
        skip_animation=False,
        output_folder=str(output_folder),
        file_name_prefix=prefix,
        frame_rate=args.frame_rate,
        image_synthesis=args.modality,
        image_width=args.width,
        image_height=args.height,
        camera_mode=args.camera_mode,
        time_scale=args.time_scale,
    )


def concat_step_videos(output_folder, prefix, num_steps):
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found; step clips were generated but not concatenated.")
        return

    clips = []
    for step in range(num_steps):
        clip = output_folder / f"{prefix}_step_{step:03}" / "Action_normal.mp4"
        if clip.exists():
            clips.append(clip)

    if not clips:
        print("No step mp4 clips found to concatenate.")
        return

    concat_list = output_folder / f"{prefix}_concat.txt"
    with open(concat_list, "w") as f:
        for clip in clips:
            f.write(f"file '{clip.resolve()}'\n")

    out_file = output_folder / f"{prefix}.mp4"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(out_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        print(f"Concatenated video: {out_file.resolve()}")
    else:
        print("ffmpeg concat failed; keeping individual step clips.")
        print(result.stderr)


def collect_step_frames(output_folder, prefix, camera):
    frames = []
    for step_dir in sorted(output_folder.glob(f"{prefix}_step_*")):
        camera_dir = step_dir / camera
        if not camera_dir.is_dir():
            continue
        frames.extend(sorted(camera_dir.glob("Action_*_normal.png")))
    return frames


def concat_step_frames(output_folder, prefix, camera, frame_rate):
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found; PNG frames were generated but not converted to mp4.")
        return

    frames = collect_step_frames(output_folder, prefix, camera)
    if not frames:
        print(f"No PNG frames found for prefix={prefix!r}, camera={camera!r}.")
        return

    concat_list = output_folder / f"{prefix}_frames_camera_{camera}.txt"
    with open(concat_list, "w") as f:
        for frame in frames:
            f.write(f"file '{frame.resolve()}'\n")

    out_file = output_folder / f"{prefix}.mp4"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-r",
            str(frame_rate),
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-pix_fmt",
            "yuv420p",
            str(out_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        print(f"Created video from {len(frames)} PNG frames: {out_file.resolve()}")
    else:
        print("ffmpeg PNG-frame conversion failed.")
        print(result.stderr)


def main():
    args = parse_args()
    os.chdir(CWAH_DIR)

    saved_info = load_pickle(args.log)
    env_task_set = load_pickle(args.dataset_path)
    episode_index = find_episode_index(env_task_set, saved_info)
    script = build_replay_script(saved_info["action"])

    prefix = args.prefix or (
        f"task_{saved_info['task_id']}_{saved_info['task_name']}_"
        f"env_{saved_info['env_id']}"
    )
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Loaded log: {args.log}")
    print(
        f"Matched dataset episode index {episode_index}: "
        f"task {saved_info['task_id']} ({saved_info['task_name']}), "
        f"env {saved_info['env_id']}"
    )
    print(f"Replay script length: {len(script)}")
    print(f"Output folder: {output_folder.resolve()}")
    print(f"File prefix: {prefix}")

    if args.dry_run:
        for line in script:
            print(line)
        return

    if args.concat_existing:
        concat_step_frames(output_folder, prefix, args.concat_camera, args.frame_rate)
        return

    xvfb_proc = None
    try:
        if args.start_xvfb or not display_is_available(args.x_display):
            print(f"Starting Xvfb on :{args.x_display}")
            xvfb_proc = maybe_start_xvfb(args.x_display)
            if not display_is_available(args.x_display):
                raise RuntimeError(f"X display :{args.x_display} is still unavailable after starting Xvfb.")

        from envs.unity_environment import UnityEnvironment

        executable_args = {
            "file_name": args.executable_file,
            "x_display": args.x_display,
            "no_graphics": False,
            "timeout_wait": 5000,
        }
        env = UnityEnvironment(
            num_agents=2,
            max_episode_length=max(len(script) + 20, 250),
            env_task_set=env_task_set,
            observation_types=["partial", "partial"],
            agent_goals=["LLM", "LLM"],
            base_port=args.base_port,
            port_id=0,
            executable_args=executable_args,
        )
        try:
            env.reset(task_id=episode_index)
            if args.replay_mode == "full":
                success, message = render_one(env, script, args, output_folder, prefix)
                print(f"render_script success: {success}")
                print(f"message: {message}")
            else:
                rendered = 0
                for step, line in enumerate(script):
                    step_prefix = f"{prefix}_step_{step:03}"
                    print(f"Rendering step {step}: {line}")
                    success, message = render_one(env, [line], args, output_folder, step_prefix)
                    print(f"  success: {success}")
                    if not success:
                        print(f"  message: {message}")
                        raise RuntimeError(f"Replay failed at step {step}: {line}")
                    rendered += 1
                print(f"Rendered {rendered} step clips under {output_folder.resolve()}")
                if not args.no_concat:
                    concat_step_videos(output_folder, prefix, rendered)
                    concat_step_frames(output_folder, prefix, args.concat_camera, args.frame_rate)
        finally:
            env.close()
    finally:
        if xvfb_proc is not None:
            xvfb_proc.terminate()


if __name__ == "__main__":
    main()
