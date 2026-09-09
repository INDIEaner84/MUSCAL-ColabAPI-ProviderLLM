"""CLI for the audio agent.

    # Executor smoke test, no model, nothing leaves the machine:
    python -m muscal_agent --call "web_search|query=liquid ai lfm"

    # Real path: audio file -> tool call -> result -> spoken answer
    MUSCAL_AGENT_DRY_RUN=0 python -m muscal_agent --wav command.wav

    # Push-to-talk (needs: pip install sounddevice)
    MUSCAL_DESKTOP=1 MUSCAL_AGENT_WRITE=1 python -m muscal_agent --mic
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent import AudioAgent
from .catalog import TOOLS_BY_NAME
from .executor import Policy
from .toolcall import parse_tool_call


def _print_catalog() -> None:
    for name, tool in TOOLS_BY_NAME.items():
        args = ", ".join(p.name for p in tool.params)
        print(f"  {name}({args})  [{tool.risk}]  {tool.description}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MUSCAL audio agent")
    parser.add_argument("--call", help="run a tool call directly, e.g. 'web_search|query=lfm'")
    parser.add_argument("--wav", help="audio file with a spoken command")
    parser.add_argument("--mic", action="store_true", help="push-to-talk loop (needs sounddevice)")
    parser.add_argument("--speak", action="store_true", help="synthesise the answer as wav")
    parser.add_argument("--fmt", default="pipe", choices=["pipe", "pythonic"])
    parser.add_argument("--model", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--answer-model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--no-answer-model", action="store_true", help="use the template fallback")
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument("--doctor", action="store_true", help="probe what this machine can run")
    args = parser.parse_args()

    if args.list_tools:
        _print_catalog()
        return

    if args.doctor:
        from .backends.capabilities import print_report

        print("muscal_agent --doctor")
        print_report()
        return

    policy = Policy.from_env()
    print(f"[policy] dry_run={policy.dry_run} write={policy.allow_write} "
          f"destructive={policy.allow_destructive} tools={policy.allowed_tools or 'all'}")

    if args.call:
        call = parse_tool_call(args.call, fmt=args.fmt)
        result = policy and AudioAgent(fmt=args.fmt, policy=policy).run_tools([call])[0]
        print(json.dumps({"call": call.serialise(args.fmt), **result.__dict__}, ensure_ascii=False, indent=2))
        if not result.ok:
            sys.exit(1)
        return

    agent = AudioAgent(
        args.model,
        None if args.no_answer_model else args.answer_model,
        fmt=args.fmt,
        policy=policy,
    )

    if args.wav:
        import soundfile as sf
        import torch

        wav, sr = sf.read(args.wav, dtype="float32")
        turn = agent.act(torch.from_numpy(wav).unsqueeze(0), sr, speak=args.speak)
        print(json.dumps(turn.as_dict(), ensure_ascii=False, indent=2))
        return

    if args.mic:
        _mic_loop(agent, args)
        return

    parser.print_help()


def _mic_loop(agent: AudioAgent, args) -> None:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        raise SystemExit("push-to-talk needs: pip install sounddevice soundfile")

    print("Push-to-talk: Enter druecken zum Aufnehmen, Enter zum Stoppen. Strg+C beendet.")
    while True:
        try:
            input("[Enter] ")
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")
            return
        print("... sprechen (Enter stoppt)")
        audio = sd.rec(int(30 * 16_000), samplerate=16_000, channels=1, dtype="float32")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")
            return
        sd.stop()
        sf.write("utterance.wav", audio[:, 0], 16_000)
        turn = agent.act(audio[:, 0], 16_000, speak=True)
        print(json.dumps(turn.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
