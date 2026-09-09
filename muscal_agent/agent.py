"""The audio agent loop: speech in, tool call out, result spoken back.

Pipeline (mirrors what the fine-tune is trained on):

    mic/audio --LFM2.5-Audio (ASR switch)--> "web_search|query=lfm2.5"
                                                    |
                                              executor (policy-gated)
                                                    |
    spoken answer <--LFM2.5 text model (summary)-- tool result

Why the answer step uses a *text* model: the fine-tune teaches the audio model
exactly one behaviour -- speech to function call. Asking the same weights to
also hold a conversation about the result would fight that. Summarising JSON
into one spoken sentence is what a small instruct model is good at.

The critical constraint inherited from Liquid's own recipe: the system prompt
is "Perform ASR." -- not because we want transcription, but because the
llama-liquid-audio-server only accepts a closed allow-list of system prompts
and forces that one at inference time. Train with anything else and the runtime
overrides the fine-tune.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .executor import Executor, Policy, ToolResult
from .toolcall import ToolCall, extract_tool_calls

# Must match the string the runtime will send. See module docstring.
AUDIO_SYSTEM_PROMPT = "Perform ASR."
TTS_SYSTEM_PROMPT = "Perform TTS. Use the US female voice."


@dataclass
class Turn:
    """One interaction: what was heard, what was called, what came back."""

    heard: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    answer: str = ""
    audio_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "heard": self.heard,
            "calls": [c.serialise() for c in self.calls],
            "results": [
                {"tool": r.name, "ok": r.ok, "error": r.error, "output": r.output} for r in self.results
            ],
            "answer": self.answer,
            "audio_path": self.audio_path,
        }


class AudioAgent:
    def __init__(
        self,
        model_id: str = "LiquidAI/LFM2.5-Audio-1.5B",
        answer_model_id: str | None = "LiquidAI/LFM2.5-1.2B-Instruct",
        *,
        fmt: str = "pipe",
        policy: Policy | None = None,
        device: str | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        self.model_id = model_id
        self.answer_model_id = answer_model_id
        self.fmt = fmt
        self.executor = Executor(policy or Policy.from_env())
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None
        self._answer = None
        self._answer_tok = None

    # -- lazy loading ------------------------------------------------------ #

    def load_audio(self):
        if self._model is None:
            from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

            device = self.device or ("cuda" if _cuda_available() else "cpu")
            self._processor = LFM2AudioProcessor.from_pretrained(self.model_id, device=device).eval()
            self._model = LFM2AudioModel.from_pretrained(self.model_id).eval().to(device)
        return self._processor, self._model

    def load_answer_model(self):
        if self._answer is None and self.answer_model_id:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self.device or ("cuda" if _cuda_available() else "cpu")
            self._answer_tok = AutoTokenizer.from_pretrained(self.answer_model_id)
            self._answer = AutoModelForCausalLM.from_pretrained(
                self.answer_model_id, dtype="auto", device_map=device
            )
        return self._answer_tok, self._answer

    # -- steps ------------------------------------------------------------- #

    def speech_to_text(self, wav, sampling_rate: int, *, system_prompt: str = AUDIO_SYSTEM_PROMPT) -> str:
        """The fine-tuned behaviour: audio -> one line of tool-call text."""
        import torch

        processor, model = self.load_audio()
        if not isinstance(wav, torch.Tensor):
            wav = torch.as_tensor(wav)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        from liquid_audio import ChatState

        chat = ChatState(processor)
        chat.new_turn("system")
        chat.add_text(system_prompt)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(wav, sampling_rate)
        chat.end_turn()
        chat.new_turn("assistant")

        out: list[str] = []
        for token in model.generate_sequential(**chat, max_new_tokens=self.max_new_tokens):
            if hasattr(token, "numel") and token.numel() == 1:
                out.append(processor.text.decode(token))
        return "".join(out).strip()

    def plan(self, text: str) -> list[ToolCall]:
        return extract_tool_calls(text, fmt=self.fmt)

    def run_tools(self, calls: list[ToolCall]) -> list[ToolResult]:
        return self.executor.run(calls)

    def summarise(self, heard: str, results: list[ToolResult]) -> str:
        """Turn tool JSON into one spoken sentence."""
        tok, model = self.load_answer_model()
        if model is None:
            return _template_answer(results)

        import torch

        payload = "\n".join(r.as_message(self.executor.policy.max_output_chars) for r in results)
        messages = [
            {
                "role": "system",
                "content": (
                    "Du bist die Stimme eines Assistenten. Antworte in einem kurzen, "
                    "gesprochenen Satz auf Deutsch. Keine Aufzaehlungen, keine Markdown-Formatierung."
                ),
            },
            {"role": "user", "content": f"Anfrage: {heard}\n\nErgebnis der Tools:\n{payload}"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    def speak(self, text: str, out_path: str = "answer.wav", *, voice_prompt: str = TTS_SYSTEM_PROMPT) -> str:
        import torch
        import torchaudio

        processor, model = self.load_audio()
        from liquid_audio import ChatState

        chat = ChatState(processor)
        chat.new_turn("system")
        chat.add_text(voice_prompt)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_text(text)
        chat.end_turn()
        chat.new_turn("assistant")

        audio_out = []
        for token in model.generate_sequential(**chat, max_new_tokens=512):
            if hasattr(token, "numel") and token.numel() > 1:
                audio_out.append(token)
        if not audio_out:
            raise RuntimeError("TTS produced no audio tokens")
        codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
        waveform = processor.decode(codes)
        torchaudio.save(out_path, waveform.cpu()[0], 24_000)
        return out_path

    # -- the whole turn ---------------------------------------------------- #

    def act(self, wav, sampling_rate: int, *, speak: bool = False, audio_out: str = "answer.wav") -> Turn:
        heard = self.speech_to_text(wav, sampling_rate)
        turn = Turn(heard=heard)
        turn.calls = self.plan(heard)
        if not turn.calls:
            turn.answer = heard  # plain transcription, nothing to execute
            return turn
        turn.results = self.run_tools(turn.calls)
        turn.answer = self.summarise(heard, turn.results)
        if speak:
            turn.audio_path = self.speak(turn.answer, audio_out)
        return turn


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _template_answer(results: list[ToolResult]) -> str:
    """Fallback when no answer model is configured."""
    ok = [r for r in results if r.ok]
    if not ok:
        failed = results[0].error if results else "kein Werkzeug ausgefuehrt"
        return f"Das hat nicht geklappt: {failed}"
    first = ok[0]
    if isinstance(first.output, dict) and "results" in first.output:
        top = first.output["results"][:2]
        return "Ich habe gefunden: " + "; ".join(str(r.get("title", "")) for r in top)
    return f"{first.name} ausgefuehrt."
