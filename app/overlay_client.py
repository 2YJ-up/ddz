from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


WAITING_MODEL_SOURCE = "waiting_for_live_state"


class RecommendationOverlay:
    def __init__(self, args: argparse.Namespace) -> None:
        import tkinter as tk

        self.args = args
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("DDZ Recommendation")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)
        self.root.configure(bg="#111827")
        self.root.geometry(f"{args.width}x{args.height}+{args.x}+{args.y}")

        self.text = tk.Text(
            self.root,
            bg="#111827",
            fg="#F9FAFB",
            insertbackground="#F9FAFB",
            bd=0,
            highlightthickness=0,
            font=("Microsoft YaHei UI", 11),
            padx=12,
            pady=10,
            wrap="word",
        )
        self.text.pack(fill="both", expand=True)
        self.text.configure(state="disabled")

        self.root.bind("<ButtonPress-1>", self._start_drag)
        self.root.bind("<B1-Motion>", self._drag)
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.root.bind("<F9>", lambda _event: self._set_force_lead_override())
        self.root.bind("<F8>", lambda _event: self._clear_live_override())
        self.text.bind("<F9>", lambda _event: self._set_force_lead_override())
        self.text.bind("<F8>", lambda _event: self._clear_live_override())
        self._drag_offset = (0, 0)
        self._last_signature = ""
        self._local_service: Any | None = None

    def run(self) -> None:
        self._refresh()
        self.root.mainloop()

    def _start_drag(self, event: Any) -> None:
        self._drag_offset = (event.x, event.y)

    def _drag(self, event: Any) -> None:
        left = self.root.winfo_pointerx() - self._drag_offset[0]
        top = self.root.winfo_pointery() - self._drag_offset[1]
        self.root.geometry(f"{self.args.width}x{self.args.height}+{left}+{top}")

    def _set_force_lead_override(self) -> None:
        self._write_live_override(
            {
                "enabled": True,
                "force_lead": True,
                "ttl_seconds": self.args.override_ttl_seconds,
                "updated_at_ms": int(time.time() * 1000),
            }
        )

    def _clear_live_override(self) -> None:
        self._write_live_override({"enabled": False, "updated_at_ms": int(time.time() * 1000)})

    def _write_live_override(self, payload: dict[str, Any]) -> None:
        path = Path(self.args.override_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._last_signature = ""

    def _refresh(self) -> None:
        try:
            result = self._request_recommendation()
            text = self._format_response(result)
        except Exception as exc:
            text = f"等待推荐...\n{type(exc).__name__}: {exc}\n\nEsc 关闭；鼠标拖动移动窗口。"
        self._set_text(text)
        self.root.after(int(max(0.2, self.args.interval) * 1000), self._refresh)

    def _request_recommendation(self) -> dict[str, Any]:
        path = Path(self.args.input)
        state = json.loads(path.read_text(encoding="utf-8"))
        state_payload = state.get("state", state)
        warnings = [str(item) for item in state_payload.get("warnings", [])]
        stale_seconds = max(0.0, time.time() - path.stat().st_mtime)
        if stale_seconds > self.args.stale_after:
            return _waiting_response(
                state_payload,
                f"实时识别文件 {stale_seconds:.1f}s 未更新，请确认 live_state_writer 正在运行",
                warnings,
            )
        if state_payload.get("recommendation_ready") is False:
            return _waiting_response(
                state_payload,
                str(state_payload.get("status_reason") or "当前画面暂不可推荐"),
                warnings,
            )
        confidence = _state_confidence(state_payload)
        if confidence < self.args.min_recognition_confidence:
            return _waiting_response(
                state_payload,
                f"识别置信度 {confidence:.2f} 低于 {self.args.min_recognition_confidence:.2f}",
                warnings,
            )
        if len(state_payload.get("self_hand", [])) == 0:
            return _waiting_response(state_payload, "等待识别到当前手牌", warnings)
        payload = {
            "state": state_payload,
            "options": {
                "mode": self.args.mode,
                "n_samples": self.args.n_samples,
                "rollout_per_action": self.args.rollout_per_action,
            },
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.args.url,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
                body = response.read().decode("utf-8")
            result = json.loads(body)
        except urllib.error.HTTPError as exc:
            if exc.code == 400:
                return _waiting_response(
                    state_payload,
                    f"识别状态无效，等待下一帧: {_http_error_detail(exc)}",
                    warnings,
                )
            raise RuntimeError(f"推荐服务返回 HTTP {exc.code}: {_http_error_detail(exc)}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if not self.args.local_fallback:
                raise RuntimeError(f"无法连接推荐服务 {self.args.url}") from exc
            try:
                result = self._request_local_recommendation(state_payload)
            except ValueError as local_exc:
                return _waiting_response(
                    state_payload,
                    f"识别状态无效，等待下一帧: {local_exc}",
                    warnings,
                )
            result.setdefault("state_warnings", [])
            result["state_warnings"] = [
                f"推荐服务暂不可用，已使用 overlay 本地 fallback: {type(exc).__name__}",
                *list(result["state_warnings"]),
            ]
        result["source_frame_id"] = state_payload.get("frame_id", "")
        result["recognition_confidence"] = state_payload.get("recognition_confidence")
        result["self_hand_confidence"] = state_payload.get("self_hand_confidence")
        result["table_confidence"] = state_payload.get("table_confidence")
        result["self_hand_count"] = len(state_payload.get("self_hand", []))
        result["status_reason"] = state_payload.get("status_reason")
        result["writer_pid"] = state_payload.get("writer_pid")
        result["writer_thresholds"] = state_payload.get("writer_thresholds", {})
        result["used_table_cards"] = state_payload.get("used_table_cards", [])
        result["detected_table_cards"] = state_payload.get("detected_table_cards", [])
        result["used_table_action_type"] = state_payload.get("used_table_action_type", "")
        result["used_table_player"] = state_payload.get("used_table_player")
        result["detected_table_player"] = state_payload.get("detected_table_player")
        result["used_table_source"] = state_payload.get("used_table_source", "")
        result["force_lead"] = bool(state_payload.get("force_lead", False))
        result["memory_retained_frames"] = state_payload.get("memory_retained_frames", 0)
        return result

    def _request_local_recommendation(self, state_payload: dict[str, Any]) -> dict[str, Any]:
        if self._local_service is None:
            from src.services.recommendation_service import RecommendationOptions, RecommendationService

            self._local_options_class = RecommendationOptions
            self._local_service = RecommendationService()
        options = self._local_options_class(
            n_samples=self.args.n_samples,
            rollout_per_action=self.args.rollout_per_action,
            mode=self.args.mode,
        )
        return self._local_service.recommend(state_payload, options)

    def _format_response(self, result: dict[str, Any]) -> str:
        model_source = result.get("model_source", "unknown")
        is_waiting = model_source == WAITING_MODEL_SOURCE or str(model_source).startswith("waiting_")
        top_actions = result.get("top_actions", [])[:3]
        best_action = "等待识别" if is_waiting else _format_action(result.get("best_action", []))
        if not is_waiting and not top_actions and not result.get("best_action"):
            best_action = "等待可推荐局面"
        lines = [
            f"当前推荐：{best_action}",
            f"模型：{model_source}",
            "",
        ]
        status_reason = result.get("status_reason")
        if status_reason:
            lines.append(f"状态：{status_reason}")
        writer_pid = result.get("writer_pid")
        writer_thresholds = result.get("writer_thresholds") or {}
        if writer_pid:
            table_threshold = writer_thresholds.get("min_table_confidence", "-")
            lines.append(f"写入器：PID {writer_pid}，桌面阈值 {table_threshold}")
        used_table_cards = result.get("used_table_cards") or []
        detected_table_cards = result.get("detected_table_cards") or []
        action_type = result.get("used_table_action_type") or "-"
        table_source = result.get("used_table_source") or "-"
        used_player = result.get("used_table_player") or "-"
        detected_player = result.get("detected_table_player") or "-"
        force_lead = bool(result.get("force_lead")) or table_source == "forced_lead"
        if force_lead:
            lines.append("上一手：无（自由出牌）；已忽略旧上一手")
            lines.append(f"来源：{table_source}，沿用帧数 {result.get('memory_retained_frames', 0)}")
        elif used_table_cards or detected_table_cards:
            lines.append(
                f"上一手：{used_player} {_format_action(used_table_cards)} ({action_type})；检测 {detected_player} {_format_action(detected_table_cards)}"
            )
            lines.append(f"来源：{table_source}，沿用帧数 {result.get('memory_retained_frames', 0)}")
        recognition_confidence = result.get("recognition_confidence")
        self_hand_confidence = result.get("self_hand_confidence")
        table_confidence = result.get("table_confidence")
        frame_id = result.get("source_frame_id")
        if recognition_confidence is not None or self_hand_confidence is not None or table_confidence is not None or frame_id:
            confidence_text = "-" if recognition_confidence is None else f"{float(recognition_confidence):.2f}"
            self_confidence_text = "-" if self_hand_confidence is None else f"{float(self_hand_confidence):.2f}"
            table_confidence_text = "-" if table_confidence is None else f"{float(table_confidence):.2f}"
            hand_count = result.get("self_hand_count", "-")
            lines.append(
                f"识别：手牌 {hand_count} 张，手牌置信度 {self_confidence_text}，桌面置信度 {table_confidence_text}"
            )
            lines.append(f"帧：{frame_id or '-'}，整体置信度 {confidence_text}")
        if status_reason or recognition_confidence is not None or self_hand_confidence is not None or frame_id:
            lines.append("")
        if top_actions:
            lines.append("Top 3")
        else:
            lines.append("等待可推荐局面")
        for item in top_actions:
            action = _format_action(item.get("action", []))
            win_rate = item.get("win_rate")
            value = f"胜率 {float(win_rate) * 100:.1f}%" if win_rate is not None else f"评分 {item.get('score')}"
            confidence = item.get("confidence", "-")
            lines.append(f"{item.get('rank')}. {action}  {value}  置信度 {confidence}")
            reasons = item.get("reasons", [])
            risks = item.get("risks", [])
            if reasons:
                lines.append("   因：" + "；".join(str(reason) for reason in reasons[:2]))
            if risks:
                lines.append("   险：" + "；".join(str(risk) for risk in risks[:2]))

        warnings = result.get("state_warnings", [])
        if warnings:
            lines.extend(["", "提示"])
            lines.extend(f"- {warning}" for warning in warnings[:3])

        lines.extend(["", f"更新时间：{time.strftime('%H:%M:%S')}  Esc关闭，拖动移动"])
        return "\n".join(lines)

    def _set_text(self, text: str) -> None:
        if text == self._last_signature:
            return
        self._last_signature = text
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self.text.configure(state="disabled")


def _format_action(action: Any) -> str:
    if not action:
        return "过"
    return " ".join(str(card) for card in action)


def _waiting_response(state: dict[str, Any], reason: str, warnings: list[str] | None = None) -> dict[str, Any]:
    merged_warnings = list(warnings or [])
    if reason:
        merged_warnings.insert(0, reason)
    return {
        "best_action": [],
        "top_actions": [],
        "state_warnings": _dedupe(merged_warnings),
        "belief_summary": {"samples_used": 0},
        "model_source": WAITING_MODEL_SOURCE,
        "status_reason": reason,
        "source_frame_id": state.get("frame_id", ""),
        "recognition_confidence": state.get("recognition_confidence"),
        "self_hand_confidence": state.get("self_hand_confidence"),
        "table_confidence": state.get("table_confidence"),
        "self_hand_count": len(state.get("self_hand", [])),
        "writer_pid": state.get("writer_pid"),
        "writer_thresholds": state.get("writer_thresholds", {}),
        "used_table_cards": state.get("used_table_cards", []),
        "detected_table_cards": state.get("detected_table_cards", []),
        "used_table_action_type": state.get("used_table_action_type", ""),
        "used_table_player": state.get("used_table_player"),
        "detected_table_player": state.get("detected_table_player"),
        "used_table_source": state.get("used_table_source", ""),
        "force_lead": bool(state.get("force_lead", False)),
        "memory_retained_frames": state.get("memory_retained_frames", 0),
    }


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _state_confidence(state: dict[str, Any]) -> float:
    if "self_hand_confidence" in state:
        return _float_or_zero(state.get("self_hand_confidence"))
    return _float_or_zero(state.get("recognition_confidence"))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8")
        data = json.loads(body)
        detail = data.get("detail", body) if isinstance(data, dict) else body
        return str(detail)
    except Exception:
        return str(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show v3 recommendation API results in a topmost overlay.")
    parser.add_argument("--input", default="examples/live_test.json", help="RecognizedFrameState JSON file to send.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/recommend", help="Recommendation endpoint.")
    parser.add_argument("--mode", default="douzero_adp", help="Recommendation mode.")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--rollout-per-action", type=int, default=64)
    parser.add_argument("--interval", type=float, default=0.5, help="Refresh interval in seconds.")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--min-recognition-confidence", type=float, default=0.2)
    parser.add_argument("--stale-after", type=float, default=3.0, help="Seconds before the input JSON is treated as stale.")
    parser.add_argument("--override-file", type=Path, default=Path("examples/live_override.json"))
    parser.add_argument("--override-ttl-seconds", type=float, default=10.0)
    parser.add_argument("--local-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--x", type=int, default=40)
    parser.add_argument("--y", type=int, default=80)
    parser.add_argument("--width", type=int, default=470)
    parser.add_argument("--height", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overlay = RecommendationOverlay(args)
    overlay.run()


if __name__ == "__main__":
    main()
