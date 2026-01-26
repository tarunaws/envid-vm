from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(slots=True)
class OrchestratorInputs:
    job_id: str
    task_selection: Dict[str, Any]
    requested_models: Dict[str, Any]


@dataclass(slots=True)
class Selection:
    enable_label_detection: bool
    enable_text_on_screen: bool
    enable_moderation: bool
    enable_transcribe: bool
    enable_famous_locations: bool
    enable_scene_by_scene: bool
    enable_key_scene: bool
    enable_high_point: bool
    enable_synopsis_generation: bool
    enable_opening_closing: bool
    enable_celebrity_detection: bool
    enable_celebrity_bio_image: bool
    requested_label_model_raw: str
    requested_label_model: str
    requested_text_model: str
    requested_moderation_model: str
    requested_key_scene_model_raw: str
    requested_key_scene_model: str
    label_engine: str
    use_vi_label_detection: bool
    use_local_ocr: bool
    use_local_moderation: bool
    allow_moderation_fallback: bool
    local_moderation_url_override: str
    use_transnetv2_for_scenes: bool
    use_pyscenedetect_for_scenes: bool
    use_clip_cluster_for_key_scenes: bool
    want_shots: bool
    want_vi_shots: bool
    want_any_vi: bool


@dataclass(slots=True)
class PreflightResult:
    selection: Selection
    precheck: Dict[str, Any]


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _as_str(value: Any, default: str) -> str:
    raw = str(value).strip() if value is not None else ""
    return raw or default


def _build_selection(inputs: OrchestratorInputs) -> Selection:
    sel = inputs.task_selection or {}
    requested = inputs.requested_models or {}

    enable_label_detection = _bool(sel.get("enable_label_detection"), True)
    enable_text_on_screen = _bool(sel.get("enable_text_on_screen"), True)
    enable_moderation = _bool(sel.get("enable_moderation"), True)
    enable_transcribe = _bool(sel.get("enable_transcribe"), True)
    enable_famous_locations = _bool(sel.get("enable_famous_locations"), False)
    enable_scene_by_scene = _bool(sel.get("enable_scene_by_scene"), True)
    enable_key_scene = _bool(sel.get("enable_key_scene"), True)
    enable_high_point = _bool(sel.get("enable_high_point"), True)
    enable_synopsis_generation = _bool(sel.get("enable_synopsis_generation"), False)
    enable_opening_closing = _bool(sel.get("enable_opening_closing"), False)
    enable_celebrity_detection = _bool(sel.get("enable_celebrity_detection"), False)
    enable_celebrity_bio_image = _bool(sel.get("enable_celebrity_bio_image"), False)

    requested_label_model_raw = _as_str(
        requested.get("label_detection_model") or sel.get("label_detection_model"), "auto"
    )
    requested_label_model = _as_str(sel.get("label_detection_model_normalized"), requested_label_model_raw)

    requested_text_model = _as_str(requested.get("text_model") or sel.get("text_model"), "tesseract")
    requested_moderation_model = _as_str(
        requested.get("moderation_model") or sel.get("moderation_model"), "nudenet"
    )

    requested_key_scene_model_raw = _as_str(
        requested.get("key_scene_detection_model") or sel.get("key_scene_detection_model"),
        "pyscenedetect_clip_cluster",
    )
    requested_key_scene_model = _as_str(sel.get("key_scene_detection_model_normalized"), requested_key_scene_model_raw)

    label_engine = _as_str(sel.get("label_engine"), "gcp_video_intelligence")
    use_vi_label_detection = _bool(
        sel.get("use_vi_label_detection"), label_engine.lower() in {"gcp_video_intelligence", "vi"}
    )

    use_local_ocr = _bool(sel.get("use_local_ocr"), True)
    use_local_moderation = _bool(sel.get("use_local_moderation"), True)
    allow_moderation_fallback = _bool(sel.get("allow_moderation_fallback"), True)
    local_moderation_url_override = _as_str(sel.get("local_moderation_url_override"), "")

    use_transnetv2_for_scenes = _bool(sel.get("use_transnetv2_for_scenes"), False)
    use_pyscenedetect_for_scenes = _bool(sel.get("use_pyscenedetect_for_scenes"), True)
    use_clip_cluster_for_key_scenes = _bool(sel.get("use_clip_cluster_for_key_scenes"), True)

    want_vi_shots = _bool(sel.get("want_vi_shots"), False)
    want_any_vi = _bool(sel.get("want_any_vi"), want_vi_shots)
    want_shots = _bool(
        sel.get("want_shots"),
        enable_scene_by_scene or enable_key_scene or enable_high_point,
    )

    return Selection(
        enable_label_detection=enable_label_detection,
        enable_text_on_screen=enable_text_on_screen,
        enable_moderation=enable_moderation,
        enable_transcribe=enable_transcribe,
        enable_famous_locations=enable_famous_locations,
        enable_scene_by_scene=enable_scene_by_scene,
        enable_key_scene=enable_key_scene,
        enable_high_point=enable_high_point,
        enable_synopsis_generation=enable_synopsis_generation,
        enable_opening_closing=enable_opening_closing,
        enable_celebrity_detection=enable_celebrity_detection,
        enable_celebrity_bio_image=enable_celebrity_bio_image,
        requested_label_model_raw=requested_label_model_raw,
        requested_label_model=requested_label_model,
        requested_text_model=requested_text_model,
        requested_moderation_model=requested_moderation_model,
        requested_key_scene_model_raw=requested_key_scene_model_raw,
        requested_key_scene_model=requested_key_scene_model,
        label_engine=label_engine,
        use_vi_label_detection=use_vi_label_detection,
        use_local_ocr=use_local_ocr,
        use_local_moderation=use_local_moderation,
        allow_moderation_fallback=allow_moderation_fallback,
        local_moderation_url_override=local_moderation_url_override,
        use_transnetv2_for_scenes=use_transnetv2_for_scenes,
        use_pyscenedetect_for_scenes=use_pyscenedetect_for_scenes,
        use_clip_cluster_for_key_scenes=use_clip_cluster_for_key_scenes,
        want_shots=want_shots,
        want_vi_shots=want_vi_shots,
        want_any_vi=want_any_vi,
    )


def orchestrate_preflight(
    *,
    inputs: OrchestratorInputs,
    precheck_models: Callable[..., Dict[str, Any]] | None = None,
    job_update: Callable[..., Any] | None = None,
    job_step_update: Callable[..., Any] | None = None,
) -> PreflightResult:
    selection = _build_selection(inputs)

    precheck: Dict[str, Any] = {}
    if precheck_models is not None:
        precheck = precheck_models(
            enable_transcribe=selection.enable_transcribe,
            enable_synopsis_generation=selection.enable_synopsis_generation,
            enable_label_detection=selection.enable_label_detection,
            enable_moderation=selection.enable_moderation,
            enable_text_on_screen=selection.enable_text_on_screen,
            enable_key_scene=selection.enable_key_scene,
            enable_scene_by_scene=selection.enable_scene_by_scene,
            enable_famous_locations=selection.enable_famous_locations,
            requested_key_scene_model=selection.requested_key_scene_model,
            requested_label_model=selection.requested_label_model,
            requested_text_model=selection.requested_text_model,
            requested_moderation_model=selection.requested_moderation_model,
            use_local_moderation=selection.use_local_moderation,
            allow_moderation_fallback=selection.allow_moderation_fallback,
            use_local_ocr=selection.use_local_ocr,
            want_any_vi=selection.want_any_vi,
            local_moderation_url_override=selection.local_moderation_url_override,
        )

    if job_update is not None:
        job_update(inputs.job_id, status="preflight", progress=0, message="Preflight completed")
    if job_step_update is not None:
        job_step_update(inputs.job_id, "preflight", status="completed", percent=100, message="Preflight completed")

    return PreflightResult(selection=selection, precheck=precheck)
