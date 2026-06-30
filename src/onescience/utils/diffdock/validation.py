from collections.abc import Mapping


SUPPORTED_DATASETS = ("pdbbind", "moad", "generalisation")
UNSUPPORTED_DATASETS = ("pdbsidechain", "distillation")


def _has_cfg(config, key):
    if isinstance(config, Mapping):
        return key in config
    return hasattr(config, key)


def _get_cfg(config, key, default=None):
    if isinstance(config, Mapping):
        return config.get(key, default)
    if hasattr(config, key):
        return getattr(config, key)
    return default


def _support_summary():
    supported = ", ".join(SUPPORTED_DATASETS)
    unsupported = ", ".join(UNSUPPORTED_DATASETS)
    return (
        "Current onescience DiffDock only supports the CGModel main path. "
        f"Supported datasets in this migration: {supported}. "
        f"Unsupported and fail-fast in this migration: {unsupported}, all_atoms/AAModel outside confidence inference, "
        "old_score_model, old_confidence_model outside confidence inference, triple_training."
    )


def _raise_if_errors(context, errors):
    if not errors:
        return
    message_lines = [
        f"{context} is not supported by the current onescience DiffDock migration.",
        _support_summary(),
        "Blocked options:",
    ]
    message_lines.extend(f"- {error}" for error in errors)
    raise NotImplementedError("\n".join(message_lines))


def validate_diffdock_request(
    config,
    *,
    context,
    check_dataset=False,
    check_all_atoms=True,
    check_old_score=False,
    check_old_confidence=False,
    check_triple_training=False,
    confidence_mode=False,
):
    errors = []

    if check_all_atoms and _get_cfg(config, "all_atoms", False) and not confidence_mode:
        errors.append(
            "`all_atoms=true` is only enabled for confidence inference. Score/training paths must keep `all_atoms=false`."
        )

    if check_old_score and _get_cfg(config, "old_score_model", False):
        errors.append(
            "`old_score_model=true` is not migrated. Only the current CGModel score-model path is supported."
        )

    old_confidence_allowed = confidence_mode or _get_cfg(config, "confidence_model_dir", None) is not None
    if (
        check_old_confidence
        and _get_cfg(config, "old_confidence_model", False)
        and not old_confidence_allowed
    ):
        errors.append(
            "`old_confidence_model=true` is only enabled for confidence inference."
        )

    if check_triple_training and _get_cfg(config, "triple_training", False):
        errors.append(
            "`triple_training=true` is not migrated. Use the standard PDBBind/MOAD/generalisation train/val flow only."
        )

    if check_dataset and _has_cfg(config, "dataset"):
        dataset = _get_cfg(config, "dataset")
        if dataset in UNSUPPORTED_DATASETS:
            errors.append(
                f"`dataset={dataset}` is not migrated. Supported datasets are: {', '.join(SUPPORTED_DATASETS)}."
            )
        elif dataset not in SUPPORTED_DATASETS:
            errors.append(
                f"`dataset={dataset}` is not part of the supported CGModel migration. "
                f"Supported datasets are: {', '.join(SUPPORTED_DATASETS)}."
            )

    _raise_if_errors(context, errors)


def validate_training_entrypoint(config, *, context="DiffDock training entrypoint"):
    validate_diffdock_request(
        config,
        context=context,
        check_dataset=True,
        check_all_atoms=True,
        check_old_score=True,
        check_old_confidence=False,
        check_triple_training=True,
    )


def validate_sampling_entrypoint(
    config,
    *,
    context="DiffDock sampling entrypoint",
    include_confidence=False,
    confidence_mode=False,
):
    validate_diffdock_request(
        config,
        context=context,
        check_dataset=False,
        check_all_atoms=True,
        check_old_score=True,
        check_old_confidence=include_confidence or _get_cfg(config, "old_confidence_model", False),
        check_triple_training=False,
        confidence_mode=confidence_mode,
    )


def validate_evaluate_entrypoint(
    config,
    *,
    context="DiffDock evaluate entrypoint",
    confidence_mode=False,
):
    validate_diffdock_request(
        config,
        context=context,
        check_dataset=True,
        check_all_atoms=True,
        check_old_score=True,
        check_old_confidence=True,
        check_triple_training=True,
        confidence_mode=confidence_mode,
    )


def validate_confidence_training_entrypoint(config, *, context="DiffDock confidence training entrypoint"):
    validate_diffdock_request(
        config,
        context=context,
        check_dataset=True,
        check_all_atoms=True,
        check_old_score=False,
        check_old_confidence=False,
        check_triple_training=True,
    )
