# StabilityMatrix: Patch logging to print to console for capture.
import sys
import logging

def _apply_logging_patch():
    # Configure Python's root logger to output to stderr at INFO level.
    # Many libraries (torch, diffusers, transformers, etc.) use the logging
    # module but output may be suppressed without a handler configured.
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)

    # Prevent transformers from suppressing its own logging.
    # wgp.py calls transformers.utils.logging.set_verbosity_error() which
    # silences all non-error messages. We neutralize those calls so model
    # loading and download messages remain visible.
    try:
        import transformers.utils.logging as tf_logging
        tf_logging.set_verbosity_error = lambda: None
        tf_logging.set_verbosity_warning = lambda: None
        tf_logging.set_verbosity(logging.INFO)
    except Exception as e:
        print(f"[StabilityMatrix] Failed to patch transformers logging: {e}", file=sys.stderr, flush=True)

    # Monkey-patch Gradio's UI notification functions to also print to console.
    # These only fire for validation/error messages, not generation progress.
    try:
        import gradio as gr
        _orig_info = getattr(gr, 'Info', None)
        _orig_warning = getattr(gr, 'Warning', None)
        _orig_error = getattr(gr, 'Error', None)
        if _orig_info is not None:
            def patched_info(message, *args, **kwargs):
                print(f"[Gradio] {message}", flush=True)
                return _orig_info(message, *args, **kwargs)
            gr.Info = patched_info
        if _orig_warning is not None:
            def patched_warning(message, *args, **kwargs):
                print(f"[Gradio] WARNING: {message}", flush=True)
                return _orig_warning(message, *args, **kwargs)
            gr.Warning = patched_warning
        if _orig_error is not None:
            def patched_error(message, *args, **kwargs):
                print(f"[Gradio] ERROR: {message}", file=sys.stderr, flush=True)
                return _orig_error(message, *args, **kwargs)
            gr.Error = patched_error
    except Exception as e:
        print(f"[StabilityMatrix] Failed to patch Gradio logging: {e}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    _apply_logging_patch()
    target_script = sys.argv[1]
    sys.argv = sys.argv[1:]
    import runpy
    runpy.run_path(target_script, run_name="__main__")