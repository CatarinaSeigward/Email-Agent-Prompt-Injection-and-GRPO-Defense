# Suppress third-party noise that we can't fix from our side.
# Add filters BEFORE any langgraph/transformers import in this package,
# so they apply to all entry points (python -m src.agent / src.eval / etc.).
import warnings

# langgraph 0.6.x: pending deprecation in its checkpoint serializer.
# We don't call JsonPlusSerializer directly — langgraph triggers it internally
# on import. Nothing to "fix" until langgraph itself bumps the default.
warnings.filterwarnings(
    "ignore",
    message=r".*allowed_objects.*will change.*",
)

# Optional: uncomment if other transient deprecation warnings clutter your
# logs (HuggingFace tokenizers / transformers occasionally emit these too).
# warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
