from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hyena_distill.demo import DemoConfig, DemoResult, run_demo

__all__ = ["DemoConfig", "DemoResult", "run_demo"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    demo = import_module(".demo", __name__)
    return getattr(demo, name)
