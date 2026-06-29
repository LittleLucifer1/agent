"""Backend adapter packages.

Each ``distillwheel.backends.<name>`` subpackage's top-level
``__init__.py`` should call ``register_adapter(...)`` so that importing
the package suffices to make the adapter discoverable. Entry-point
based loading (see ``distillwheel.core.registry.load_entry_points``)
does this automatically when distillwheel is installed.
"""
