"""Synchronized task BC entry; legacy independent sharding is unsupported."""
from .train_task_mix_v2 import main
from .task_mix_windows import windows as _windows

if __name__ == "__main__":
    main()
