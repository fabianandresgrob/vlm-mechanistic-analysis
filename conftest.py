# Root conftest — prevents pytest from collecting test functions from source modules.
collect_ignore_glob = [
    "feature_search/*.py",
    "eva/*.py",
    "chain_of_embedding/*.py",
    "sae_convergence/*.py",
    "vti/*.py",
    "scripts/*.py",
    "analysis/*.py",
]
