python3 -m black afa
python3 -m isort afa
python3 -m pylint afa
python3 -m pydocstyle --convention=google afa
python3 -m mypy --install-types --strict afa