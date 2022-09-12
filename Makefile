# https://gdevops.gitlab.io/tuto_project/software_quality/black/black.html#black-in-a-makefile

black:
	pre-commit run black

check_all_files:
	pre-commit run --all-files

test:
	pytest tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y .
