.PHONY: test
test:
	python -m pytest -vv --ignore=lib --ignore=lib64 .
