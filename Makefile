.PHONY: test
test:
	python -m pytest --ignore=lib --ignore=lib64 .
