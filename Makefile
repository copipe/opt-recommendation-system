.PHONY: jupyterlab
jupyterlab:
	jupyter lab  --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''

.PHONY: test-all
test-all:
	black src
	mypy src
