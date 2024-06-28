IMAGE_NAME=spice_ev_simulator
TAG=latest

ui:
	streamlit run ui.py	

format-all:
	poetry run pre-commit run --all-files

install:
	poetry install --no-root && poetry run pre-commit install

run:
	./generate.py --config ./b_on/generate.cfg
	./simulate.py --config ./b_on/simulate.cfg b_on/autogen/scenario.json
	poetry run streamlit run ui.py

run-streamlit:
	poetry run streamlit run simulator.py

test:
	poetry run pytest tests

build:
	docker build -t $(IMAGE_NAME)_final:$(TAG) --target=final .
	docker build -t $(IMAGE_NAME)_test:$(TAG) --target=test .

test-docker:
	docker run --rm -it $(IMAGE_NAME)_test:$(TAG)

run-docker:
	docker run --rm -it $(IMAGE_NAME)_final:$(TAG)

.PHONY: format run build build-mac run-docker
