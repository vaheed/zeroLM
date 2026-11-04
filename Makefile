CC ?= cc
PYTHON ?= python3
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?=

.PHONY: all clean build-model check-generated test benchmark pipeline

all: zerolm

zerolm: main.c weights.zlm
	$(CC) $(CFLAGS) -o $@ main.c $(LDFLAGS) -lm

build-model:
	$(PYTHON) build_model.py

check-generated:
	$(PYTHON) build_model.py --check

test: zerolm
	./zerolm --self-test
	$(PYTHON) scripts/evaluate.py --write-report
	$(PYTHON) scripts/weight_audit.py

benchmark: zerolm
	$(PYTHON) scripts/benchmark.py --iterations 100

pipeline: build-model
	$(MAKE) clean
	$(MAKE) all
	$(MAKE) check-generated
	$(MAKE) test
	$(MAKE) benchmark

clean:
	rm -f zerolm zerolm.exe *.o
