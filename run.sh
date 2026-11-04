#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x ./zerolm ]; then
  if command -v make >/dev/null 2>&1; then
    make
  elif command -v cc >/dev/null 2>&1; then
    cc -O2 -std=c11 -Wall -Wextra -Wpedantic -o zerolm main.c -lm
  else
    echo "Error: install a C11 compiler (gcc or clang) to build ZeroLM." >&2
    exit 1
  fi
fi
exec ./zerolm "$@"
