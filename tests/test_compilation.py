#######################################################################
# Test Compilation: Verify all chapter files import without errors
#
# This test suite imports each of the Python files from Chapters 1-13
# to ensure they compile correctly with the current Python version
# and dependencies. We use importlib to dynamically import each module
# and catch any syntax or import errors.
#######################################################################

import sys
import os
import importlib.util
import pytest

# Add the repo root to the path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# All chapter files to test
CHAPTER_FILES = [
    ("chapter01", "tic_tac_toe.py"),
    ("chapter02", "ten_armed_testbed.py"),
    ("chapter03", "grid_world.py"),
    ("chapter04", "car_rental.py"),
    ("chapter04", "gamblers_problem.py"),
    ("chapter04", "grid_world.py"),
    ("chapter05", "blackjack.py"),
    ("chapter05", "infinite_variance.py"),
    ("chapter06", "cliff_walking.py"),
    ("chapter06", "maximization_bias.py"),
    ("chapter06", "random_walk.py"),
    ("chapter06", "windy_grid_world.py"),
    ("chapter07", "random_walk.py"),
    ("chapter08", "expectation_vs_sample.py"),
    ("chapter08", "maze.py"),
    ("chapter08", "trajectory_sampling.py"),
    ("chapter09", "random_walk.py"),
    ("chapter09", "square_wave.py"),
    ("chapter10", "access_control.py"),
    ("chapter10", "mountain_car.py"),
    ("chapter11", "counterexample.py"),
    ("chapter12", "mountain_car.py"),
    ("chapter12", "random_walk.py"),
    ("chapter13", "short_corridor.py"),
]


@pytest.mark.parametrize("chapter,filename", CHAPTER_FILES,
                         ids=[f"{ch}/{fn}" for ch, fn in CHAPTER_FILES])
def test_chapter_file_compiles(chapter, filename):
    """Test that each chapter file can be imported (compiled) without errors."""
    filepath = os.path.join(REPO_ROOT, chapter, filename)

    # Verify the file exists
    assert os.path.isfile(filepath), f"File not found: {filepath}"

    # Try to compile the file (this checks syntax without executing)
    with open(filepath, 'r') as f:
        source = f.read()

    try:
        compile(source, filepath, 'exec')
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {chapter}/{filename}: {e}")


@pytest.mark.parametrize("chapter,filename", CHAPTER_FILES,
                         ids=[f"{ch}/{fn}" for ch, fn in CHAPTER_FILES])
def test_chapter_file_importable(chapter, filename):
    """Test that each chapter file can be loaded as a module without errors."""
    filepath = os.path.join(REPO_ROOT, chapter, filename)
    module_name = f"{chapter}_{filename.replace('.py', '')}"

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    assert spec is not None, f"Could not create module spec for {filepath}"

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.fail(f"Failed to import {chapter}/{filename}: {type(e).__name__}: {e}")


def test_total_file_count():
    """Verify we are testing the expected number of chapter files."""
    assert len(CHAPTER_FILES) == 24, (
        f"Expected 24 chapter files, found {len(CHAPTER_FILES)} in test list"
    )
