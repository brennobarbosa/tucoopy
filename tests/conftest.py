import sys

# OneDrive/Windows environments sometimes deny writes inside the repo, which breaks
# Python's attempt to create `__pycache__` entries during test collection.
sys.dont_write_bytecode = True

