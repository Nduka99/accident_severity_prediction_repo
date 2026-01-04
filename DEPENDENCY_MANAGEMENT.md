# Deep Dive: Dependency Management & Stability Strategy

## The Root Cause: Why Did It Crash?
You experienced a **502 Bad Gateway** because the backend worker process silently crashed. This wasn't a Python "Exception" that could be caught with a `try/except` block; it was likely a **Segmentation Fault** (C-level memory error).

### The "Pickle" Problem
Machine Learning models (like your `robust_scaler.pkl`) are serialized using Python's `pickle` module. 
- **What Pickle Does**: It takes the exact memory structure of a Python object and saves it to a file.
- **The Catch**: Libraries like `scikit-learn` rely on compiled C-extensions for performance. If you save a model with `scikit-learn==1.2.2`, the file contains the memory layout expected by version 1.2.2.
- **The Crash**: When you try to load that file with `scikit-learn==1.3.0`, the new library code expects a different memory layout. It tries to read data from the wrong place in memory, and the operating system kills the process immediately for safety.

---

## Strategy: Managing Dependencies for Stability

To prevent this from crashing your project in the future, we use a tiered approach to dependency management.

### Level 1: Semantic Versioning (The "Loose" Way)
Libraries follow `MAJOR.MINOR.PATCH` (e.g., `1.3.0`).
- **Major**: Breaking changes (API changes).
- **Minor**: New features, backwards compatible.
- **Patch**: Bug fixes.

*Why this failed you*: In `requirements.txt`, you likely had just `scikit-learn`. `pip` installs the *latest* version (1.3.0), but your model was trained on an older one (1.2.2). Even a "minor" update can break binary serialization (pickles).

### Level 2: Strict Pinning (The "Safe" Way)
In your `requirements.txt`, you explicitly state the exact version you need.
```text
scikit-learn==1.2.2
pandas==1.5.3
numpy==1.24.3
```
**Pros**: Guarantees the deployed environment matches the dev environment.
**Cons**: You don't get updates automatically (which is actually a "pro" for stability).

### Level 3: Lock Files (The "Robust" Way)
Tools like **Poetry** or **pip-tools** generate a "lock file" (`poetry.lock`). This file records the exact version of *every single sub-dependency* (e.g., `scikit-learn` depends on `numpy`, `scipy`, `joblib`).
- It ensures that *everyone* (you, your colleague, the production server) has bit-for-bit identical environments.

### Level 4: Containerization (The "Immutable" Way)
Using **Docker**, you capture the Operating System, System Libraries, Python Version, and Pip Packages into a single "Image".
- Once an image is built, it never changes. You deploy that exact image.

---

## Strategy: How to Upgrade safely

You asked: *"Assuming we want to upgrade to more recent ones..."*
You cannot simply "upgrade" the library if you have saved model artifacts (`.pkl` files) that depend on the old version.

### The Upgrade Workflow
1. **Create Upgrade Branch**: Make a new git branch `chore/upgrade-deps`.
2. **Update Requirements**: Change `scikit-learn==1.2.2` to `scikit-learn==1.3.0`.
3. **Retrain Models**: You **MUST** re-run your training pipeline (Jupyter Notebooks) in this new environment. This will generate *new* `.pkl` files compatible with 1.3.0.
4. **Run Tests**: Verify the new model performance is equal to or better than the old one.
5. **Deploy**: Push the code changes *and* the new model artifacts together.

## Implementation For This Project

For right now, to get your system back online, we must align the environment with the existing artifacts.

1.  **Identify Training Version**: The error log explicitly stated the scaler was seemingly pickled with `1.2.2`.
2.  **Pin Version**: We will edit `requirements-backend.txt` to enforce `scikit-learn==1.2.2`.
3.  **Rebuild**: When you redeploy/rebuild Docker, it will install the older, compatible version.
