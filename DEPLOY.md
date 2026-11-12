# Deployment

The app needs two Python environments (Keras 3 + torch, and TensorFlow 2.10). The Dockerfile builds both inside one container, so the hosted site is a single link.

## Option A: Hugging Face Space (Docker)
1. Create a free account at huggingface.co and make a new Space: SDK = Docker, template = Blank, hardware = CPU basic.
2. Install Git LFS once: `git lfs install`
3. Clone the empty Space and copy this project into it, excluding virtual environments, `outputs/`, `_backup_before_final/` and `tests/`.
4. Put this block at the very top of the Space's `README.md`:
   ```
   ---
   title: Unified ML Internship App
   sdk: docker
   app_port: 7860
   ---
   ```
5. Commit and push:
   ```
   git add .
   git commit -m "Deploy app"
   git push
   ```
   The first build installs both environments and takes several minutes. Watch the Space's "Logs" tab.
6. The site opens at `https://huggingface.co/spaces/<user>/<space-name>`. Webcam and microphone work because the site is served over HTTPS.

Notes
- `.gitattributes` sends weights (`.keras`, `.h5`, `.pt`, `.onnx`, `.safetensors`, `.bin`) through Git LFS. This is required for the 1.3 GB voice model.
- The first request to each task loads its model and is slower.
- The TensorFlow worker runs on the CPU in the container.
- Free CPU Spaces sleep after inactivity and wake on the next visit.

## Option B: run the container yourself
```
docker build -t unified-app .
docker run -p 7860:7860 unified-app
```
Open http://localhost:7860

## Option C: GitHub only
Push the same project to a GitHub repository (Git LFS enabled). Others run it with the setup steps in `README.md`.
