# Running the churn service in Docker

Short notes on the image of the service: what goes into it, how it is built
and what the container needs to keep its state.

## Why a container

- The image holds the Python version, the dependencies, the code and the
  dataset of the service. It runs the same on a laptop, a CI runner and a
  server, without a virtual environment to set up on each of them.
- An image is built once and run as many containers; it does not change while
  they run. Whatever changes, like a trained model, lives outside it.

## What goes into the image

- `.dockerignore` lists what may go in rather than what may not: everything
  is excluded except `requirements.txt`, `src/` and `data/`. A file added to
  the project later, like a `.env` or a trained model, stays out until it is
  listed, instead of leaking into the image by default.
- `.venv` stays out: it is built for macOS, the image runs Linux and installs
  its own dependencies. It would also add 271 MB to every build.
- `models/` stays out: a model pickled on the host is tied to the host
  scikit-learn version, and the trained model is state, not code.
- Only `requirements.txt` is installed, not `requirements-dev.txt`: pytest
  and ruff are not needed to serve requests.

## How it is built

- `python:3.12-slim`: the Python of the local `.venv`, so the image runs the
  code the tests ran on, on a Debian image without compilers and docs.
  scikit-learn, pandas and numpy come as prebuilt wheels and need none.
- `requirements.txt` is copied and installed before the code. Docker reuses a
  layer while its inputs do not change, so a change of the code alone skips
  the minute of `pip install`.
- `PYTHONUNBUFFERED=1` sends every log line to `docker logs` at once instead
  of keeping it in a buffer until the buffer fills or the process ends.
- `PROJECT_ROOT` is found from the location of `src/`, so with the code in
  `/app/src` the dataset is read from `/app/data` and models are written to
  `/app/models` without a setting.
- Most of the 700 MB of the image are numpy, scipy, scikit-learn and pandas;
  the code and the dataset take less than 300 KB.

## Running it

- `CMD` starts uvicorn in the exec form, a JSON list, not a shell string:
  uvicorn is then process 1 and receives the `SIGTERM` of `docker stop`, so it
  finishes the running requests and stops at once. In the shell form the
  shell is process 1 and does not pass the signal on, so uvicorn is killed
  after 10 seconds instead.
- `--host 0.0.0.0`: `127.0.0.1` inside a container is the container itself,
  and a published port (`-p 8000:8000`) would reach nothing.
- No `--reload`: it watches the files for changes, which do not happen in an
  image, and costs a process.
- One process, no `--workers`: the served model lives in the memory of the
  process. With several workers a model trained by one of them would be
  served by it alone, and `/model/status` would answer differently from one
  request to the next. Several instances need a shared model store first.
- The service runs as the user `app`, not as root. Loading the model unpickles
  a file, which runs code from it; if the file is ever replaced, that code
  runs without the rights to change the code of the service or install
  anything. `app` owns `/app/models` only, where the service writes; the code
  and the dataset stay owned by root and read-only to it.

## Keeping the trained model

- A container starts from the image every time, so a model written to its
  own filesystem is lost with it, and with `--rm` that is on every stop.
- `-v churn-models:/app/models` mounts a named volume there. The volume lives
  outside every container: the next one mounts it again and loads the model
  and the history the previous one saved.
- A new named volume takes the content and the owner of the directory it is
  mounted on, so it belongs to `app` and the service can write to it. A bind
  mount of a host directory keeps the host permissions instead, and on Linux
  the user of the container may not be allowed to write there.
- The volume is set by `docker run`, not by the Dockerfile. `VOLUME` in a
  Dockerfile can only declare a mount point, not name a volume: without `-v`
  every container would get a new anonymous volume, and with `--rm` it is
  deleted with the container. Where the state is kept is a decision of the
  place the image runs, not of the image.

## The health check

- `HEALTHCHECK` calls `/health` every 30 seconds, from inside the container.
  The slim image has no curl, so the call is a line of Python with `urllib`;
  a status of `400` or above, or no answer within the timeout, fails it.
- After three failures in a row `docker ps` shows the container `unhealthy`.
  Docker itself only reports it and does not restart the container. Swarm
  replaces an unhealthy one, and Compose can hold back the services that
  depend on it until it is healthy.
- `/health` answers `200` without a model, so a container waiting for its
  first training stays `healthy`. See [monitoring.md](monitoring.md) for why.

## Beyond one container

- Docker Compose describes the run in a file: the port, the volume and the
  environment, instead of flags typed every time.
- A multi-stage build installs the dependencies in one stage and copies only
  the installed packages into the final one; it pays off when the build needs
  compilers that the running service does not.
- Pinning every dependency, transitive ones included, in a lock file makes the
  image the same on every build; with version ranges a rebuild may pick up a
  new release.
- Several instances behind a load balancer need the model in shared storage,
  like an object store or a model registry, and the history in a database,
  rather than in a volume of one host.
