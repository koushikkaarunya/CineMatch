# CineMatch

CineMatch is a movie discovery and recommendation project with a React + Vite
frontend and a FastAPI backend. The backend exposes movie search and details,
movie recommendations, MovieLens demo-user recommendations, and recommendations
from selected favorite movies.

## Project status

The local project includes recommendation experiments, historical evaluations,
the API, and the frontend. The serving catalog, ratings, and plot-vector
artifacts are checked in; the original MPST source CSV is stored with Git LFS.
See [data/README.md](data/README.md) and
[the local setup guide](docs/phase16-setup-and-deployment.md) before running it.

The evaluation reports document experimental results and their limitations;
they are not a guarantee of recommendation quality. In particular, the Phase
14.1/14.2 relevance labels were AI-assisted and user-approved, not independent
human judgments.

## Local development

Use Python 3.14.6 and Node 24.19.0. The setup guide explains dependency
installation, required local data files, the FastAPI server, and the Vite
frontend. It also describes the distinction between the full research
environment and the smaller serving dependency set.

Run backend tests from the project root:

```sh
python -m unittest discover -s tests -v
```

Run frontend tests and build from `frontend/`:

```sh
pnpm install --frozen-lockfile
pnpm test
pnpm build
```

## Data and licensing

Some datasets and generated serving artifacts are committed to Git, and the
MPST source CSV is stored with Git LFS. MovieLens and plot-source terms apply to
data and derived outputs; review the source notes and applicable terms before
redistributing data or operating a public service. The current repository has
no project-wide source-code license. Do not assume that public visibility
grants permission to reuse its code.

## Deployment

The local setup guide contains the Phase 16 hosting assessment. No live website
or public API is deployed by this repository yet.
