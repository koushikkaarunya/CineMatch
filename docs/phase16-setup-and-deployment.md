# CineMatch Phase 16: reproducibility and deployment readiness

Reviewed and updated on 2026-10-12. This guide prepares a possible Cloudflare
Pages frontend and OCI Ampere A1 FastAPI backend. It does not create accounts,
transfer runtime data, build or publish an image, or deploy a service.

## Current measurements

The checked-in `reports/phase16_resource_measurements.json` records the
following local measurements from an Apple Silicon Mac, not from Linux or OCI:

| Measurement | Observed value | Meaning |
| --- | ---: | --- |
| API import + lifespan startup | 12.082 seconds | Includes loading the four runtime artifacts and creating the in-memory recommender |
| Process peak RSS | 2,783,870,976 bytes (about 2.59 GiB) | One API process; use at least 4 GiB RAM in a cloud sizing trial |
| Runtime data files | 49,131,170 bytes (46.86 MiB) | The four files listed below |
| Frontend production bundle | about 0.24 MiB | Static Vite output; comfortably below Pages' 25 MiB per-file cap |

Five warm in-process ASGI samples had median latencies of 0.37 ms for health,
3.09 ms for search, 0.76 ms for details, 406 ms for movie recommendations,
4.01 s for a user, and 1.32 s for selected favorites. These exclude TCP, TLS,
network latency, cold starts, and cloud contention. With five samples, the
reported nearest-rank p95 is just the maximum sample, not a stable tail
estimate. The local user recommendation route is the observed latency hotspot.

## Local setup and checks

The repository pins the serving dependencies in `requirements-backend.txt` and
uses `.python-version` for Python 3.14.6. A separate ignored
`requirements-local-lock.txt` is a machine-local environment snapshot; do not
use or publish it as a portable deployment lock. For a clean serving setup:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-backend.txt
python -m pip check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

In another terminal, run backend tests from the project root:

```sh
./.venv/bin/python -m unittest discover -s tests -v
```

For the frontend, use Node 24.19.0 from `frontend/.nvmrc` and pnpm 11.25.0
from `frontend/package.json`:

```sh
cd frontend
corepack enable
corepack prepare pnpm@11.25.0 --activate
pnpm install --frozen-lockfile
pnpm test
pnpm build
```

The full backend test suite currently passes 161 tests. The focused metadata
enrichment suite passes 14 tests. The enrichment test that prints “Resuming”
and “Querying Wikidata” patches `fetch_wikidata_batch` with a local mock and
uses a temporary progress file; those messages describe the mocked request,
not a live query. An exact-ID cached snapshot is reused by default; only
explicitly running `scripts/enrich_metadata.py` with no matching snapshot (or
requesting a refresh) reaches WDQS. This API deployment does not run that
enrichment script.

The local Vite proxy sends `/api` to `http://127.0.0.1:8000`. The browser
production build instead reads `VITE_API_BASE_URL`; configure that value in
Cloudflare Pages. Vite variables are public and compiled into the bundle, so
never place secrets in them.

## API data requirements and provenance

At startup, the API loads these local artifacts:

| File | Local size | Role/source |
| --- | ---: | --- |
| `data/enriched/movies.csv` | 27,589,273 bytes (about 26.3 MiB) | MovieLens catalog joined with structured Wikidata credit metadata |
| `data/cleaned/movies.csv` | 14,279,076 bytes (about 13.6 MiB) | Cleaned MovieLens catalog and Wikipedia-source MPST plot/tag fields |
| `data/cleaned/ratings.csv` | 2,382,888 bytes (about 2.27 MiB) | MovieLens latest-small user ratings |
| `data/models/all_minilm_l6_v2_plots.npz` | 4,879,933 bytes (about 4.65 MiB) | Cached all-MiniLM-L6-v2 plot vectors |
| **Total** | **49,131,170 bytes (46.86 MiB)** | Exact current byte sum |

The four runtime artifacts above are checked into Git and are available to
deployment builds. The raw MovieLens archive, MPST source download, Wikidata
response snapshot, intermediate tables, experimental chunked-embedding cache,
and audit labels are not required by ordinary API startup and are excluded
from the container build context. The MPST source CSV is stored with Git LFS
for reproducibility, but Vercel does not need to fetch it to run the API. The
model cache
avoids importing/downloading PyTorch and Sentence Transformers during normal
startup. If the cache is missing or fails its catalog fingerprint validation,
startup may try to load the pretrained model; the container sets the Hugging
Face offline flags so this fails clearly rather than downloading a replacement.

`CINEMATCH_DATA_ROOT` now selects the data directory. It defaults to this
repository's `data/` directory, preserving local commands. A container sets it
to `/app/data`. The helper changes paths only; it does not transform data or
change ranking behavior.

### Rights gate before any public service

Do not transfer or publish the runtime files until their rights are resolved:

- MovieLens latest-small is described for research use with attribution; the
  dataset README requires permission for commercial or revenue-bearing use.
- The MPST dataset card labels the dataset CC BY 4.0 and asks for citation, but
  this project's retained plot text was selected from rows whose synopsis
  source is Wikipedia. The card's license alone does not establish that every
  underlying plot synopsis can be redistributed or publicly displayed without
  attribution/share-alike obligations. Review the source-level obligations.
- Wikidata structured credit data is CC0. The cached all-MiniLM-L6-v2 model is
  Apache-2.0. These do not grant rights to other datasets in the catalog.
- The repository has no project-wide source-code license. Public GitHub
  visibility does not grant permission to reuse the code.

This is a provenance summary, not a legal conclusion. A public site serving
recommendations from those records still transfers/uses the underlying data,
even if the CSVs are absent from Git. Until the rights question is resolved,
keep the API private and do not copy these files to OCI or another service.

## Prepared OCI backend container

`Dockerfile` uses the official Python 3.14.6 slim Bookworm image for Linux
ARM64, installs only `requirements-backend.txt`, checks that package set, copies
the API modules and the four runtime files, runs as an unprivileged user, and
starts exactly one Uvicorn worker on `0.0.0.0:$PORT`. `.dockerignore` is
allowlist-oriented: the four named data files and the application/runtime
modules are the only project inputs admitted to the build context. Raw sources,
model caches other than the validated plot vectors, tests, reports, virtual
environments, and frontend dependencies are excluded.

The local build command, once Docker is installed and dataset rights are
cleared, is:

```sh
docker build --platform linux/arm64 -t cinematch-api:local .
docker run --rm -p 8000:8000 \
  -e PORT=8000 \
  -e CINEMATCH_CORS_ORIGINS=http://localhost:5173 \
  cinematch-api:local
```

Then check `http://127.0.0.1:8000/api/v1/health`. The current development
machine has no Docker executable, so the image has **not** been built or
validated here. PyPI currently publishes CPython 3.14 Linux AArch64 wheels for
the native numerical packages used by the service (NumPy, SciPy, pandas,
scikit-learn, and pydantic-core); the Python 3.14.6 image is available for
ARM64. This is compatibility evidence, not a successful Linux image build.

When authorized, runtime files can be copied to a private OCI VM data directory
and mounted read-only at `/app/data`, instead of embedding them in an image.
The OCI service must set `CINEMATCH_DATA_ROOT=/app/data`. Either method sends
the MovieLens/plot data to OCI; neither is appropriate before the rights review.
Do not put the files in a public bucket or public image registry.

## Proposed public architecture and actual constraints

### Cloudflare Pages frontend

Cloudflare Pages supports Vite builds and serves the generated static `dist/`
directory. Current Free limits include 500 builds/month, 20,000 files per site,
and 25 MiB per individual file. The measured frontend bundle is small. Suggested
dashboard configuration after explicit approval:

- Connect the GitHub `CineMatch` repository; set production branch `main`.
- Set the project root directory to `frontend`.
- Set `PNPM_VERSION=11.25.0`, then build with
  `pnpm install --frozen-lockfile && pnpm build`.
- Set build output directory to `dist`.
- Pin Node to `24.19.0` using the existing `frontend/.nvmrc`. Pages' current
  build image defaults to pnpm 10.11.1, so explicitly set `PNPM_VERSION=11.25.0`
  rather than relying on package-manager auto-detection.
- Set production `VITE_API_BASE_URL` to `https://<real-api-host>/api/v1`.
  `frontend/.env.production.example` uses the reserved `.invalid` hostname and
  is documentation only; it must be replaced before a real build.
- Once connected, pause automatic production deployments until the API and
  rights review are ready. Pages Git integration normally builds from Git
  changes; this guide has not connected GitHub or triggered a build.

### OCI Ampere A1 FastAPI backend

The local 2.59 GiB peak fits an A1 VM with 8 GiB RAM with headroom. The
Always-Free allowance currently provides up to 1,500 OCPU-hours and 9,000
GB-hours per month (equivalent to 2 OCPUs and 12 GB memory), a 200 GB block
volume allowance, and 10 TB/month outbound transfer in the home region. Start
with one OCPU and 8 GB RAM (744 OCPU-hours and 5,952 GB-hours in a 31-day
month), then measure Linux startup and memory before increasing it. Two OCPUs
for a full 31-day month would be 1,488 hours, close to the 1,500-hour cap.
Capacity may be unavailable in the selected home region.

There is no guarantee of uptime: OCI may reclaim an A1 instance after seven
days when its CPU p95, network, and (for A1) memory utilization are all below
20%. A low-traffic recommendation demo could meet that idle condition. Free
capacity can also be unavailable. OCI sign-up typically requests a phone and
payment card; do not upgrade to Pay As You Go or create resources outside the
Always Free labels. Quotas and terms can change, so check the console's current
limits before creating anything. Exceeding a free quota or selecting a paid
resource may incur charges.

### HTTPS and CORS are launch blockers until a real API origin exists

Browsers will block a Cloudflare Pages HTTPS page from calling an HTTP API.
The API therefore needs a trusted HTTPS origin. The CORS middleware already
accepts exact comma-separated origins through `CINEMATCH_CORS_ORIGINS`; after
the Pages project provides its final `https://<name>.pages.dev` origin, set
that exact value on the backend. Do not use `*`.

For routine automated HTTPS on a public backend, use a domain you control and a
reverse proxy such as Caddy. No domain was provided in this task. A domain is
not strictly required: Let's Encrypt now issues IP-address certificates, but
they expire after six days. Certbot 5.4+ can request and renew them, with a
custom deploy hook needed to reload the reverse proxy because its web-server
installers do not yet support IP certificates. The VM's public IP must remain
stable, port 80 must serve ACME challenges, and renewal/reload must be tested.
This is feasible at no certificate charge, but is more fragile and operationally
demanding than a domain. Until the renewal hook is prepared and tested on the
chosen host, a trusted production API URL is not ready. Do not deploy an
HTTP-only endpoint.

### Why not choose a different free backend now?

The measured process peak is above the 512 MiB free Render instance and above
the 2 GiB Vercel Python function limit. OCI A1 is the best fit among the
examined options for persistent memory and zero listed VM charge, but includes
idle reclamation, capacity, TLS, account, and rights risks. Google Cloud Run
can provide managed HTTPS and scale-to-zero, but requires a billing account;
free usage is shared with other projects and network/build/storage charges may
apply. It is a useful comparison, not a guaranteed zero-cost substitute.

## Exact manual steps after approval

No cloud account or resource has been created. Once the data rights and HTTPS
host questions are resolved, the user will need to:

1. Create/authorize Cloudflare and OCI accounts. OCI may request phone and card
   verification. Select an OCI home region with A1 availability; never upgrade
   the account or select a non-Always-Free resource without a separate decision.
2. In Cloudflare Pages, connect the GitHub repository and configure the
   production branch, `frontend` root, pnpm build, `dist` output, and the
   final production `VITE_API_BASE_URL`. Keep production auto-deploy paused
   until the backend is ready.
3. Create one A1 ARM64 VM inside its current Always Free quota, secure SSH,
   install Docker, and establish a trusted HTTPS hostname/certificate. Expose
   only the HTTPS reverse proxy publicly; keep Uvicorn bound behind it.
4. After authorization to transfer the licensed runtime data, securely stage
   the four runtime artifacts or the built image, run the container, and
   configure the exact Pages origin in `CINEMATCH_CORS_ORIGINS`.
5. Test the public health endpoint, movie search, one movie recommendation, one
   existing-user recommendation, browser CORS, and the Pages website. Watch
   OCI usage/quota and verify the live endpoints rather than assuming success.

## Official references reviewed 2026-10-12

- [Cloudflare Pages build configuration](https://developers.cloudflare.com/pages/configuration/build-configuration/)
- [Cloudflare Pages build image and Node version](https://developers.cloudflare.com/pages/configuration/build-image/)
- [Cloudflare Pages limits](https://developers.cloudflare.com/pages/platform/limits/)
- [Cloudflare Pages Git integration](https://developers.cloudflare.com/pages/get-started/git-integration/)
- [OCI Always Free resources, quotas, and idle reclamation](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)
- [OCI Free Tier sign-up](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier.htm)
- [OCI service limits](https://docs.oracle.com/en-us/iaas/Content/General/service-limits/default.htm)
- [Let's Encrypt IP-address certificates](https://letsencrypt.org/2026/03/11/shorter-certs-certbot)
- [MovieLens latest-small terms](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)
- [MPST dataset card](https://huggingface.co/datasets/cryptexcode/MPST)
- [Wikidata licensing](https://www.wikidata.org/wiki/Wikidata:Licensing)
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Python 3.14.6 slim Bookworm image](https://hub.docker.com/layers/library/python/3.14.6-slim-bookworm/images/sha256-855b02caff87c19c1a39b8e411ea00428e115d212ff82210caf5028691fef185)
- [NumPy 2.5.3](https://pypi.org/project/numpy/2.5.3/), [pandas 3.0.6](https://pypi.org/project/pandas/3.0.6/), [SciPy 1.18.1](https://pypi.org/project/scipy/1.18.1/), [scikit-learn 1.9.1](https://pypi.org/project/scikit-learn/1.9.1/), and [pydantic-core 2.50.1](https://pypi.org/project/pydantic_core/2.50.1/) package files
