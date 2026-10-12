# CineMatch Phase 16: reproducibility and hosting notes

This note describes the current local setup, what the API needs at runtime, and
deployment options reviewed on 2026-10-12. No account was created and no public
service was deployed.

## Local setup

### Python API and recommendation code

Use Python 3.14.6, matching `.python-version`. From the project root:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local-lock.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

The local lock records the installed project environment, including research and
model-building packages. `requirements-backend.txt` is the smaller, fully pinned
serving dependency closure measured from this environment; it intentionally
omits training/model-download packages. These exact versions were checked on
Python 3.14.6 on Apple Silicon, not independently built in a Linux container.
The backend set is sufficient only when the validated plot-vector artifact and
all three required data files are present. The application uses stored plot
embeddings and does not load PyTorch or Sentence Transformers on that path.
Both requirement files pin versions but do not contain package hashes, so they
are version-reproducible rather than cryptographically locked artifacts.

In a second terminal, with the same Python environment active, run checks:

```sh
python -m unittest discover -s tests -v
```

### React frontend

Use Node 24.19.0 (`frontend/.nvmrc`) and pnpm 11.25.0 (`frontend/package.json`).
From `frontend/`:

```sh
corepack enable
corepack prepare pnpm@11.25.0 --activate
pnpm install --frozen-lockfile
pnpm test
pnpm build
pnpm dev
```

For local development the Vite server proxies `/api` to `127.0.0.1:8000`.
`frontend/.env.local` is machine-specific and ignored. Copy
`frontend/.env.example` to `frontend/.env.local` if setting a different local
API port. Do not put secrets in Vite variables: they are embedded in the
browser bundle.

## Files the current API loads

`build_hybrid_recommender()` reads these files:

- `data/enriched/movies.csv` (26 MB): displayed catalog and director/writer/cast/crew metadata.
- `data/cleaned/movies.csv` (14 MB): canonical catalog and plot text.
- `data/cleaned/ratings.csv` (2.3 MB): MovieLens interactions for collaborative filtering.
- `data/models/all_minilm_l6_v2_plots.npz` (4.7 MB): cached plot vectors.

The measured sum of these runtime data files is recorded in
`reports/phase16_resource_measurements.json`. The raw MovieLens archive, MPST
source file, intermediate processed files, and experimental chunked embedding
cache are not read by ordinary API startup. Do not add them to a deploy image
unless a new runtime path needs them.

The API imports NumPy, Pandas, SciPy, scikit-learn, FastAPI and Uvicorn. With the
matching embedding cache, `torch` and `sentence_transformers` are absent from
`sys.modules` after recommender construction. If the cache is missing, stale or
does not match the catalog fingerprint, startup will try to load the model;
deployment must therefore include the validated cache and set Hugging Face
offline variables, or explicitly provision the much heavier model runtime.

## Measurements and limits

Run the local measurement again with:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/phase16_measure_resources.py
```

The JSON report includes this machine's app import/startup time, process peak
RSS, warm API timings, and filesystem sizes. It performs requests in-process
through ASGI, so those timings exclude TCP, TLS, and internet latency. With five
samples, nearest-rank p95 is the maximum sample and should not be treated as a
stable tail-latency estimate. The RSS is for one process running the whole
recommendation engine and its collaborative matrix. Results vary by hardware
and package builds.

The latest rerun measured about 12.1 seconds until the FastAPI lifespan was
ready, 2.59 GiB process peak RSS, and 46.86 MiB of deployment data. Warm ASGI
median times were about 3 ms for search, 0.8 ms for movie details, 0.41 s for
movie recommendations, 4.01 s for a user, and 1.32 s for selected favorites.
These are local-machine observations, not promises about a cloud VM.

### Wikidata tests

The enrichment tests use mocked batch-fetch functions and temporary progress
files. One test captures and asserts the “Querying Wikidata” progress text, but
the message is printed before the mocked fetch function is called; it does not
make a live request. The isolated metadata-enrichment test run passed 14 tests
without that console output. Live Wikidata access is only reached by explicitly
running the metadata-enrichment script. Keep future tests patching the network
boundary and using temporary snapshot/progress paths.

## Hosting assessment

### Recommended practical architecture: Cloudflare Pages + Google Cloud Run

Host the static Vite build on Cloudflare Pages and the existing FastAPI app in
Cloud Run. Set `VITE_API_BASE_URL` at build time to the Cloud Run HTTPS service
URL, and set `CINEMATCH_CORS_ORIGINS` on the API to the exact Pages origin. This
avoids browser mixed-content blocking and gives the API a managed HTTPS URL
without buying a domain. Keep the API at one Uvicorn worker, start with request-
based billing, minimum instances zero, and a maximum of one instance. The local
peak RSS was 2.59 GiB, so use a 4 GiB memory allocation as an initial candidate;
this is a sizing estimate from a Mac measurement, not a Cloud Run measurement.
Cloud Run supports 2 vCPU with up to 8 GiB RAM, so 2 vCPU / 4 GiB is a plausible
starting point. A one-instance cap limits surprise scaling but means requests
queue during slow recommendation calls. The measured user route took about 4 s
warm locally.

Cloud Run's request-based free allowance currently includes 180,000 vCPU-seconds,
360,000 GiB-seconds, and 2 million requests per month. At 2 vCPU / 4 GiB, the
CPU and memory allowances each correspond to about 90,000 active instance
seconds before any shared billing-account usage. The free allowance is shared
across projects on the billing account. Outbound transfer has a 1 GiB/month
free allowance within North America; a deployment in India or serving users
globally may incur egress charges. Cloud Build, Artifact Registry, and other
Google Cloud resources can also be billable. Google Cloud requires a billing
account for the standard free tier. Therefore this is a credible $0 option at
low usage, but not guaranteed to remain free. Set budget alerts and review the
billing configuration before enabling it.

### Frontend: Cloudflare Pages

The built frontend is about 0.24 MiB, well below Pages' 25 MiB per-file limit;
the Free plan currently allows 500 builds per month. Static hosting does not
run this Python API. Do not add Workers functions or another product without
checking its quota and billing terms.

### Alternative if avoiding usage-based cloud billing is the priority: OCI

Oracle Always Free offers up to 2 Ampere OCPUs, 12 GB memory, and 10 TB monthly
outbound transfer, enough resources for this measured process. A single
Always Free VM can run FastAPI continuously and serve the built frontend too.
However, it is not the smoothest public setup: free instances can be reclaimed
after seven days below Oracle's utilization thresholds; capacity may be
unavailable; sign-up typically asks for a phone number and credit card; and a
browser-facing site needs HTTPS. Using a public IP with HTTPS requires managing
short-lived IP certificates, or the user must obtain a domain and configure
TLS. A Pages frontend must never call a plain HTTP IP. Oracle states its Always
Free resources do not expire after the trial, but choosing non-free resources
or exceeding allowances may cost money. OCI is a reasonable alternative only if
the user accepts its maintenance and certificate work.

### Alternatives considered

- **Render Free:** The plan has 512 MB and 0.1 CPU, below the measured 2.59 GiB
  peak. It sleeps after 15 minutes idle, may take about a minute to wake, and
  local files are ephemeral. It cannot host this existing backend without a
  substantial runtime redesign.
- **Vercel Hobby:** Its 2 GB Python function memory cap is below the measured
  RSS. The 500 MB uncompressed function bundle cap also requires careful
  packaging, though the local 1.1 GB virtual environment is not itself a deploy
  artifact. Its Hobby terms are for personal, noncommercial use.
- **Hugging Face Spaces:** Static Spaces are free, but its current docs say
  Docker/Gradio compute Spaces require a paid plan; therefore it is not a
  no-subscription backend choice.

## Publication and account steps still require you

No account was connected and no deployment was attempted. To publish later,
you will need to choose a host, create/authorize its account, decide whether to
link a Git provider or upload the source manually, and explicitly approve any
billing setup. A public Git repository is not required by this guide, but a
host's Git-based workflow will ask you to grant repository access. Before any
public launch, review dataset/publication rights below. None of these actions
has been performed in this phase.

## Data and model use before publication

The pinned `all-MiniLM-L6-v2` model is Apache-2.0 licensed. MovieLens latest
terms permit noncommercial research and redistribution with conditions, and
require permission for commercial or revenue-bearing use. The MPST dataset card
labels its dataset CC BY 4.0 and requests paper citations; individual plot
synopses also originate from third-party sources. Review source attribution,
redistribution, and public-display obligations before making the catalog
public. This guide is not a legal determination. Do not publish the data files
or deploy the site until you have resolved those rights.

## Official references checked on 2026-10-12

- [Cloud Run pricing and request-based free allowance](https://cloud.google.com/run/pricing)
- [Cloud Run CPU and memory configuration](https://docs.cloud.google.com/run/docs/configuring/services/cpu)
- [Cloudflare Pages limits](https://developers.cloudflare.com/pages/platform/limits/)
- [OCI Always Free resources, capacity and idle-instance rules](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)
- [OCI Free Tier sign-up and account terms](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier.htm)
- [Render free service limits](https://render.com/docs/free) and [compute sizes](https://render.com/docs/compute-plans)
- [Vercel Function limits](https://vercel.com/docs/functions/limitations) and [Hobby-plan restrictions](https://vercel.com/docs/plans/hobby)
- [Hugging Face Spaces overview and hardware/lifecycle limits](https://huggingface.co/docs/hub/spaces-overview)
- [MovieLens latest-small usage terms](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)
- [MPST dataset card and license notice](https://huggingface.co/datasets/cryptexcode/MPST)
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Let's Encrypt IP-address certificate support](https://letsencrypt.org/2026/03/11/shorter-certs-certbot)

## Operational caveats

- The existing recommendation API has no user accounts, rate limiting, or
  persistent user preferences. MovieLens user IDs are demo dataset IDs, not
  real CineMatch accounts.
- The API's browser CORS allowlist is disabled by default. For a separate static
  host, set a comma-separated list of exact trusted frontend origins in
  `CINEMATCH_CORS_ORIGINS`; never use `*` for this deployment.
- The API has no authentication because its current demo data is public. Do not
  add private user data without a separate security design.
- No deployment credentials, account setup, GitHub connection, public source
  publishing, or actual deployment is part of this phase.
