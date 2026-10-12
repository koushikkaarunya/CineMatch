import { useEffect, useState } from 'react'
import {
  getFavoriteRecommendations,
  getMovie,
  getMovieRecommendations,
  getUserRecommendations,
  searchMovies,
} from './api.js'

function readRoute() {
  const params = new URLSearchParams(window.location.search)
  return { view: params.get('view') || 'discover', movieId: params.get('movie') }
}

function useRoute() {
  const [route, setRoute] = useState(readRoute)
  useEffect(() => {
    const onPopState = () => setRoute(readRoute())
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  function navigate(next) {
    const params = new URLSearchParams()
    if (next.view && next.view !== 'discover') params.set('view', next.view)
    if (next.movieId) params.set('movie', String(next.movieId))
    const suffix = params.toString()
    window.history.pushState({}, '', `${window.location.pathname}${suffix ? `?${suffix}` : ''}`)
    setRoute(readRoute())
    if (typeof window.scrollTo === 'function') {
      window.scrollTo({ top: 0, behavior: 'smooth' })
    }
  }
  return [route, navigate]
}

function BrandMark() {
  return <span className="brand-mark" aria-hidden="true"><span /><span /><span /></span>
}

function Header({ view, onNavigate }) {
  return (
    <header className="site-header">
      <a className="brand" href="/" onClick={(event) => { event.preventDefault(); onNavigate({ view: 'discover' }) }}>
        <BrandMark />
        <span>Cine<span className="brand-light">Match</span></span>
      </a>
      <nav className="main-nav" aria-label="Main navigation">
        <button className={view === 'discover' ? 'nav-link active' : 'nav-link'} onClick={() => onNavigate({ view: 'discover' })} aria-current={view === 'discover' ? 'page' : undefined}>Discover</button>
        <button className={view === 'personalized' ? 'nav-link active' : 'nav-link'} onClick={() => onNavigate({ view: 'personalized' })} aria-current={view === 'personalized' ? 'page' : undefined}>For you</button>
      </nav>
      <div className="header-note"><span className="status-dot" /> Movie discovery, made personal</div>
    </header>
  )
}

const DEFAULT_EXCLUDE_IDS = []

function SearchBox({ label = 'Search the movie catalog', onSelect, compact = false, excludeIds = DEFAULT_EXCLUDE_IDS }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const excludeKey = excludeIds.join(',')

  useEffect(() => {
    const trimmed = query.trim()
    if (!trimmed) {
      setResults([])
      setLoading(false)
      setError('')
      return undefined
    }

    const controller = new AbortController()
    const timer = window.setTimeout(async () => {
      setLoading(true)
      setError('')
      try {
        const response = await searchMovies(trimmed, { limit: 12, signal: controller.signal })
        setResults((response.items || []).filter((movie) => !excludeIds.includes(movie.movie_id)))
      } catch (requestError) {
        if (requestError.name !== 'AbortError') setError(requestError.message)
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }, 300)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [query, excludeKey])

  return (
    <div className={`search-wrap${compact ? ' compact' : ''}`}>
      <label className="sr-only" htmlFor={compact ? 'movie-search-small' : 'movie-search'}>{label}</label>
      <span className="search-icon" aria-hidden="true">⌕</span>
      <input
        id={compact ? 'movie-search-small' : 'movie-search'}
        type="search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Search by title..."
        autoComplete="off"
      />
      {loading && <span className="search-hint" role="status">Searching</span>}
      {query.trim() && (
        <div className="search-results" role="region" aria-label="Movie search results">
          {error && <p className="inline-error" role="alert">{error}</p>}
          {!error && loading && <p className="result-note">Looking through the catalog…</p>}
          {!error && !loading && results.length === 0 && <p className="result-note">No matching titles found.</p>}
          {!error && results.map((movie) => (
            <button key={movie.movie_id} className="search-result" onClick={() => { onSelect(movie); setQuery('') }}>
              <span className="result-art" aria-hidden="true">{movie.title?.slice(0, 1) || 'F'}</span>
              <span className="result-copy"><strong>{movie.title}</strong><small>{movie.release_year || 'Year unavailable'}{movie.genres?.length ? ` · ${movie.genres.slice(0, 3).join(' · ')}` : ''}</small></span>
              <span className="result-arrow" aria-hidden="true">↗</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function MovieArtwork({ title, small = false }) {
  const initial = (title || 'Film').replace(/^(The|A|An)\s+/i, '').charAt(0).toUpperCase()
  return (
    <div className={`movie-art${small ? ' small' : ''}`} aria-label={`Poster unavailable for ${title}`}>
      <div className="art-frame"><BrandMark /><span className="art-initial">{initial}</span><span className="art-caption">CINEMATCH<br />CATALOG</span></div>
    </div>
  )
}

function MovieCard({ movie, onOpen, recommendation = false }) {
  return (
    <article className="movie-card">
      <button className="movie-card-button" onClick={() => onOpen(movie.movie_id)} aria-label={`Open details for ${movie.title}`}>
        <MovieArtwork title={movie.title} />
        <span className="card-copy">
          <span className="card-title">{movie.title}</span>
          <span className="card-meta">{movie.release_year || 'Year unavailable'}{movie.genres?.length ? ` · ${movie.genres.slice(0, 2).join(' / ')}` : ''}</span>
          {recommendation && <span className="card-open">Explore film <span aria-hidden="true">↗</span></span>}
        </span>
      </button>
    </article>
  )
}

function Discover({ onOpen }) {
  return (
    <main>
      <section className="hero-section">
        <div className="hero-copy">
          <p className="eyebrow"><span className="eyebrow-line" /> A better way to find your next film</p>
          <h1>Good stories<br /><em>find their people.</em></h1>
          <p className="hero-description">Search a catalog of films, explore the details, and find recommendations grounded in the movies you love.</p>
          <SearchBox onSelect={(movie) => onOpen(movie.movie_id)} />
          <p className="catalog-note">MovieLens catalog · 9,742 films · Some plots and credits may be unavailable</p>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="hero-ring ring-one" /><div className="hero-ring ring-two" />
          <div className="hero-ticket"><span>TONIGHT'S<br />PICK</span><BrandMark /><span className="ticket-id">CM · 001</span></div>
          <div className="hero-orbit orbit-a" /><div className="hero-orbit orbit-b" />
          <span className="hero-art-caption">YOUR NEXT<br />FAVORITE IS<br /><i>OUT THERE.</i></span>
        </div>
      </section>
      <section className="how-section">
        <div><p className="eyebrow">START WITH A TITLE</p><h2>Follow the story<br />where it takes you.</h2></div>
        <p>Open a film to see its available plot and credits, then ask CineMatch to find related titles. Or choose a few favorites to shape a new-user recommendation list.</p>
      </section>
    </main>
  )
}

function CreditGroup({ title, people }) {
  if (!people?.length) return null
  return <div className="credit-group"><span className="credit-label">{title}</span><p>{people.map((person) => person.name).filter(Boolean).join(', ') || 'Not listed'}</p></div>
}

function getExplanations(item) {
  const source = item.seed_explanations || item.content_explanation
  if (source && typeof source === 'object' && !Array.isArray(source)) {
    return Object.entries(source).filter(([, text]) => typeof text === 'string' && text.trim()).map(([seed, text]) => ({ seed, text }))
  }
  if (typeof item.explanation === 'string' && item.explanation.trim()) return [{ seed: null, text: item.explanation }]
  return []
}

function ResultCard({ item, onOpen }) {
  const explanations = getExplanations(item)
  const score = item.fusion_score ?? item.hybrid_score ?? item.mean_reciprocal_rank
  const rank = item.content_rank ?? item.collaborative_rank ?? item.best_seed_rank
  const signal = item.contributing_sources || item.recommendation_source || (item.content_rank ? 'content' : item.collaborative_rank ? 'collaborative' : '')
  return (
    <article className="recommendation-card">
      <button className="recommendation-main" onClick={() => onOpen(item.movie_id)} aria-label={`Open ${item.title}`}>
        <MovieArtwork title={item.title} small />
        <span className="recommendation-title"><strong>{item.title}</strong><small>{item.release_year || 'Explore movie details'}</small></span>
        <span className="recommendation-arrow" aria-hidden="true">↗</span>
      </button>
      <div className="recommendation-evidence">
        <div className="evidence-label-row">
          {signal && <span className="source-pill">{signal.replaceAll('_', ' ')}</span>}
          {rank != null && <span className="rank-label">Rank {rank}</span>}
          {score != null && <span className="score-label">Model score · {Number(score).toFixed(3)} <span>(not a probability)</span></span>}
        </div>
        {explanations.length ? explanations.map(({ seed, text }) => (
          <p className="evidence-text" key={`${seed}-${text}`}>
            {seed && <span className="seed-label">From {seed}: </span>}{text}
          </p>
        )) : <p className="evidence-missing">No explanation was returned for this result.</p>}
      </div>
    </article>
  )
}

function ResultsPanel({ response, loading, error, onOpen, emptyLabel = 'No recommendations are available for this request.' }) {
  if (loading) return <div className="state-panel" role="status"><span className="spinner" />Finding films that connect…</div>
  if (error) return <div className="state-panel error-panel" role="alert"><strong>We couldn’t load recommendations.</strong><span>{error}</span></div>
  if (!response) return null
  const items = response.items || []
  return (
    <section className="results-section" aria-live="polite">
      <div className="results-heading"><div><p className="eyebrow">YOUR RESULTS</p><h3>{items.length ? 'A few films to explore' : 'No matches this time'}</h3></div><span className="result-count">{items.length} {items.length === 1 ? 'film' : 'films'}</span></div>
      {response.message && <p className="response-message">{response.message}</p>}
      {response.status === 'needs_preferences' && <div className="state-panel">Choose one or more favorite films to get started.</div>}
      {items.length === 0 && response.status !== 'needs_preferences' && <div className="state-panel">{emptyLabel}</div>}
      <div className="recommendation-list">{items.map((item) => <ResultCard key={item.movie_id} item={item} onOpen={onOpen} />)}</div>
      {items.length > 0 && <p className="score-disclaimer">Scores and ranks describe this model’s ordering; they are not probabilities or a promise of relevance.</p>}
    </section>
  )
}

function MovieDetails({ movieId, onBack, onOpen, backLabel = '← Back to discover' }) {
  const [movie, setMovie] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [response, setResponse] = useState(null)
  const [recommendationLoading, setRecommendationLoading] = useState(false)
  const [recommendationError, setRecommendationError] = useState('')

  useEffect(() => {
    let active = true
    setLoading(true); setMovie(null); setError(''); setResponse(null)
    getMovie(movieId).then((value) => { if (active) setMovie(value) }).catch((problem) => { if (active) setError(problem.message) }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [movieId])

  async function recommend() {
    setRecommendationLoading(true); setRecommendationError(''); setResponse(null)
    try { setResponse(await getMovieRecommendations(movieId, 8)) } catch (problem) { setRecommendationError(problem.message) } finally { setRecommendationLoading(false) }
  }

  if (loading) return <main className="page-container"><div className="state-panel" role="status"><span className="spinner" />Loading movie details…</div></main>
  if (error) return <main className="page-container"><button className="back-link" onClick={onBack}>{backLabel}</button><div className="state-panel error-panel" role="alert">{error}</div></main>
  if (!movie) return null

  return (
    <main className="page-container detail-page">
      <button className="back-link" onClick={onBack}>{backLabel}</button>
      <section className="detail-hero">
        <MovieArtwork title={movie.title} />
        <div className="detail-copy">
          <p className="eyebrow">MOVIE DETAILS <span className="catalog-id">CATALOG ID {movie.movie_id}</span></p>
          <h1>{movie.title}</h1>
          <div className="genre-list">{movie.release_year && <span className="year-tag">{movie.release_year}</span>}{(movie.genres || []).map((genre) => <span className="genre-tag" key={genre}>{genre}</span>)}</div>
          <p className="plot-copy">{movie.plot || 'A plot summary is not available for this title in the catalog.'}</p>
          <button className="button button-primary" onClick={recommend} disabled={recommendationLoading}>
            {recommendationLoading ? 'Finding related films…' : 'Find films like this'} <span aria-hidden="true">↗</span>
          </button>
        </div>
      </section>
      <section className="credits-section" aria-label="Movie credits">
        <CreditGroup title="Director" people={movie.directors} />
        <CreditGroup title="Writers" people={movie.writers} />
        <CreditGroup title="Cast" people={movie.cast} />
        <CreditGroup title="Selected crew" people={movie.selected_crew} />
        {!movie.directors?.length && !movie.writers?.length && !movie.cast?.length && !movie.selected_crew?.length && <p className="missing-note">Credits are not available for this movie in the catalog.</p>}
      </section>
      <ResultsPanel response={response} loading={recommendationLoading} error={recommendationError} onOpen={onOpen} />
    </main>
  )
}

function Personalized({ onOpen }) {
  const [userId, setUserId] = useState('')
  const [userResponse, setUserResponse] = useState(null)
  const [userLoading, setUserLoading] = useState(false)
  const [userError, setUserError] = useState('')
  const [favorites, setFavorites] = useState([])
  const [favoriteResponse, setFavoriteResponse] = useState(null)
  const [favoriteLoading, setFavoriteLoading] = useState(false)
  const [favoriteError, setFavoriteError] = useState('')

  async function submitUser(event) {
    event.preventDefault(); setUserLoading(true); setUserError(''); setUserResponse(null)
    try { setUserResponse(await getUserRecommendations(userId.trim(), 8)) } catch (problem) { setUserError(problem.message) } finally { setUserLoading(false) }
  }
  function addFavorite(movie) {
    setFavorites((current) => current.some((item) => item.movie_id === movie.movie_id) ? current : [...current, movie])
    setFavoriteResponse(null); setFavoriteError('')
  }
  async function submitFavorites() {
    setFavoriteLoading(true); setFavoriteError(''); setFavoriteResponse(null)
    try { setFavoriteResponse(await getFavoriteRecommendations(favorites.map((movie) => movie.movie_id), 8)) } catch (problem) { setFavoriteError(problem.message) } finally { setFavoriteLoading(false) }
  }

  return (
    <main className="page-container personal-page">
      <section className="page-intro"><p className="eyebrow"><span className="eyebrow-line" /> MAKE IT YOURS</p><h1>Your taste,<br /><em>your next watch.</em></h1><p>Start with a MovieLens profile or pick a few films you already love.</p></section>
      <div className="personal-grid">
        <section className="personal-panel">
          <div className="panel-number">01</div><p className="eyebrow">EXISTING MOVIELENS USER</p><h2>Use your movie history</h2>
          <p className="panel-description">Enter a user ID from the MovieLens dataset. This is a dataset identifier, not an account login.</p>
          <form className="user-form" onSubmit={submitUser}>
            <label htmlFor="user-id">MovieLens user ID</label>
            <div className="form-row"><input id="user-id" inputMode="numeric" pattern="[0-9]+" value={userId} onChange={(event) => setUserId(event.target.value)} placeholder="For example, 1" required /><button className="button button-primary" type="submit" disabled={userLoading || !userId.trim()}>{userLoading ? 'Loading…' : 'Get picks'} <span aria-hidden="true">↗</span></button></div>
          </form>
          <ResultsPanel response={userResponse} loading={userLoading} error={userError} onOpen={onOpen} />
        </section>
        <section className="personal-panel favorites-panel">
          <div className="panel-number">02</div><p className="eyebrow">NEW TO CINEMATCH?</p><h2>Pick a few favorites</h2>
          <p className="panel-description">Search the catalog and choose films you enjoy. CineMatch will look for content connections.</p>
          <SearchBox compact label="Search for a favorite movie" onSelect={addFavorite} excludeIds={favorites.map((movie) => movie.movie_id)} />
          {favorites.length > 0 ? <div className="selected-list" aria-label="Selected favorite movies">{favorites.map((movie) => <span className="selected-chip" key={movie.movie_id}>{movie.title}<button onClick={() => { setFavorites((current) => current.filter((item) => item.movie_id !== movie.movie_id)); setFavoriteResponse(null) }} aria-label={`Remove ${movie.title}`}>×</button></span>)}</div> : <p className="selection-hint">Your selected movies will appear here.</p>}
          <button className="button button-primary favorite-submit" onClick={submitFavorites} disabled={favoriteLoading || favorites.length === 0}>{favoriteLoading ? 'Finding your matches…' : `Find recommendations${favorites.length ? ` · ${favorites.length}` : ''}`} <span aria-hidden="true">↗</span></button>
          <ResultsPanel response={favoriteResponse} loading={favoriteLoading} error={favoriteError} onOpen={onOpen} />
        </section>
      </div>
      <p className="personal-footnote">Recommendations come from the available CineMatch models and catalog signals. Some titles may have incomplete plot or credit information.</p>
    </main>
  )
}

export default function App() {
  const [route, navigate] = useRoute()
  return (
    <div className="app-shell">
      <Header view={route.movieId ? 'discover' : route.view} onNavigate={navigate} />
      {route.movieId
        ? <MovieDetails
            key={route.movieId}
            movieId={route.movieId}
            onBack={() => navigate({ view: route.view || 'discover' })}
            backLabel={route.view === 'personalized' ? '← Back to For you' : '← Back to discover'}
            onOpen={(movieId) => navigate({ view: route.view || 'discover', movieId })}
          />
        : route.view === 'personalized'
          ? <Personalized onOpen={(movieId) => navigate({ view: 'personalized', movieId })} />
          : <Discover onOpen={(movieId) => navigate({ view: 'discover', movieId })} />}
      <footer className="site-footer"><a className="brand footer-brand" href="/" onClick={(event) => { event.preventDefault(); navigate({ view: 'discover' }) }}><BrandMark /><span>Cine<span className="brand-light">Match</span></span></a><span>Movie data and recommendations are provided for discovery.</span><span className="footer-mark">A catalog of connections <i>✳</i></span></footer>
    </div>
  )
}
