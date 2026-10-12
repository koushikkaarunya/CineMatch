const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '/api/v1').replace(/\/$/, '')

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        Accept: 'application/json',
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        ...options.headers,
      },
    })
  } catch {
    if (options.signal?.aborted) throw new DOMException('Request was aborted', 'AbortError')
    throw new Error('Could not reach CineMatch. Check that the API server is running and try again.')
  }

  let payload
  try {
    payload = await response.json()
  } catch {
    if (!response.ok) {
      throw new Error(`The CineMatch server is not reachable (${response.status}). Please check that the API is running and try again.`)
    }
    throw new Error('The server returned an unreadable response. Please try again.')
  }

  if (!response.ok) {
    const detail = payload?.detail
    const message = typeof detail === 'string'
      ? detail
      : detail?.message || detail?.msg || (Array.isArray(detail) ? detail.map((item) => item.message || item.msg || JSON.stringify(item)).join(', ') : '')
    throw new Error(message || `Request failed (${response.status}). Please try again.`)
  }
  return payload
}

export function searchMovies(query, { limit = 12, signal } = {}) {
  const params = new URLSearchParams({ q: query, limit: String(limit) })
  return request(`/movies?${params}`, { signal })
}

export function getMovie(movieId) {
  return request(`/movies/${encodeURIComponent(movieId)}`)
}

export function getMovieRecommendations(movieId, topK = 8) {
  return request(`/movies/${encodeURIComponent(movieId)}/recommendations?top_k=${topK}`)
}

export function getUserRecommendations(userId, topK = 8) {
  return request(`/users/${encodeURIComponent(userId)}/recommendations?top_k=${topK}`)
}

export function getFavoriteRecommendations(movieIds, topK = 8) {
  return request('/recommendations/from-favorites', {
    method: 'POST',
    body: JSON.stringify({ favorite_movie_ids: movieIds, top_k: topK }),
  })
}
