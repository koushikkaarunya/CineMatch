import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App.jsx'
import * as api from './api.js'

vi.mock('./api.js', () => ({
  getFavoriteRecommendations: vi.fn(),
  getMovie: vi.fn(),
  getMovieRecommendations: vi.fn(),
  getUserRecommendations: vi.fn(),
  searchMovies: vi.fn(),
}))

const sampleMovie = {
  movie_id: 1,
  title: 'Toy Story (1995)',
  release_year: 1995,
  genres: ['Adventure', 'Animation'],
  plot: 'A toy comes alive.',
  plot_tags: ['friendship'],
  directors: [{ name: 'John Lasseter' }],
  writers: [],
  cast: [],
  selected_crew: [],
}

beforeEach(() => {
  window.history.replaceState({}, '', '/')
  window.scrollTo = vi.fn()
  api.searchMovies.mockResolvedValue({ items: [{ movie_id: 1, title: 'Toy Story (1995)', release_year: 1995, genres: ['Animation'] }] })
})

afterEach(() => cleanup())

describe('CineMatch frontend', () => {
  it('searches the catalog after a short debounce and opens the chosen movie', async () => {
    const user = userEvent.setup()
    api.getMovie.mockResolvedValue(sampleMovie)
    render(<App />)

    await user.type(screen.getByRole('searchbox', { name: 'Search the movie catalog' }), 'toy story')
    const result = await screen.findByRole('button', { name: /Toy Story \(1995\).*Animation/ })
    await user.click(result)

    expect(await screen.findByRole('heading', { name: 'Toy Story (1995)' })).toBeInTheDocument()
    expect(screen.getByText('A toy comes alive.')).toBeInTheDocument()
    expect(window.location.search).toContain('movie=1')
  })

  it('shows a useful search error when the API cannot be reached', async () => {
    const user = userEvent.setup()
    api.searchMovies.mockRejectedValue(new Error('API server is not reachable.'))
    render(<App />)

    await user.type(screen.getByRole('searchbox', { name: 'Search the movie catalog' }), 'unknown')
    expect(await screen.findByRole('alert')).toHaveTextContent('API server is not reachable.')
  })

  it('shows recommendation loading, then actual API explanations', async () => {
    const user = userEvent.setup()
    api.getMovie.mockResolvedValue(sampleMovie)
    let finishRequest
    api.getMovieRecommendations.mockReturnValue(new Promise((resolve) => { finishRequest = resolve }))
    window.history.replaceState({}, '', '/?movie=1')
    render(<App />)

    await screen.findByText('A toy comes alive.')
    await user.click(screen.getByRole('button', { name: /Find films like this/ }))
    expect(screen.getByRole('status')).toHaveTextContent('Finding films that connect')
    finishRequest({
      status: 'content_recommendations',
      message: 'These are catalog content matches.',
      items: [{ movie_id: 2, title: 'Toy Story 2 (1999)', content_rank: 1, explanation: 'Shared genres: adventure and animation.' }],
    })

    expect(await screen.findByText('Shared genres: adventure and animation.')).toBeInTheDocument()
    expect(screen.getByText(/not probabilities or a promise of relevance/i)).toBeInTheDocument()
  })

  it('sends selected favorites using the backend field contract and renders results', async () => {
    const user = userEvent.setup()
    api.getFavoriteRecommendations.mockResolvedValue({
      status: 'content_based_from_selected_movies',
      message: 'Based on selected titles.',
      items: [{ movie_id: 2, title: 'Toy Story 2 (1999)', best_seed_rank: 1, seed_explanations: { 1: 'Shared director: John Lasseter.' } }],
    })
    render(<App />)
    await user.click(screen.getByRole('button', { name: 'For you' }))
    await user.type(screen.getByRole('searchbox', { name: 'Search for a favorite movie' }), 'toy')
    await user.click(await screen.findByRole('button', { name: /Toy Story \(1995\).*Animation/ }))
    await user.click(screen.getByRole('button', { name: /Find recommendations/ }))

    await waitFor(() => expect(api.getFavoriteRecommendations).toHaveBeenCalledWith([1], 8))
    expect(await screen.findByText('Shared director: John Lasseter.')).toBeInTheDocument()
    expect(screen.getByText(/dataset identifier, not an account login/i)).toBeInTheDocument()
  })

  it('fetches and displays recommendations for an existing MovieLens user ID', async () => {
    const user = userEvent.setup()
    api.getUserRecommendations.mockResolvedValue({
      status: 'collaborative_recommendations',
      message: 'Based on your user history.',
      items: [
        { movie_id: 3, title: 'Other Film (2000)', fusion_score: 0.88, collaborative_rank: 1, explanation: 'High rating predicted by collaborative filtering.' },
      ],
    })
    render(<App />)
    await user.click(screen.getByRole('button', { name: 'For you' }))
    const input = screen.getByLabelText('MovieLens user ID')
    await user.type(input, '42')
    await user.click(screen.getByRole('button', { name: /Get picks/ }))

    expect(await screen.findByText('Other Film (2000)')).toBeInTheDocument()
    expect(screen.getByText('High rating predicted by collaborative filtering.')).toBeInTheDocument()
    expect(api.getUserRecommendations).toHaveBeenCalledWith('42', 8)
  })

  it('handles user recommendation error gracefully when user is unknown', async () => {
    const user = userEvent.setup()
    api.getUserRecommendations.mockRejectedValue(new Error('Unknown user_id: 9999'))
    render(<App />)
    await user.click(screen.getByRole('button', { name: 'For you' }))
    const input = screen.getByLabelText('MovieLens user ID')
    await user.type(input, '9999')
    await user.click(screen.getByRole('button', { name: /Get picks/ }))

    expect(await screen.findByText('Unknown user_id: 9999')).toBeInTheDocument()
    expect(screen.getByText(/We couldn’t load recommendations/i)).toBeInTheDocument()
  })

  it('renders fallback notes when plot and credits are missing', async () => {
    const sparseMovie = {
      movie_id: 99,
      title: 'Sparse Film (2020)',
      release_year: 2020,
      genres: [],
      plot: null,
      plot_tags: [],
      directors: [],
      writers: [],
      cast: [],
      selected_crew: [],
    }
    api.getMovie.mockResolvedValue(sparseMovie)
    window.history.replaceState({}, '', '/?movie=99')
    render(<App />)

    expect(await screen.findByRole('heading', { name: 'Sparse Film (2020)' })).toBeInTheDocument()
    expect(screen.getByText('A plot summary is not available for this title in the catalog.')).toBeInTheDocument()
    expect(screen.getByText('Credits are not available for this movie in the catalog.')).toBeInTheDocument()
  })

  it('allows removing a selected favorite movie before submitting', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: 'For you' }))
    await user.type(screen.getByRole('searchbox', { name: 'Search for a favorite movie' }), 'toy')
    await user.click(await screen.findByRole('button', { name: /Toy Story \(1995\).*Animation/ }))

    expect(screen.getByText(/Find recommendations · 1/)).toBeInTheDocument()
    const removeBtn = screen.getByRole('button', { name: 'Remove Toy Story (1995)' })
    await user.click(removeBtn)

    expect(screen.getByText('Your selected movies will appear here.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Find recommendations/ })).toBeDisabled()
  })

  it('navigates back to discover view when clicking back link', async () => {
    const user = userEvent.setup()
    api.getMovie.mockResolvedValue(sampleMovie)
    window.history.replaceState({}, '', '/?movie=1')
    render(<App />)

    expect(await screen.findByRole('heading', { name: 'Toy Story (1995)' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '← Back to discover' }))

    expect(screen.getByRole('heading', { name: /Good stories/i })).toBeInTheDocument()
    expect(window.location.search).toBe('')
  })

  it('navigates back to personalized view when opened with view=personalized', async () => {
    const user = userEvent.setup()
    api.getMovie.mockResolvedValue(sampleMovie)
    window.history.replaceState({}, '', '/?view=personalized&movie=1')
    render(<App />)

    expect(await screen.findByRole('heading', { name: 'Toy Story (1995)' })).toBeInTheDocument()
    const backBtn = screen.getByRole('button', { name: '← Back to For you' })
    await user.click(backBtn)

    expect(await screen.findByRole('heading', { name: /Your taste/i })).toBeInTheDocument()
    expect(window.location.search).toBe('?view=personalized')
  })

  it('renders an empty result note when movie search yields no items', async () => {
    const user = userEvent.setup()
    api.searchMovies.mockResolvedValue({ items: [], count: 0 })
    render(<App />)

    await user.type(screen.getByRole('searchbox', { name: 'Search the movie catalog' }), 'nomatchfilm')
    expect(await screen.findByText('No matching titles found.')).toBeInTheDocument()
  })
})
