import { useEffect, useState } from 'react'
import type { Prediction, TeamsResponse } from './types'

/**
 * Two endpoints, so two hooks and no query library -- see CLAUDE.md. Reach for
 * TanStack Query when the endpoint count passes ~4.
 */

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  const body = await res.json().catch(() => null)
  // FastAPI puts validation failures in `detail`; surface that, not "422".
  if (!res.ok) throw new Error(body?.detail ?? `${res.status} ${res.statusText}`)
  return body as T
}

/** Fetched once at mount. The list is static for the life of the process. */
export function useTeams() {
  const [data, setData] = useState<TeamsResponse>({ teams: [], tournaments: [] })
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    json<TeamsResponse>('/api/teams').then(setData).catch((e) => setError(e.message))
  }, [])

  return { ...data, error }
}

export function usePrediction(
  home: string,
  away: string,
  neutral: boolean,
  tournament: string,
) {
  const [data, setData] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Callers pass "" until a name matches a real team, so half-typed input
    // never reaches the API. The model rejects a team playing itself.
    if (!home || !away || home === away) {
      setData(null)
      setError(home && home === away ? 'A team cannot play itself.' : null)
      return
    }

    const ctl = new AbortController()
    setLoading(true)
    setError(null)
    json<Prediction>('/api/predict', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ home, away, neutral, tournament }),
      signal: ctl.signal,
    })
      .then((p) => {
        setData(p)
        setLoading(false)
      })
      .catch((e: Error) => {
        if (e.name === 'AbortError') return // superseded by a newer fixture
        setData(null)
        setError(e.message)
        setLoading(false)
      })

    return () => ctl.abort()
  }, [home, away, neutral, tournament])

  return { data, loading, error }
}
