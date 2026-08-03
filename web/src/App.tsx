import { useState } from 'react'
import { usePrediction, useTeams } from './api'
import { Form } from './components/Form'
import { H2H } from './components/H2H'
import { Markets } from './components/Markets'
import { OutcomeBar } from './components/OutcomeBar'
import { ScoreGrid, TopScorelines } from './components/ScoreGrid'
import { TeamPicker } from './components/TeamPicker'

/** Shortcuts to fixtures worth looking at: a derby, a classic, a mismatch. */
const FIXTURES: [string, string][] = [
  ['England', 'Germany'],
  ['Brazil', 'France'],
  ['Brazil', 'San Marino'],
  ['Japan', 'Australia'],
]

export default function App() {
  const { teams, tournaments, error: teamsError } = useTeams()
  const [home, setHome] = useState(FIXTURES[0][0])
  const [away, setAway] = useState(FIXTURES[0][1])
  const [neutral, setNeutral] = useState(false)
  const [tournament, setTournament] = useState('Friendly')

  const team = (name: string) => teams.find((t) => t.name === name)
  // Only send names that exist, so half-typed input never becomes a 422.
  const known = (name: string) => (team(name) ? name : '')
  const { data, loading, error } = usePrediction(known(home), known(away), neutral, tournament)

  // Each match is counted once per side.
  const played = teams.reduce((n, t) => n + t.played, 0) / 2

  const status =
    teamsError ?? error ?? (loading ? 'Predicting…' : data ? 'Live model output' : '')

  return (
    <div className="wrap">
      <div className="masthead">
        <h1>
          <i className="fa-solid fa-futbol mark" aria-hidden="true" /> Match Predictor
        </h1>
        <div className="provenance">
          {played ? played.toLocaleString() : '—'} internationals · 2000–2026
        </div>
      </div>

      <div className="fixture">
        <div className="picks">
          <TeamPicker
            id="home"
            label="Home"
            value={home}
            colour={team(home)?.color ?? 'var(--ink-faint)'}
            onChange={setHome}
          />
          <button
            className="swap"
            title="Swap home and away"
            aria-label="Swap home and away"
            onClick={() => {
              setHome(away)
              setAway(home)
            }}
          >
            <i className="fa-solid fa-right-left" aria-hidden="true" />
          </button>
          <TeamPicker
            id="away"
            label="Away"
            value={away}
            colour={team(away)?.color ?? 'var(--ink-faint)'}
            onChange={setAway}
          />
        </div>
        <datalist id="teams">
          {teams.map((t) => (
            <option value={t.name} key={t.name} />
          ))}
        </datalist>
        <div className="opts">
          <i className="fa-solid fa-trophy opt-icon" aria-hidden="true" />
          <select
            aria-label="Competition"
            value={tournament}
            onChange={(e) => setTournament(e.target.value)}
          >
            {tournaments.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
          <label className="toggle">
            <input
              type="checkbox"
              checked={neutral}
              onChange={(e) => setNeutral(e.target.checked)}
            />
            <i className="fa-solid fa-location-dot opt-icon" aria-hidden="true" /> Neutral venue
          </label>
          <span className="spacer" />
          <span className="hint">{status}</span>
        </div>
      </div>

      <div className="shortcuts">
        {FIXTURES.map(([h, a]) => (
          <button
            className="chip"
            key={`${h}|${a}`}
            aria-pressed={h === home && a === away}
            onClick={() => {
              setHome(h)
              setAway(a)
            }}
          >
            {h} v {a}
          </button>
        ))}
      </div>

      {data ? (
        <>
          <OutcomeBar
            outcome={data.outcome}
            expected={data.expected_goals}
            home={data.home}
            away={data.away}
          />
          <div className="panels">
            <ScoreGrid grid={data.grid} home={data.home} away={data.away} />
            <TopScorelines scorelines={data.scorelines} />
            <Markets markets={data.markets} home={data.home} away={data.away} />
            <div className="panel p-history">
              <h2 className="ph">History</h2>
              <div className="hsplit">
                <H2H h2h={data.h2h} home={data.home} away={data.away} />
                <Form form={data.form} home={data.home} away={data.away} />
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="empty">
          {loading ? 'Fitting the model…' : 'Pick two teams to see a prediction.'}
        </div>
      )}

      <div className="footnote">
        <strong>How this is built.</strong> Outcome probabilities are a geometric blend of a
        Poisson goal model and XGBoost; the scoreline grid is rescaled onto those margins, so
        every number here derives from one consistent object. Cards, possession and shot data
        are absent from the dataset, so nothing here shows them.
      </div>
    </div>
  )
}
