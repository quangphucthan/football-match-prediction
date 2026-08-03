import { pct } from '../format'
import type { Outcome } from '../types'

type Props = {
  outcome: Outcome
  expected: { home: number; away: number }
  home: string
  away: string
}

export function OutcomeBar({ outcome, expected, home, away }: Props) {
  return (
    <div className="verdict">
      <div className="odds">
        <div className="home">
          <div className="pct">{pct(outcome.home)}%</div>
          <div className="who">{home}</div>
        </div>
        <div>
          <div className="pct">{pct(outcome.draw)}%</div>
          <div className="who">Draw</div>
        </div>
        <div className="away">
          <div className="pct">{pct(outcome.away)}%</div>
          <div className="who">{away}</div>
        </div>
      </div>
      <div className="bar">
        <i className="h" style={{ width: `${outcome.home * 100}%` }} />
        <i className="d" style={{ width: `${outcome.draw * 100}%` }} />
        <i className="a" style={{ width: `${outcome.away * 100}%` }} />
      </div>
      <div className="xg">
        <span className="lab">Expected goals</span>
        <b style={{ color: 'var(--home)' }}>{expected.home.toFixed(2)}</b>
        <span className="lab">—</span>
        <b style={{ color: 'var(--away)' }}>{expected.away.toFixed(2)}</b>
      </div>
    </div>
  )
}
