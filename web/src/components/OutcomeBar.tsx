import { pct } from '../format'
import type { Outcome } from '../types'
import { Flag } from './Flag'

type Props = {
  outcome: Outcome
  expected: { home: number; away: number }
  home: string
  away: string
  homeColour: string
  awayColour: string
}

export function OutcomeBar({ outcome, expected, home, away, homeColour, awayColour }: Props) {
  return (
    <div className="verdict">
      <div className="odds">
        <div className="home">
          <div className="pct">{pct(outcome.home)}%</div>
          {/* Identity only -- it never encodes which side is which. 41% of team
              colours fall under 3:1 against the dark panel and a third of
              fixtures pair two indistinguishable ones, so the bar below stays
              on the fixed --home / --away tokens. */}
          <div className="who">
            <Flag team={home} colour={homeColour} />
            {home}
          </div>
        </div>
        <div>
          <div className="pct">{pct(outcome.draw)}%</div>
          <div className="who">Draw</div>
        </div>
        <div className="away">
          <div className="pct">{pct(outcome.away)}%</div>
          <div className="who">
            <Flag team={away} colour={awayColour} />
            {away}
          </div>
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
