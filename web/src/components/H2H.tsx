import type { H2H as H2HData } from '../types'

export function H2H({ h2h, home, away }: { h2h: H2HData; home: string; away: string }) {
  return (
    <div>
      <div className="played">
        Head to head · {h2h.played} meeting{h2h.played === 1 ? '' : 's'} since 2000
      </div>
      {h2h.played ? (
        <>
          <div className="h2hbar">
            <i style={{ width: `${(h2h.home_wins / h2h.played) * 100}%`, background: 'var(--home)' }} />
            <i style={{ width: `${(h2h.draws / h2h.played) * 100}%`, background: 'var(--ink-faint)' }} />
            <i style={{ width: `${(h2h.away_wins / h2h.played) * 100}%`, background: 'var(--away)' }} />
          </div>
          <div className="h2hcount">
            <span>
              {home} {h2h.home_wins}
            </span>
            <span>{h2h.draws} drawn</span>
            <span>
              {away} {h2h.away_wins}
            </span>
          </div>
          <div className="recent" style={{ marginTop: 12 }}>
            {h2h.recent.map((r) => (
              <div className="r" key={`${r.date}-${r.home}`}>
                <span>{r.date}</span>
                <span>
                  {r.home} v {r.away}
                </span>
                <span className="sc">{r.score}</span>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="hint">Never played since 2000.</div>
      )}
    </div>
  )
}
