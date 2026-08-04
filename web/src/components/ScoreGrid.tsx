import { Fragment } from 'react'
import { pct } from '../format'
import { peak, window95 } from '../grid'
import type { Scoreline } from '../types'

const tint = (i: number, j: number) =>
  i > j ? 'var(--home)' : i < j ? 'var(--away)' : 'var(--ink-faint)'

export function ScoreGrid({
  grid,
  home,
  away,
}: {
  grid: number[][]
  home: string
  away: string
}) {
  const top = peak(grid)
  const n = window95(grid)
  const axis = Array.from({ length: n + 1 }, (_, k) => k)

  return (
    <div className="panel p-matrix">
      <h2 className="ph">Scoreline probability</h2>
      <div className="matrixscroll">
        {/* Cell size is capped: left to 1fr the cells ballooned to ~110px and
            the panel towered over everything beside it. */}
        <div
          className="matrix"
          style={{
            gridTemplateColumns: `20px repeat(${n + 1},minmax(26px,40px))`,
            justifyContent: 'start',
          }}
        >
          <div className="ax" />
          {axis.map((j) => (
            <div className="ax" key={`head${j}`}>
              {j}
            </div>
          ))}
          {axis.map((i) => (
            <Fragment key={i}>
              <div className="ax">{i}</div>
              {axis.map((j) => {
                const p = grid[i][j]
                const a = Math.min(1, Math.pow(p / top.p, 0.55))
                return (
                  <div
                    key={j}
                    className={`cell${i === top.i && j === top.j ? ' peak' : ''}`}
                    style={{
                      background: `color-mix(in srgb, ${tint(i, j)} ${(a * 82).toFixed(1)}%, var(--inset))`,
                    }}
                    title={`${i}-${j} · ${pct(p)}%`}
                  >
                    {p >= 0.02 ? pct(p) : ''}
                  </div>
                )
              })}
            </Fragment>
          ))}
        </div>
      </div>
      <div className="matrixlegend">
        <span className="axtitle">
          ↓ {home} &nbsp;·&nbsp; → {away}
        </span>
        <span className="key">
          <i style={{ background: 'var(--home)' }} /> home
          <i style={{ background: 'var(--ink-faint)' }} /> draw
          <i style={{ background: 'var(--away)' }} /> away
        </span>
      </div>
    </div>
  )
}

export function TopScorelines({ scorelines }: { scorelines: Scoreline[] }) {
  const topP = scorelines[0].p
  return (
    <div className="panel p-scores">
      <h2 className="ph">Most likely scores</h2>
      <div className="sl">
        {scorelines.map((s) => {
          const [i, j] = s.score.split('-').map(Number)
          return (
            <div className="row" key={s.score}>
              <span className="s">{s.score}</span>
              <span className="track">
                <i style={{ width: `${(s.p / topP) * 100}%`, background: tint(i, j) }} />
              </span>
              <span className="p">{pct(s.p)}%</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
