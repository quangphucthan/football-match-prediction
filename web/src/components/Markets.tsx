import { pct } from '../format'
import type { Markets as MarketsData } from '../types'

function Row({ label, v }: { label: string; v: number }) {
  return (
    <div className="row">
      <span>{label}</span>
      <span className="track">
        <i style={{ width: `${v * 100}%` }} />
      </span>
      <span className="v">{pct(v)}%</span>
    </div>
  )
}

export function Markets({
  markets,
  home,
  away,
}: {
  markets: MarketsData
  home: string
  away: string
}) {
  return (
    <div className="panel p-markets">
      <h2 className="ph">Goal markets</h2>
      <div className="mk">
        <Row label="Over 1.5 goals" v={markets.over_1_5} />
        <Row label="Over 2.5 goals" v={markets.over_2_5} />
        <Row label="Over 3.5 goals" v={markets.over_3_5} />
        <Row label="Both teams score" v={markets.btts} />
        <Row label={`${home} clean sheet`} v={markets.home_clean_sheet} />
        <Row label={`${away} clean sheet`} v={markets.away_clean_sheet} />
      </div>
    </div>
  )
}
