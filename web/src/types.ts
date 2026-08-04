/**
 * Mirrors the payload built by Model.predict in model.py.
 *
 * Hand-written on purpose: two endpoints do not justify openapi-typescript
 * codegen. Add it if the API passes ~5 endpoints and drift becomes real.
 */

export type Team = { name: string; color: string; played: number }
/** One <optgroup>: a confederation (or Friendly/World), tournaments A-Z inside. */
export type TournamentGroup = { region: string; tournaments: string[] }
export type TeamsResponse = { teams: Team[]; tournaments: TournamentGroup[] }

export type Outcome = { home: number; draw: number; away: number }
export type Scoreline = { score: string; p: number }

export type Markets = {
  over_1_5: number
  over_2_5: number
  over_3_5: number
  btts: number
  home_clean_sheet: number
  away_clean_sheet: number
}

export type H2HMatch = {
  date: string
  home: string
  away: string
  score: string
  neutral: boolean
}

export type H2H = {
  played: number
  home_wins: number
  draws: number
  away_wins: number
  recent: H2HMatch[]
}

export type Result = 'W' | 'D' | 'L'

export type Prediction = {
  home: string
  away: string
  neutral: boolean
  tournament: string
  outcome: Outcome
  expected_goals: { home: number; away: number }
  scorelines: Scoreline[]
  markets: Markets
  /** Full 11x11. Row index = home goals, column = away goals. */
  grid: number[][]
  h2h: H2H
  form: { home: Result[]; away: Result[] }
}
