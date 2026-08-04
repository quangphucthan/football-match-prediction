import { FLAGS } from '../flags'

/**
 * The team's flag, falling back to its colour swatch for the handful of sides
 * flag-icons has no flag for -- Zanzibar, Tibet, Kurdistan, Northern Cyprus,
 * Chagos Islands, Sint Eustatius.
 *
 * Identity only. Which side is home is carried by the --home / --away tokens,
 * never by this: see CLAUDE.md on why flag colours cannot encode data.
 *
 * Decorative on purpose -- the team name is adjacent text in both call sites,
 * so announcing the flag as well would just repeat it.
 */
export function Flag({ team, colour }: { team: string; colour: string }) {
  const code = FLAGS[team]
  return code ? (
    <i className={`swatch fi-${code}`} title={team} aria-hidden="true" />
  ) : (
    <i className="swatch" style={{ backgroundColor: colour }} title={team} aria-hidden="true" />
  )
}
