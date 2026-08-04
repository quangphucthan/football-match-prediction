import { useEffect, useMemo, useRef, useState } from 'react'
import type { Team } from '../types'
import { Flag } from './Flag'

type Props = {
  id: string
  label: string
  value: string
  teams: Team[]
  onChange: (name: string) => void
}

/**
 * Searchable team picker.
 *
 * Hand-rolled rather than a combobox library, and no longer a native
 * <datalist>: the datalist popup is drawn by the browser, so its width,
 * position and height cannot be styled at all -- it renders as a narrow list
 * detached from the field. This is ~70 lines and gives a panel that matches the
 * input's width, sits directly under it, and scrolls after VISIBLE_ROWS.
 */

/** Rows shown before the list scrolls. Paired with --row in styles.css. */
const VISIBLE_ROWS = 8

export function TeamPicker({ id, label, value, teams, onChange }: Props) {
  // null means "not searching" -- the input shows the current selection.
  const [query, setQuery] = useState<string | null>(null)
  const [active, setActive] = useState(0)
  const listRef = useRef<HTMLUListElement>(null)
  const open = query !== null

  const matches = useMemo(() => {
    if (query === null) return []
    const q = query.trim().toLowerCase()
    return q ? teams.filter((t) => t.name.toLowerCase().includes(q)) : teams
  }, [query, teams])

  // Keep the highlighted row on screen when arrowing past the fold.
  useEffect(() => {
    listRef.current?.children[active]?.scrollIntoView({ block: 'nearest' })
  }, [active])

  const close = () => setQuery(null)

  const pick = (name: string) => {
    onChange(name)
    close()
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') return close()
    if (e.key === 'Enter' && open && matches[active]) {
      e.preventDefault()
      return pick(matches[active].name)
    }
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
    e.preventDefault()
    if (!open) return setQuery('')
    const step = e.key === 'ArrowDown' ? 1 : -1
    setActive((i) => Math.min(Math.max(i + step, 0), matches.length - 1))
  }

  return (
    <div className="field">
      <label className="cap" htmlFor={id}>
        {label}
      </label>
      {/* focusout rather than input blur, so clicks inside the list do not close it */}
      <div
        className="inputrow"
        onBlur={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node)) close()
        }}
      >
        <Flag team={value} colour={teams.find((t) => t.name === value)?.color ?? 'var(--ink-faint)'} />
        <input
          id={id}
          role="combobox"
          aria-expanded={open}
          aria-controls={`${id}-options`}
          aria-activedescendant={open && matches[active] ? `${id}-opt-${active}` : undefined}
          aria-autocomplete="list"
          autoComplete="off"
          value={query ?? value}
          placeholder={value}
          onChange={(e) => {
            setQuery(e.target.value)
            setActive(0)
          }}
          onFocus={() => setQuery('')}
          // Also on click: after picking, focus never left the input, so focus
          // alone would not fire again and the field would look dead.
          onClick={() => query === null && setQuery('')}
          onKeyDown={onKeyDown}
        />
        {open && (
          // Holding the mousedown stops the input losing focus before the click lands.
          <ul
            className="options"
            id={`${id}-options`}
            role="listbox"
            ref={listRef}
            style={{ maxHeight: `calc(${VISIBLE_ROWS} * var(--row) + 8px)` }}
            onMouseDown={(e) => e.preventDefault()}
          >
            {matches.map((t, i) => (
              <li
                key={t.name}
                id={`${id}-opt-${i}`}
                role="option"
                aria-selected={i === active}
                className="option"
                onMouseMove={() => setActive(i)}
                onClick={() => pick(t.name)}
              >
                <Flag team={t.name} colour={t.color} />
                {t.name}
              </li>
            ))}
            {matches.length === 0 && <li className="option empty-option">No team matches</li>}
          </ul>
        )}
      </div>
    </div>
  )
}
