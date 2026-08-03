type Props = {
  id: string
  label: string
  value: string
  colour: string
  onChange: (name: string) => void
}

/**
 * Native <datalist> does the filtering over 237 teams, so no combobox library.
 * The swatch is the team's real flag colour and is identity only -- data
 * encoding uses the --home / --away tokens.
 */
export function TeamPicker({ id, label, value, colour, onChange }: Props) {
  return (
    <div className="field">
      <label className="cap" htmlFor={id}>
        {label}
      </label>
      <div className="inputrow">
        <i className="swatch" style={{ background: colour }} />
        <input
          id={id}
          list="teams"
          autoComplete="off"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      </div>
    </div>
  )
}
