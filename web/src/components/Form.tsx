import type { Result } from '../types'

function Dots({ form }: { form: Result[] }) {
  return (
    <span className="dots">
      {form.map((r, i) => (
        <i className={`dot ${r}`} key={i}>
          {r}
        </i>
      ))}
    </span>
  )
}

export function Form({
  form,
  home,
  away,
}: {
  form: { home: Result[]; away: Result[] }
  home: string
  away: string
}) {
  return (
    <div>
      <div className="played">Recent form · newest first</div>
      <div className="formrow">
        <span className="nm">{home}</span>
        <Dots form={form.home} />
      </div>
      <div className="formrow">
        <span className="nm">{away}</span>
        <Dots form={form.away} />
      </div>
    </div>
  )
}
