/** Scoreline-grid geometry. Kept out of the .tsx so `node --test` can run it. */

/** Row (home goals), column (away goals) and value of the most likely score. */
export function peak(grid: number[][]) {
  let i = 0
  let j = 0
  let p = 0
  grid.forEach((row, r) =>
    row.forEach((v, c) => {
      if (v > p) {
        p = v
        i = r
        j = c
      }
    }),
  )
  return { i, j, p }
}

/**
 * Largest goal index to display: the smallest square 0..n holding `mass` of the
 * probability. The API sends the whole 11x11 because lopsided fixtures sit well
 * outside 0-5 -- Brazil v San Marino peaks at 6-0. A peak-plus-two rule left a
 * fifth of the mass off-screen, hence covering an explicit share instead.
 */
export function window95(grid: number[][], mass = 0.95) {
  const { i, j } = peak(grid)
  const covered = (k: number) => {
    let s = 0
    for (let r = 0; r <= k; r++) for (let c = 0; c <= k; c++) s += grid[r][c]
    return s
  }
  let n = Math.max(5, i, j)
  while (n < grid.length - 1 && covered(n) < mass) n++
  return n
}
