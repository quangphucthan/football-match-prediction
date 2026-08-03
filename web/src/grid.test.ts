import test from 'node:test'
import assert from 'node:assert/strict'
import { peak, window95 } from './grid.ts'

/** 11x11 of zeros with the given cells filled: [row, col, p]. */
const grid = (...cells: [number, number, number][]) => {
  const g = Array.from({ length: 11 }, () => Array(11).fill(0))
  for (const [i, j, p] of cells) g[i][j] = p
  return g
}

test('peak finds the most likely scoreline', () => {
  const { i, j } = peak(grid([1, 1, 0.3], [6, 0, 0.4]))
  assert.deepEqual([i, j], [6, 0])
})

test('window never crops below 0-5', () => {
  assert.equal(window95(grid([1, 1, 0.6], [0, 0, 0.4])), 5)
})

test('window reaches a peak outside 0-5', () => {
  // The Brazil v San Marino shape: everything sits at 6-0.
  assert.ok(window95(grid([6, 0, 1])) >= 6)
})

test('window grows until it covers the mass', () => {
  assert.equal(window95(grid([1, 1, 0.5], [8, 8, 0.5])), 8)
})

test('window stays inside the grid even when the mass never adds up', () => {
  assert.equal(window95(grid([0, 0, 0.1])), 10)
})
