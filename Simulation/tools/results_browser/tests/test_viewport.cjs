// Run with: node tools/results_browser/tests/test_viewport.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../results_browser.html'), 'utf8');
const source = html.match(/function limitViewport\([\s\S]*?\n\}/)[0];
const limit = vm.runInNewContext(source + '\nlimitViewport');
const check = (request, min, max, step, expected) =>
  assert.deepEqual(Array.from(limit(request, min, max, step)), expected);
check([-100, 100], 0, 24, 1, [0, 24]); // Maximum zoom-out.
check([5, 5.01], 0, 24, 1, [5, 6]); // Minimum one hour.
check([-10, -6], 0, 24, 1, [0, 4]); // Pan past start keeps width.
check([25, 29], 0, 24, 1, [20, 24]); // Pan past end keeps width.
check([23.9, 24], 0, 24, 1, [23, 24]);
check([0, 100], 0, 0.5, 1, [0, 0.5]); // Short runs stay within data.
const start = 1600000000000;
check([start, start + 10], start, start + 86400000, 3600000,
      [start, start + 3600000]); // Calendar time uses milliseconds.
new vm.Script(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
const wheelSource = html.match(/function wheelViewport\([\s\S]*?\n\}/)[0];
const wheel = vm.runInNewContext(source + '\n' + wheelSource + '\nwheelViewport');
assert.deepEqual(Array.from(wheel([5, 9], 0, 24, 1, 300, 0, .5)), [7, 11]);
assert.deepEqual(Array.from(wheel([5, 9], 0, 24, 1, 0, -300, .5, true)), [3, 7]);
assert.deepEqual(Array.from(wheel([20, 24], 0, 24, 1, 600, 0, .5)), [20, 24]);
assert.deepEqual(Array.from(wheel([5, 6], 0, 24, 1, 0, -100, .5)), [5, 6]);
assert.equal(wheel([5, 9], 0, 24, 1, 0, -100, .5)[1] -
             wheel([5, 9], 0, 24, 1, 0, -100, .5)[0] < 4, true);
// Repeated wheel input paints all charts only once per animation frame.
let scheduled = [], paints = 0;
const updateSource = html.match(/function updateViewport\([\s\S]*?\n\}/)[0];
const context = vm.createContext({requestAnimationFrame: fn => { scheduled.push(fn); return 1; }});
vm.runInContext(source + '\n' + updateSource + `
let view=null, paintFrame=null;
let chartViews=[{bounds:[0,24],minimum:0,maximum:24,minimumSpan:1,paint:()=>recordPaint()}];
`, context);
context.recordPaint = () => paints++;
vm.runInContext('updateViewport([2,6]);updateViewport([3,7]);updateViewport([4,8]);', context);
assert.equal(scheduled.length, 1);
scheduled[0]();
assert.equal(paints, 1);
assert.deepEqual(Array.from(vm.runInContext('chartViews[0].bounds', context)), [4, 8]);
const colorSource = html.match(/function runColors\([\s\S]*?\n\}/)[0];
const runColors = vm.runInNewContext(colorSource + '\nrunColors');
assert.equal(runColors([]).size, 0);
for (const count of [1, 2, 3, 8, 9, 20, 276]) {
  const runs = Array.from({length: count}, (_, i) => ({id: `run_${i}`}));
  const colors = runColors(runs);
  assert.equal(colors.size, count);
  assert.equal(new Set(colors.values()).size, count);
  const hues = Array.from(colors.values(), c => Number(c.match(/hsl\(([^,]+)/)[1]))
    .sort((a, b) => a - b);
  for (let i = 0; i < count; i++) {
    const next = i + 1 < count ? hues[i + 1] : hues[0] + 360;
    assert.ok(Math.abs(next - hues[i] - 360 / count) < 1e-9);
  }
}
console.log('Viewport, input, painting, unique maximally spaced hues and syntax: passed');
