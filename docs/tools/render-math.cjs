// Edit data-tex attributes in index.html, then run: node tools/render-math.cjs
// Static HTML + MathML output needs no browser-side math script.
const fs = require('node:fs');
const path = require('node:path');
const katex = require('../assets/katex/katex.min.js');
const file = path.join(__dirname, '../index.html');
let html = fs.readFileSync(file, 'utf8');
// Each generated formula ends at a marker, allowing safe regeneration.
html = html.replace(/(<span class="math [^"]*" data-display="(true|false)" data-tex="([^"]*)">)(?:<\/span>(?!<!--\/math-->)|[\s\S]*?<!--\/math-->)/g,
 (_, opening, display, encoded) => {
  const tex = encoded.replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  return opening + katex.renderToString(tex, {displayMode: display === 'true', throwOnError: true, strict: 'error', output: 'htmlAndMathml', trust: false}) + '</span><!--/math-->';
 });
fs.writeFileSync(file, html);
console.log('Rendered all data-tex formulas to static HTML and accessible MathML.');
